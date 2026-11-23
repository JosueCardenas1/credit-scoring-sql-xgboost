"""
WoE (Weight of Evidence) e IV (Information Value): las herramientas clásicas de scoring bancario.

Idea:
1) Cada variable se parte en tramos (bins), por ejemplo edad 18-30, 30-40, ...
2) En cada tramo se compara qué porcentaje de los BUENOS y qué porcentaje de los MALOS cae ahí:
       WoE = ln( %buenos_en_tramo / %malos_en_tramo )
   WoE > 0  -> el tramo tiene proporcionalmente más buenos (menor riesgo)
   WoE < 0  -> el tramo tiene proporcionalmente más malos (mayor riesgo)
3) El IV resume cuánto discrimina la variable completa:
       IV = suma sobre tramos de (%buenos - %malos) * WoE
   Regla práctica: <0.02 inútil, 0.02-0.1 débil, 0.1-0.3 media, >0.3 fuerte.
"""
import numpy as np
import pandas as pd

FALTANTE = "Faltante"


def calcular_cortes(x: pd.Series, n_bins: int = 5):
    """Define los cortes de los tramos usando SOLO la muestra de entrenamiento."""
    valores = x.dropna()
    if valores.nunique() <= n_bins:
        return None  # pocas categorías (ej. 0/1): cada valor es su propio tramo
    # qcut parte en cuantiles (tramos con igual número de clientes)
    _, cortes = pd.qcut(valores, q=n_bins, retbins=True, duplicates="drop")
    cortes[0], cortes[-1] = -np.inf, np.inf  # extremos abiertos para valores nuevos en test
    return cortes


def asignar_tramo(x: pd.Series, cortes):
    """Convierte cada valor en la etiqueta de su tramo; los NaN van a un tramo propio."""
    if cortes is None:
        tramo = x.astype("object")
    else:
        tramo = pd.cut(x, bins=cortes).astype("object")
    return tramo.where(x.notna(), FALTANTE).astype(str)


def tabla_woe(tramo: pd.Series, y: pd.Series):
    """Calcula WoE e IV por tramo. y = 1 si es malo (default)."""
    t = pd.DataFrame({"tramo": tramo, "malo": y})
    g = t.groupby("tramo")["malo"].agg(clientes="count", malos="sum")
    g["buenos"] = g["clientes"] - g["malos"]
    # Sumamos 0.5 para evitar ln(0) cuando un tramo no tiene malos o buenos
    pct_buenos = (g["buenos"] + 0.5) / (g["buenos"].sum() + 0.5)
    pct_malos = (g["malos"] + 0.5) / (g["malos"].sum() + 0.5)
    g["tasa_default"] = g["malos"] / g["clientes"]
    g["woe"] = np.log(pct_buenos / pct_malos)
    g["iv_tramo"] = (pct_buenos - pct_malos) * g["woe"]
    return g


class TransformadorWoE:
    """Aprende los tramos y WoE en train (fit) y los aplica a cualquier muestra (transform)."""

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self.cortes, self.tablas = {}, {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for col in X.columns:
            self.cortes[col] = calcular_cortes(X[col], self.n_bins)
            tramo = asignar_tramo(X[col], self.cortes[col])
            self.tablas[col] = tabla_woe(tramo, y)
        return self

    def iv(self) -> pd.Series:
        """IV total de cada variable, de mayor a menor."""
        return pd.Series({c: t["iv_tramo"].sum() for c, t in self.tablas.items()}).sort_values(ascending=False)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        salida = {}
        for col in X.columns:
            tramo = asignar_tramo(X[col], self.cortes[col])
            # Reemplazamos cada tramo por su WoE; un tramo no visto en train recibe 0 (neutral)
            salida[col] = tramo.map(self.tablas[col]["woe"]).fillna(0.0)
        return pd.DataFrame(salida, index=X.index)
