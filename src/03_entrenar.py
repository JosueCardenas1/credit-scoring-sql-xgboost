"""
Paso 4: entrenar y evaluar dos modelos de probabilidad de default (PD).
  A) Regresión logística con variables WoE -> el estándar de los bancos (fácil de explicar al regulador).
  B) XGBoost -> modelo de machine learning, normalmente más preciso pero menos interpretable.
Luego: métricas (AUC, Gini, KS), scorecard en puntos y punto de corte.
"""
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # para guardar gráficos sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from woe import TransformadorWoE

RAIZ = Path(__file__).resolve().parents[1]
DB = RAIZ / "data" / "credito.db"
REP = RAIZ / "reports"
SEMILLA = 42  # fija la aleatoriedad para que los resultados se puedan replicar

VARIABLES = [
    "utilizacion", "edad", "ratio_deuda", "ingreso_mensual", "lineas_abiertas",
    "creditos_hipotecarios", "dependientes", "atrasos_30_59", "atrasos_60_89", "atrasos_90",
    "total_atrasos", "alguna_vez_atrasado", "ingreso_faltante", "deuda_mensual", "ingreso_per_capita",
]


# ---------------------------------------------------------------- métricas
def metricas(y, pd_hat):
    """AUC, Gini y KS a partir de la probabilidad de default estimada."""
    auc = roc_auc_score(y, pd_hat)
    fpr, tpr, umbrales = roc_curve(y, pd_hat)
    ks = np.max(tpr - fpr)  # máxima distancia entre distribuciones de malos y buenos
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks}


def umbral_ks(y, pd_hat):
    """Punto de corte donde el KS es máximo (separa mejor buenos de malos)."""
    fpr, tpr, umbrales = roc_curve(y, pd_hat)
    return umbrales[np.argmax(tpr - fpr)]


# ---------------------------------------------------------------- scorecard
def a_puntaje(pd_hat, base=600, odds_base=50, pdo=20):
    """
    Convierte la PD en puntaje tipo buró: 600 puntos = odds buenos:malos de 50:1,
    y cada 20 puntos (PDO) las odds se duplican. Más puntos = menos riesgo.
    """
    factor = pdo / np.log(2)
    offset = base - factor * np.log(odds_base)
    odds_buenos = (1 - pd_hat) / pd_hat
    return offset + factor * np.log(odds_buenos)


def main():
    REP.mkdir(exist_ok=True)

    # 1) Extraer los datos con SQL
    with sqlite3.connect(DB) as con:
        df = pd.read_sql(f"SELECT default_2a, {', '.join(VARIABLES)} FROM clientes_modelo", con)
    X, y = df[VARIABLES], df["default_2a"]

    # 2) Separar entrenamiento (70%) y prueba (30%).
    #    stratify=y mantiene la misma tasa de default en ambas muestras.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEMILLA)
    print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}  Tasa default: {y.mean():.2%}")

    # 3) WoE e IV: se aprenden SOLO con train para no filtrar información del test (data leakage)
    woe = TransformadorWoE(n_bins=5).fit(X_tr, y_tr)
    iv = woe.iv()
    print("\nInformation Value por variable:\n", iv.round(3).to_string())
    iv.rename("IV").to_csv(REP / "information_value.csv")

    # Nos quedamos con variables con IV >= 0.02 (las demás casi no discriminan)
    candidatas = iv[iv >= 0.02].index.tolist()  # ya vienen ordenadas de mayor a menor IV

    # Quitamos variables redundantes: si dos WoE tienen correlación > 0.7, nos quedamos con la
    # de mayor IV. Si no, la multicolinealidad puede dar coeficientes con signo absurdo.
    corr = woe.transform(X_tr[candidatas]).corr().abs()
    seleccion = []
    for var in candidatas:
        if all(corr.loc[var, ya] <= 0.7 for ya in seleccion):
            seleccion.append(var)
    print("\nVariables elegidas para la logística:", seleccion)
    W_tr, W_te = woe.transform(X_tr[seleccion]), woe.transform(X_te[seleccion])

    # 4A) Regresión logística sobre WoE.
    # Con WoE = ln(%buenos/%malos), todos los coeficientes deben ser NEGATIVOS: más WoE = menos
    # riesgo = menor PD. Si alguno sale positivo (señal de redundancia), lo sacamos y reestimamos.
    while True:
        logit = LogisticRegression(max_iter=1000).fit(W_tr[seleccion], y_tr)
        coef = pd.Series(logit.coef_[0], index=seleccion)
        if (coef <= 0).all():
            break
        sale = coef.idxmax()
        print(f"Sale '{sale}' por signo incorrecto (coef = {coef[sale]:.3f})")
        seleccion.remove(sale)
    W_tr, W_te = W_tr[seleccion], W_te[seleccion]
    pd_logit = logit.predict_proba(W_te)[:, 1]  # columna 1 = probabilidad de default
    coefs = pd.Series(logit.coef_[0], index=seleccion).sort_values()
    print("\nCoeficientes de la logística (WoE):\n", coefs.round(3).to_string())

    # 4B) XGBoost sobre las variables originales (maneja los NaN por sí solo)
    xgb = XGBClassifier(
        n_estimators=300,      # número de árboles
        max_depth=4,           # profundidad de cada árbol (controla complejidad)
        learning_rate=0.05,    # cuánto aporta cada árbol nuevo
        subsample=0.8,         # cada árbol ve el 80% de las filas (reduce sobreajuste)
        colsample_bytree=0.8,  # y el 80% de las columnas
        eval_metric="auc",
        random_state=SEMILLA,
    )
    xgb.fit(X_tr, y_tr)
    pd_xgb = xgb.predict_proba(X_te)[:, 1]

    # 5) Comparar modelos en la muestra de prueba
    res = pd.DataFrame({
        "Logística WoE": metricas(y_te, pd_logit),
        "XGBoost": metricas(y_te, pd_xgb),
    }).T
    print("\nMétricas en test:\n", res.round(3).to_string())
    res.to_csv(REP / "metricas_test.csv")

    # 6) Scorecard y punto de corte (para la logística, el modelo "oficial")
    corte = umbral_ks(y_te, pd_logit)
    aprobado = pd_logit < corte  # se aprueba si la PD está por debajo del corte
    tn, fp, fn, tp = confusion_matrix(y_te, ~aprobado).ravel()
    print(f"\nCorte (PD) = {corte:.3f}  -> puntaje de corte ≈ {a_puntaje(corte):.0f}")
    print(f"Tasa de aprobación: {aprobado.mean():.1%}")
    print(f"Default entre aprobados: {y_te[aprobado].mean():.2%}  vs  población: {y_te.mean():.2%}")
    print(f"Malos detectados (rechazados): {tp:,} de {tp + fn:,}  |  Buenos rechazados: {fp:,}")

    pd.DataFrame({"default_real": y_te.values, "pd_logit": pd_logit, "puntaje": a_puntaje(pd_logit),
                  "pd_xgb": pd_xgb}).to_csv(REP / "predicciones_test.csv", index=False)

    # 7) Gráficos
    fig, ax = plt.subplots(figsize=(6, 5))
    for nombre, p in [("Logística WoE", pd_logit), ("XGBoost", pd_xgb)]:
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc_score(y_te, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="Curva ROC (test)")
    ax.legend()
    fig.tight_layout(); fig.savefig(REP / "roc.png", dpi=120); plt.close(fig)

    imp = pd.Series(xgb.feature_importances_, index=VARIABLES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.plot.barh(ax=ax, color="#2f6db3")
    ax.set(title="Importancia de variables (XGBoost)", xlabel="Importancia (gain)")
    fig.tight_layout(); fig.savefig(REP / "importancia_xgb.png", dpi=120); plt.close(fig)
    print("\nResultados guardados en reports/")


if __name__ == "__main__":
    main()
