"""
Paso 1: cargar el CSV de Kaggle ("Give Me Some Credit") a una base de datos SQLite.

¿Por qué SQLite? En un banco los datos viven en una base de datos (Oracle, SQL Server...)
y se extraen con SQL. SQLite simula eso en un solo archivo, sin instalar un servidor.
"""
import sqlite3
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]          # carpeta del proyecto
CSV = RAIZ / "data" / "cs-training.csv"              # archivo descargado de Kaggle
DB = RAIZ / "data" / "credito.db"                    # base de datos que vamos a crear

# Nombres originales (inglés, con guiones) -> nombres cortos en español, más fáciles de usar en SQL
RENOMBRAR = {
    "Unnamed: 0": "id_cliente",
    "SeriousDlqin2yrs": "default_2a",                      # 1 = cayó en mora grave (90+ días) en 2 años
    "RevolvingUtilizationOfUnsecuredLines": "utilizacion",  # saldo de tarjetas / línea total
    "age": "edad",
    "NumberOfTime30-59DaysPastDueNotWorse": "atrasos_30_59",
    "DebtRatio": "ratio_deuda",                            # pagos de deuda mensuales / ingreso
    "MonthlyIncome": "ingreso_mensual",
    "NumberOfOpenCreditLinesAndLoans": "lineas_abiertas",
    "NumberOfTimes90DaysLate": "atrasos_90",
    "NumberRealEstateLoansOrLines": "creditos_hipotecarios",
    "NumberOfTime60-89DaysPastDueNotWorse": "atrasos_60_89",
    "NumberOfDependents": "dependientes",
}


def main():
    df = pd.read_csv(CSV)
    df = df.rename(columns=RENOMBRAR)

    # Revisión rápida: tamaño, tasa de default y valores faltantes
    print(f"Filas: {len(df):,}")
    print(f"Tasa de default: {df['default_2a'].mean():.2%}")
    print("Faltantes por columna:\n", df.isna().sum()[df.isna().sum() > 0])

    # Guardamos en la tabla 'clientes_raw'. if_exists='replace' la reescribe si ya existe.
    with sqlite3.connect(DB) as con:
        df.to_sql("clientes_raw", con, if_exists="replace", index=False)
    print(f"Tabla clientes_raw creada en {DB.name}")


if __name__ == "__main__":
    main()
