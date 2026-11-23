"""
Paso 2 y 3: ejecutar los scripts SQL sobre la base de datos.
- sql/02_variables.sql crea la tabla 'clientes_modelo'.
- sql/03_exploracion.sql muestra la tasa de default por decil de utilización.
"""
import sqlite3
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DB = RAIZ / "data" / "credito.db"
SQL = RAIZ / "sql"


def main():
    with sqlite3.connect(DB) as con:
        # executescript corre un archivo .sql completo (varias sentencias)
        con.executescript((SQL / "02_variables.sql").read_text(encoding="utf-8"))
        n = con.execute("SELECT COUNT(*) FROM clientes_modelo").fetchone()[0]
        print(f"Tabla clientes_modelo creada: {n:,} filas")

        # read_sql corre una consulta y devuelve el resultado como DataFrame
        eda = pd.read_sql((SQL / "03_exploracion.sql").read_text(encoding="utf-8"), con)
    print("\nTasa de default por decil de utilización:")
    print(eda.to_string(index=False))
    eda.to_csv(RAIZ / "reports" / "default_por_decil.csv", index=False)


if __name__ == "__main__":
    main()
