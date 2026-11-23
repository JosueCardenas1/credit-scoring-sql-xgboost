-- Paso 2: construcción de variables (feature engineering) en SQL.
-- Toma la tabla cruda 'clientes_raw' y crea 'clientes_modelo', lista para modelar.

DROP TABLE IF EXISTS clientes_modelo;

CREATE TABLE clientes_modelo AS
WITH limpio AS (
    -- 1) Limpieza: en este dataset, 96 y 98 en los conteos de atraso son códigos de error,
    --    no clientes con 96 atrasos reales. Los convertimos en NULL (dato faltante).
    SELECT
        id_cliente,
        default_2a,
        edad,
        ingreso_mensual,
        dependientes,
        lineas_abiertas,
        creditos_hipotecarios,
        ratio_deuda,
        -- La utilización debería estar entre 0 y 1; topamos los valores absurdos en 1.5
        CASE WHEN utilizacion > 1.5 THEN 1.5 ELSE utilizacion END              AS utilizacion,
        CASE WHEN atrasos_30_59 >= 96 THEN NULL ELSE atrasos_30_59 END         AS atrasos_30_59,
        CASE WHEN atrasos_60_89 >= 96 THEN NULL ELSE atrasos_60_89 END         AS atrasos_60_89,
        CASE WHEN atrasos_90    >= 96 THEN NULL ELSE atrasos_90    END         AS atrasos_90
    FROM clientes_raw
    WHERE edad >= 18                       -- descartamos edades imposibles (hay una edad 0)
)
SELECT
    l.*,
    -- 2) Variables nuevas a partir de las originales
    COALESCE(atrasos_30_59, 0) + COALESCE(atrasos_60_89, 0) + COALESCE(atrasos_90, 0)
                                                        AS total_atrasos,
    CASE WHEN COALESCE(atrasos_30_59, 0) + COALESCE(atrasos_60_89, 0)
              + COALESCE(atrasos_90, 0) > 0 THEN 1 ELSE 0 END
                                                        AS alguna_vez_atrasado,
    CASE WHEN ingreso_mensual IS NULL THEN 1 ELSE 0 END AS ingreso_faltante,
    -- Deuda mensual en soles/dólares: ratio_deuda * ingreso (solo si hay ingreso)
    CASE WHEN ingreso_mensual > 0 THEN ratio_deuda * ingreso_mensual END
                                                        AS deuda_mensual,
    -- Ingreso por persona en el hogar
    CASE WHEN ingreso_mensual IS NOT NULL
         THEN ingreso_mensual / (COALESCE(dependientes, 0) + 1) END
                                                        AS ingreso_per_capita,
    -- 3) Función de ventana: decil de utilización (1 = los que menos usan su línea)
    NTILE(10) OVER (ORDER BY utilizacion)               AS decil_utilizacion
FROM limpio AS l;
