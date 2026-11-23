-- Paso 3: análisis exploratorio con SQL.
-- ¿Cómo cambia la tasa de default según el decil de utilización de la línea?
-- Si sube de forma ordenada, la variable discrimina bien entre buenos y malos pagadores.

SELECT
    decil_utilizacion,
    COUNT(*)                          AS clientes,
    SUM(default_2a)                   AS defaults,
    ROUND(AVG(default_2a) * 100, 2)   AS tasa_default_pct,
    ROUND(MIN(utilizacion), 3)        AS util_min,
    ROUND(MAX(utilizacion), 3)        AS util_max
FROM clientes_modelo
GROUP BY decil_utilizacion
ORDER BY decil_utilizacion;
