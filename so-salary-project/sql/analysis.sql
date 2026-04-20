-- Median and mean salary by country, limited to countries with at least 50 rows.
WITH ranked AS (
    SELECT
        Country,
        ConvertedCompYearly,
        ROW_NUMBER() OVER (
            PARTITION BY Country
            ORDER BY ConvertedCompYearly
        ) AS rn,
        COUNT(*) OVER (PARTITION BY Country) AS cnt,
        AVG(ConvertedCompYearly) OVER (PARTITION BY Country) AS mean_salary
    FROM responses
    WHERE ConvertedCompYearly IS NOT NULL
)
SELECT
    Country,
    MAX(cnt) AS n,
    CAST(MAX(mean_salary) AS INT) AS mean_salary,
    CAST(AVG(
        CASE
            WHEN rn IN ((cnt + 1) / 2, (cnt + 2) / 2) THEN ConvertedCompYearly
        END
    ) AS INT) AS median_salary
FROM ranked
GROUP BY Country
HAVING MAX(cnt) >= 50
ORDER BY median_salary DESC
LIMIT 20;

-- SQL vs no-SQL salary gap among data roles.
SELECT
    CASE WHEN lang_sql = 1 THEN 'knows SQL' ELSE 'no SQL' END AS sql_flag,
    COUNT(*) AS n,
    ROUND(AVG(ConvertedCompYearly), 2) AS avg_salary,
    ROUND(AVG(log_salary), 4) AS avg_log_salary
FROM responses
WHERE ConvertedCompYearly IS NOT NULL
  AND is_data_role = 1
GROUP BY sql_flag
ORDER BY avg_salary DESC;

-- Single-skill premium example.
SELECT
    'python' AS skill,
    ROUND(AVG(CASE WHEN lang_python = 1 THEN ConvertedCompYearly END), 2) AS with_skill,
    ROUND(AVG(CASE WHEN lang_python = 0 THEN ConvertedCompYearly END), 2) AS without_skill,
    ROUND(
        (
            AVG(CASE WHEN lang_python = 1 THEN ConvertedCompYearly END) /
            AVG(CASE WHEN lang_python = 0 THEN ConvertedCompYearly END) - 1
        ) * 100,
        2
    ) AS premium_pct
FROM responses
WHERE is_data_role = 1;

-- Benchmarking table for role, experience bucket, and country group.
SELECT
    country_group,
    exp_bucket,
    COUNT(*) AS n,
    ROUND(AVG(ConvertedCompYearly), 2) AS avg_salary
FROM responses
WHERE ConvertedCompYearly IS NOT NULL
GROUP BY country_group, exp_bucket
ORDER BY country_group, exp_bucket;
