from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "survey_clean.parquet"
SQLITE_PATH = PROJECT_ROOT / "data" / "processed" / "survey.db"


def main() -> None:
    df = pd.read_parquet(PARQUET_PATH)

    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(SQLITE_PATH) as conn:
        df.to_sql("responses", conn, if_exists="replace", index=False)
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_responses_country ON responses (Country);
            CREATE INDEX IF NOT EXISTS idx_responses_data_role ON responses (is_data_role);
            CREATE INDEX IF NOT EXISTS idx_responses_salary ON responses (ConvertedCompYearly);
            CREATE INDEX IF NOT EXISTS idx_responses_country_group ON responses (country_group);
            CREATE INDEX IF NOT EXISTS idx_responses_exp_bucket ON responses (exp_bucket);
            """
        )

        row_count = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    print(f"Loaded {row_count:,} rows into {SQLITE_PATH}")


if __name__ == "__main__":
    main()
