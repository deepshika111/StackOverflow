from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "survey_results_public.csv"
PROCESSED_PARQUET = PROJECT_ROOT / "data" / "processed" / "survey_clean.parquet"
MULTISELECT_METADATA = PROJECT_ROOT / "data" / "processed" / "multiselect_reference.json"
ETL_SUMMARY = PROJECT_ROOT / "outputs" / "etl_summary.json"

SALARY_COLUMN = "ConvertedCompYearly"
SALARY_MIN = 10_000
SALARY_MAX = 500_000

MULTISELECT_SPECS = {
    "LanguageHaveWorkedWith": (("LanguageHaveWorkedWith",), "lang", 25),
    "DatabaseHaveWorkedWith": (("DatabaseHaveWorkedWith",), "db", 15),
    "PlatformHaveWorkedWith": (("PlatformHaveWorkedWith",), "plat", 12),
    "ToolsTechHaveWorkedWith": (
        ("ToolsTechHaveWorkedWith", "DevEnvsHaveWorkedWith"),
        "tool",
        15,
    ),
}

TIER1_COUNTRIES = {
    "United States of America",
    "United Kingdom of Great Britain and Northern Ireland",
    "United Kingdom",
    "Germany",
    "Canada",
    "Australia",
    "Switzerland",
    "Netherlands",
}

DATA_ROLE_PATTERNS = (
    "data scientist",
    "data analyst",
    "data or business analyst",
    "data engineer",
    "business intelligence",
    "machine learning",
    "ml engineer",
)

EXPERIENCE_COLUMN_CANDIDATES = ("YearsCodePro", "WorkExp", "YearsCode")
FULL_TIME_EMPLOYMENT_VALUES = ("Employed, full-time", "Employed")


def safe_token(value: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return token or "unknown"


def parse_years_code_pro(value: object) -> float | np.nan:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    if text == "less than 1 year":
        return 0.5
    if text == "more than 50 years":
        return 51.0

    try:
        return float(text)
    except ValueError:
        return np.nan


def encode_education(value: object) -> int:
    if pd.isna(value):
        return 3

    text = str(value).strip().lower()
    if "primary" in text:
        return 1
    if "secondary school" in text:
        return 2
    if "some college" in text or "without earning a degree" in text:
        return 3
    if "associate degree" in text:
        return 4
    if "bachelor" in text:
        return 5
    if "master" in text:
        return 6
    if "professional degree" in text:
        return 7
    if "doctoral degree" in text or text.startswith("doctorate"):
        return 8
    return 3


def one_hot_multiselect(
    df: pd.DataFrame, col: str, prefix: str, top_n: int
) -> tuple[pd.DataFrame, list[str]]:
    if col not in df.columns:
        return df, []

    exploded = df[col].dropna().str.split(";").explode().str.strip()
    top_values = exploded.value_counts().head(top_n).index.tolist()

    for value in top_values:
        safe_name = safe_token(value)
        pattern = rf"(?:^|;\s*){re.escape(value)}(?:\s*;|$)"
        df[f"{prefix}_{safe_name}"] = (
            df[col].fillna("").str.contains(pattern, regex=True).astype(int)
        )

    return df, top_values


def first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def is_full_time_employment(value: object) -> bool:
    if pd.isna(value):
        return False

    text = str(value).strip()
    if text in FULL_TIME_EMPLOYMENT_VALUES:
        return True

    lowered = text.lower()
    return "employed, full-time" in lowered


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Stack Overflow survey data.")
    parser.add_argument("--input", type=Path, default=RAW_CSV, help="Path to raw CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_PARQUET,
        help="Path to cleaned parquet output",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    summary: dict[str, object] = {"raw_rows": int(len(df))}

    df = df[df["Employment"].apply(is_full_time_employment)]
    summary["full_time_rows"] = int(len(df))

    df[SALARY_COLUMN] = pd.to_numeric(df[SALARY_COLUMN], errors="coerce")
    invalid_salary_rows = int(df[SALARY_COLUMN].isna().sum())
    valid_salary_mask = df[SALARY_COLUMN].between(SALARY_MIN, SALARY_MAX)
    summary["removed_missing_or_invalid_salary_rows"] = invalid_salary_rows
    summary["removed_salary_outlier_rows"] = int((~valid_salary_mask & df[SALARY_COLUMN].notna()).sum())
    df = df[valid_salary_mask].copy()
    summary["salary_filtered_rows"] = int(len(df))

    selected_values: dict[str, list[str]] = {}
    selected_source_columns: dict[str, str | None] = {}
    for logical_name, (candidates, prefix, top_n) in MULTISELECT_SPECS.items():
        source_column = first_existing_column(df, candidates)
        selected_source_columns[logical_name] = source_column
        if source_column is None:
            selected_values[logical_name] = []
            continue
        df, values = one_hot_multiselect(df, source_column, prefix, top_n)
        selected_values[logical_name] = values

    experience_column = first_existing_column(df, EXPERIENCE_COLUMN_CANDIDATES)
    if experience_column is None:
        raise KeyError(
            "Could not find any supported experience column. "
            f"Tried: {', '.join(EXPERIENCE_COLUMN_CANDIDATES)}"
        )

    df["years_exp"] = df[experience_column].apply(parse_years_code_pro)
    df["exp_bucket"] = pd.cut(
        df["years_exp"],
        bins=[-0.1, 2, 5, 10, 20, np.inf],
        labels=["0-2yr", "3-5yr", "6-10yr", "11-20yr", "20yr+"],
    )
    df["country_group"] = df["Country"].apply(
        lambda country: "tier1"
        if country in TIER1_COUNTRIES
        else ("india" if country == "India" else "other")
    )
    df["is_data_role"] = df["DevType"].fillna("").str.lower().apply(
        lambda text: int(any(pattern in text for pattern in DATA_ROLE_PATTERNS))
    )
    df["edu_num"] = df["EdLevel"].apply(encode_education)
    df["log_salary"] = np.log(df[SALARY_COLUMN])

    lang_cols = [column for column in df.columns if column.startswith("lang_")]
    breadth_cols = [
        column
        for column in df.columns
        if column.startswith(("lang_", "db_", "plat_", "tool_"))
    ]
    df["n_languages"] = df[lang_cols].sum(axis=1) if lang_cols else 0
    df["skill_breadth"] = df[breadth_cols].sum(axis=1) if breadth_cols else 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ETL_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(args.output, index=False)

    metadata_payload = {
        "multiselect_top_values": selected_values,
        "multiselect_source_columns": selected_source_columns,
        "experience_source_column": experience_column,
        "generated_columns": [column for column in df.columns if column.startswith(("lang_", "db_", "plat_", "tool_"))],
    }
    MULTISELECT_METADATA.write_text(json.dumps(metadata_payload, indent=2))

    summary.update(
        {
            "final_rows": int(len(df)),
            "final_columns": int(df.shape[1]),
            "experience_source_column": experience_column,
            "tool_source_column": selected_source_columns.get("ToolsTechHaveWorkedWith"),
            "output_path": str(args.output),
        }
    )
    ETL_SUMMARY.write_text(json.dumps(summary, indent=2))

    print(
        f"Saved {len(df):,} rows and {df.shape[1]:,} columns to {args.output}"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
