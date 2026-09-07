"""Profile selected O*NET source tables before ingestion."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "onet"
    / "31.0"
    / "files"
    / "db_31_0_excel"
)

TABLES = {
    "Occupation Data.xlsx": ["O*NET-SOC Code"],
    "Job Titles.xlsx": ["O*NET-SOC Code", "Job Title"],
    "Sample of Reported Titles.xlsx": [
        "O*NET-SOC Code",
        "Reported Job Title",
    ],
    "Task Statements.xlsx": ["Task ID"],
    "Software Skills.xlsx": [
        "O*NET-SOC Code",
        "Workplace Example",
    ],
    "Essential Skills.xlsx": [
        "O*NET-SOC Code",
        "Element ID",
        "Scale ID",
    ],
    "Transferable Skills.xlsx": [
        "O*NET-SOC Code",
        "Element ID",
        "Scale ID",
    ],
    "Knowledge.xlsx": [
        "O*NET-SOC Code",
        "Element ID",
        "Scale ID",
    ],
    "Job Zones.xlsx": ["O*NET-SOC Code"],
    "Content Model Reference.xlsx": ["Element ID"],
}


def print_value_counts(df: pd.DataFrame, column: str) -> None:
    """Print compact value counts for a column."""
    if column not in df.columns:
        return

    values = df[column].astype("string").fillna("<missing>")
    counts = values.value_counts(dropna=False).sort_index()

    print(f"{column} values:")
    for value, count in counts.items():
        print(f"  {value}: {count}")


def profile_table(filename: str, keys: list[str]) -> None:
    """Print row counts, uniqueness, missingness, and rating metadata."""
    path = ONET_DIR / filename
    df = pd.read_excel(path)

    print(f"\n{'=' * 80}")
    print(f"FILE: {filename}")
    print(f"ROWS: {len(df):,}")
    print(f"COLUMNS: {len(df.columns)}")
    print(f"EXACT DUPLICATES: {df.duplicated().sum():,}")
    print(f"DUPLICATE KEYS {keys}: {df.duplicated(keys).sum():,}")

    missing_keys = df[keys].isna().sum()
    print("MISSING KEY VALUES:")
    for column, count in missing_keys.items():
        print(f"  {column}: {count:,}")

    if "O*NET-SOC Code" in df.columns:
        print(
            "UNIQUE OCCUPATIONS:",
            f"{df['O*NET-SOC Code'].nunique():,}",
        )

    if "Element ID" in df.columns:
        print("UNIQUE ELEMENTS:", f"{df['Element ID'].nunique():,}")

    if "Workplace Example" in df.columns:
        print(
            "UNIQUE SOFTWARE EXAMPLES:",
            f"{df['Workplace Example'].nunique():,}",
        )

    for column in (
        "Scale ID",
        "Recommend Suppress",
        "Not Relevant",
        "Task Type",
    ):
        print_value_counts(df, column)

    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], format="%m/%Y", errors="coerce")
        print(f"DATE RANGE: {dates.min()} to {dates.max()}")


def main() -> None:
    """Profile all selected O*NET tables."""
    for filename, keys in TABLES.items():
        profile_table(filename, keys)


if __name__ == "__main__":
    main()