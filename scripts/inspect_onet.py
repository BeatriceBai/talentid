"""Inspect schemas and sample records from selected O*NET source tables."""

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

SOURCE_FILES = (
    "Occupation Data.xlsx",
    "Job Titles.xlsx",
    "Sample of Reported Titles.xlsx",
    "Task Statements.xlsx",
    "Software Skills.xlsx",
    "Essential Skills.xlsx",
    "Transferable Skills.xlsx",
    "Knowledge.xlsx",
    "Job Zones.xlsx",
    "Content Model Reference.xlsx",
)


def shorten(value: object, limit: int = 100) -> str:
    """Convert a value to compact, single-line text."""
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else f"{text[:limit]}..."


def inspect_file(filename: str) -> None:
    """Print sheet names, columns, and one sample record."""
    path = ONET_DIR / filename

    if not path.exists():
        print(f"\nMISSING: {filename}")
        return

    with pd.ExcelFile(path) as workbook:
        print(f"\n{'=' * 80}")
        print(f"FILE: {filename}")
        print(f"SHEETS: {workbook.sheet_names}")

        for sheet_name in workbook.sheet_names:
            sample = pd.read_excel(
                workbook,
                sheet_name=sheet_name,
                nrows=1,
            )

            print(f"\nSHEET: {sheet_name}")
            print(f"COLUMNS ({len(sample.columns)}):")

            for column in sample.columns:
                if sample.empty:
                    example = "<no sample row>"
                else:
                    example = shorten(sample.iloc[0][column])

                print(f"  - {column}: {example}")


def main() -> None:
    """Inspect all selected O*NET files."""
    print(f"O*NET directory: {ONET_DIR}")

    for filename in SOURCE_FILES:
        inspect_file(filename)


if __name__ == "__main__":
    main()