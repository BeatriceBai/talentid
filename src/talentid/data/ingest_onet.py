"""Transform the O*NET 31.0 Excel snapshot into normalized Parquet tables."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ONET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "onet"
    / "31.0"
    / "files"
    / "db_31_0_excel"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "onet" / "31.0"


def read_source(filename: str) -> pd.DataFrame:
    """Read one required O*NET Excel file."""
    path = ONET_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing O*NET source file: {path}")

    return pd.read_excel(path)


def clean_text(series: pd.Series) -> pd.Series:
    """Strip whitespace while preserving missing values."""
    return series.astype("string").str.strip()


def parse_month(series: pd.Series) -> pd.Series:
    """Parse O*NET month/year values."""
    return pd.to_datetime(series, format="%m/%Y", errors="coerce")


def concat_frames(
    frames: list[pd.DataFrame],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Concatenate frames without pandas all-null-column ambiguity."""
    prepared = [frame.dropna(axis=1, how="all") for frame in frames]
    result = pd.concat(prepared, ignore_index=True)

    if columns is not None:
        result = result.reindex(columns=columns)

    return result


def stable_software_id(name: str) -> str:
    """Create a deterministic ID for software names without O*NET IDs."""
    normalized = " ".join(name.casefold().split())
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")[:40]
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:10]
    slug = slug or "software"
    return f"software:{slug}:{digest}"


def build_occupations() -> pd.DataFrame:
    """Create the occupation dimension."""
    occupations = read_source("Occupation Data.xlsx").rename(
        columns={
            "O*NET-SOC Code": "onet_soc_code",
            "Title": "title",
            "Description": "description",
        }
    )

    zones = read_source("Job Zones.xlsx").rename(
        columns={
            "O*NET-SOC Code": "onet_soc_code",
            "Job Zone": "job_zone",
            "Date": "job_zone_date",
            "Domain Source": "job_zone_source",
        }
    )

    zones["job_zone_date"] = parse_month(zones["job_zone_date"])
    zones["job_zone"] = pd.to_numeric(
        zones["job_zone"], errors="coerce"
    ).astype("Int64")

    result = occupations.merge(
        zones[
            [
                "onet_soc_code",
                "job_zone",
                "job_zone_date",
                "job_zone_source",
            ]
        ],
        on="onet_soc_code",
        how="left",
        validate="one_to_one",
    )

    for column in ("onet_soc_code", "title", "description"):
        result[column] = clean_text(result[column])

    result["has_job_zone_data"] = result["job_zone"].notna()
    return result.sort_values("onet_soc_code").reset_index(drop=True)


def build_job_titles() -> pd.DataFrame:
    """Combine alternate, short, and reported occupation titles."""
    source = read_source("Job Titles.xlsx")

    alternate = pd.DataFrame(
        {
            "onet_soc_code": source["O*NET-SOC Code"],
            "job_title": source["Job Title"],
            "title_type": "alternate",
            "source_detail": source["Source(s)"].astype("string"),
            "shown_in_my_next_move": pd.Series(
                pd.NA,
                index=source.index,
                dtype="boolean",
            ),
        }
    )

    short_source = source[source["Short Title"].notna()]
    short = pd.DataFrame(
        {
            "onet_soc_code": short_source["O*NET-SOC Code"],
            "job_title": short_source["Short Title"],
            "title_type": "short",
            "source_detail": short_source["Source(s)"].astype("string"),
            "shown_in_my_next_move": pd.Series(
                pd.NA,
                index=short_source.index,
                dtype="boolean",
            ),
        }
    )

    reported_source = read_source("Sample of Reported Titles.xlsx")
    reported = pd.DataFrame(
        {
            "onet_soc_code": reported_source["O*NET-SOC Code"],
            "job_title": reported_source["Reported Job Title"],
            "title_type": "reported",
            "source_detail": pd.Series(
                pd.NA,
                index=reported_source.index,
                dtype="string",
            ),
            "shown_in_my_next_move": (
                reported_source["Shown in My Next Move"]
                .astype("string")
                .str.upper()
                .eq("Y")
            ),
        }
    )

    result = concat_frames([alternate, short, reported])

    result["onet_soc_code"] = clean_text(result["onet_soc_code"])
    result["job_title"] = clean_text(result["job_title"])
    result["title_type"] = clean_text(result["title_type"])
    result["source_detail"] = clean_text(result["source_detail"])

    result = result.dropna(subset=["onet_soc_code", "job_title"])
    result["_title_key"] = result["job_title"].str.casefold()

    result = result.drop_duplicates(
        ["onet_soc_code", "_title_key", "title_type"]
    ).drop(columns="_title_key")

    return result.sort_values(
        ["onet_soc_code", "title_type", "job_title"]
    ).reset_index(drop=True)


def build_tasks() -> pd.DataFrame:
    """Create the occupation-task relationship table."""
    tasks = read_source("Task Statements.xlsx").rename(
        columns={
            "O*NET-SOC Code": "onet_soc_code",
            "Task ID": "task_id",
            "Task": "task",
            "Task Type": "task_type",
            "Incumbents Responding": "incumbents_responding",
            "Date": "source_date",
            "Domain Source": "domain_source",
        }
    )

    tasks["onet_soc_code"] = clean_text(tasks["onet_soc_code"])
    tasks["task"] = clean_text(tasks["task"])
    tasks["task_type"] = clean_text(tasks["task_type"]).fillna(
        "Unspecified"
    )
    tasks["domain_source"] = clean_text(tasks["domain_source"])
    tasks["task_id"] = pd.to_numeric(
        tasks["task_id"], errors="raise"
    ).astype("Int64")
    tasks["incumbents_responding"] = pd.to_numeric(
        tasks["incumbents_responding"], errors="coerce"
    ).astype("Int64")
    tasks["source_date"] = parse_month(tasks["source_date"])

    return tasks[
        [
            "task_id",
            "onet_soc_code",
            "task",
            "task_type",
            "incumbents_responding",
            "source_date",
            "domain_source",
        ]
    ].sort_values("task_id").reset_index(drop=True)


def build_rated_skills(
    filename: str,
    skill_type: str,
) -> pd.DataFrame:
    """Transform an O*NET IM/LV skill table into relationship rows."""
    data = read_source(filename).rename(
        columns={
            "O*NET-SOC Code": "onet_soc_code",
            "Element ID": "element_id",
            "Element Name": "skill_name",
            "Scale ID": "scale_id",
            "Data Value": "data_value",
            "Recommend Suppress": "recommend_suppress",
            "Not Relevant": "not_relevant",
            "Date": "source_date",
            "Domain Source": "domain_source",
        }
    )

    suppressed = (
        data["recommend_suppress"]
        .fillna("N")
        .astype("string")
        .str.upper()
        .eq("Y")
    )

    data = data[~suppressed].copy()

    for column in (
        "onet_soc_code",
        "element_id",
        "skill_name",
        "scale_id",
        "domain_source",
    ):
        data[column] = clean_text(data[column])

    data["data_value"] = pd.to_numeric(
        data["data_value"], errors="coerce"
    )
    data["source_date"] = parse_month(data["source_date"])
    data["not_relevant"] = (
        data["not_relevant"]
        .astype("string")
        .str.upper()
        .eq("Y")
    )

    keys = ["onet_soc_code", "element_id", "skill_name"]

    ratings = (
        data.pivot_table(
            index=keys,
            columns="scale_id",
            values="data_value",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"IM": "importance", "LV": "level"})
    )
    ratings.columns.name = None

    if "importance" not in ratings:
        ratings["importance"] = pd.NA
    if "level" not in ratings:
        ratings["level"] = pd.NA

    metadata = (
        data.groupby(keys, as_index=False)
        .agg(
            source_date=("source_date", "max"),
            domain_source=("domain_source", "first"),
            not_relevant=("not_relevant", "max"),
        )
    )

    result = ratings.merge(
        metadata,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    result["skill_id"] = "onet:" + result["element_id"]
    result["skill_type"] = skill_type
    result["hot_technology"] = False
    result["in_demand"] = False
    result["category_element_id"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )
    result["category_name"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    return result


def build_software_skills() -> pd.DataFrame:
    """Create occupation-software relationships."""
    data = read_source("Software Skills.xlsx").rename(
        columns={
            "O*NET-SOC Code": "onet_soc_code",
            "Workplace Example": "skill_name",
            "Element ID": "category_element_id",
            "Element Name": "category_name",
            "Hot Technology": "hot_technology",
            "In Demand": "in_demand",
        }
    )

    for column in (
        "onet_soc_code",
        "skill_name",
        "category_element_id",
        "category_name",
    ):
        data[column] = clean_text(data[column])

    data["skill_id"] = data["skill_name"].map(stable_software_id)
    data["skill_type"] = "software"
    data["importance"] = float("nan")
    data["level"] = float("nan")
    data["hot_technology"] = (
        data["hot_technology"].astype("string").str.upper().eq("Y")
    )
    data["in_demand"] = (
        data["in_demand"].astype("string").str.upper().eq("Y")
    )
    data["not_relevant"] = False
    data["source_date"] = pd.NaT
    data["domain_source"] = "O*NET Software Skills"

    return data


def join_unique(values: pd.Series) -> str:
    """Join unique nonmissing strings."""
    unique = sorted(set(values.dropna().astype(str)))
    return " | ".join(unique)


def build_skills(
    rated: pd.DataFrame,
    software: pd.DataFrame,
) -> pd.DataFrame:
    """Create a canonical skill dimension."""
    reference = read_source("Content Model Reference.xlsx").rename(
        columns={
            "Element ID": "element_id",
            "Description": "description",
        }
    )
    reference["element_id"] = clean_text(reference["element_id"])

    rated_skills = rated[
        ["skill_id", "element_id", "skill_name", "skill_type"]
    ].drop_duplicates()

    rated_skills = rated_skills.merge(
        reference[["element_id", "description"]],
        on="element_id",
        how="left",
        validate="many_to_one",
    )
    rated_skills["category_element_id"] = pd.Series(
        pd.NA,
        index=rated_skills.index,
        dtype="string",
    )
    rated_skills["category_name"] = pd.Series(
        pd.NA,
        index=rated_skills.index,
        dtype="string",
    )

    software_skills = (
        software.groupby(
            ["skill_id", "skill_name", "skill_type"],
            as_index=False,
        )
        .agg(
            category_element_id=(
                "category_element_id",
                join_unique,
            ),
            category_name=("category_name", join_unique),
        )
    )
    software_skills["element_id"] = pd.Series(
        pd.NA,
        index=software_skills.index,
        dtype="string",
    )
    software_skills["description"] = (
        "Software or technology skill. O*NET categories: "
        + software_skills["category_name"]
    )

    columns = [
        "skill_id",
        "element_id",
        "skill_name",
        "skill_type",
        "description",
        "category_element_id",
        "category_name",
    ]

    result = concat_frames(
        [rated_skills[columns], software_skills[columns]],
        columns=columns,
    )

    return result.sort_values("skill_id").reset_index(drop=True)


def validate_tables(
    occupations: pd.DataFrame,
    job_titles: pd.DataFrame,
    tasks: pd.DataFrame,
    skills: pd.DataFrame,
    relationships: pd.DataFrame,
) -> None:
    """Validate primary keys and foreign keys."""
    if occupations["onet_soc_code"].duplicated().any():
        raise ValueError("Duplicate occupation IDs found")

    if tasks["task_id"].duplicated().any():
        raise ValueError("Duplicate task IDs found")

    if skills["skill_id"].duplicated().any():
        raise ValueError("Duplicate skill IDs found")

    if relationships.duplicated(["onet_soc_code", "skill_id"]).any():
        raise ValueError("Duplicate occupation-skill relationships found")

    occupation_ids = set(occupations["onet_soc_code"])
    skill_ids = set(skills["skill_id"])

    occupation_references = {
        "job_titles": set(job_titles["onet_soc_code"]),
        "tasks": set(tasks["onet_soc_code"]),
        "occupation_skills": set(relationships["onet_soc_code"]),
    }

    for table_name, referenced_ids in occupation_references.items():
        unknown = referenced_ids - occupation_ids
        if unknown:
            sample = sorted(unknown)[:5]
            raise ValueError(
                f"Unknown occupation IDs in {table_name}: {sample}"
            )

    unknown_skills = set(relationships["skill_id"]) - skill_ids
    if unknown_skills:
        raise ValueError(
            f"Unknown skill IDs: {sorted(unknown_skills)[:5]}"
        )


def main() -> None:
    """Run the complete O*NET ingestion pipeline."""
    occupations = build_occupations()
    job_titles = build_job_titles()
    tasks = build_tasks()

    rated_frames = [
        build_rated_skills(
            "Essential Skills.xlsx",
            "essential",
        ),
        build_rated_skills(
            "Transferable Skills.xlsx",
            "transferable",
        ),
        build_rated_skills(
            "Knowledge.xlsx",
            "knowledge",
        ),
    ]
    rated = concat_frames(
        rated_frames,
        columns=list(rated_frames[0].columns),
    )

    software = build_software_skills()

    relationship_columns = [
        "onet_soc_code",
        "skill_id",
        "skill_name",
        "skill_type",
        "importance",
        "level",
        "hot_technology",
        "in_demand",
        "not_relevant",
        "source_date",
        "domain_source",
        "category_element_id",
        "category_name",
    ]

    relationships = concat_frames(
        [
            rated[relationship_columns],
            software[relationship_columns],
        ],
        columns=relationship_columns,
    )

    skills = build_skills(rated, software)

    occupations["has_task_data"] = occupations[
        "onet_soc_code"
    ].isin(tasks["onet_soc_code"])
    occupations["has_skill_data"] = occupations[
        "onet_soc_code"
    ].isin(relationships["onet_soc_code"])

    validate_tables(
        occupations,
        job_titles,
        tasks,
        skills,
        relationships,
    )

    tables = {
        "occupations": occupations,
        "job_titles": job_titles,
        "tasks": tasks,
        "skills": skills,
        "occupation_skills": relationships,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        output_path = OUTPUT_DIR / f"{name}.parquet"
        table.to_parquet(output_path, index=False)
        print(f"{name}: {len(table):,} rows -> {output_path}")


if __name__ == "__main__":
    main()