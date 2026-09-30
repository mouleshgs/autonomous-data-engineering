from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from langchain_core.output_parsers import JsonOutputParser
UPLOAD_DIR = Path(__file__).resolve().parents[2] / "data" / "uploads"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def quality(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"overall": 0, "completeness": 0, "consistency": 0, "validity": 0, "uniqueness": 0}
    total = max(df.size, 1)
    completeness = round((1 - df.isna().sum().sum() / total) * 100, 1)
    duplicates = df.duplicated().sum()
    uniqueness = round((1 - duplicates / max(len(df), 1)) * 100, 1)
    string_cols = df.select_dtypes(include="object")
    consistency = 100.0
    if not string_cols.empty:
        padded = string_cols.map(lambda value: isinstance(value, str) and value != value.strip())
        consistency -= min(25, padded.sum().sum() / max(string_cols.size, 1) * 100)
    validity = 100.0
    for column in df.select_dtypes(include="number"):
        validity -= min(20, (df[column] < 0).sum() / max(len(df), 1) * 100)
    overall = round(.30 * completeness + .25 * consistency + .25 * validity + .20 * uniqueness, 1)
    return {
        "overall": float(overall),
        "completeness": float(round(completeness, 1)),
        "consistency": float(round(consistency, 1)),
        "validity": float(round(validity, 1)),
        "uniqueness": float(round(uniqueness, 1)),
    }


def load_frame(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if suffix == ".json":
        return pd.read_json(file_path)
    raise ValueError(f"Unsupported structured format: {suffix}")


def profile_frame(df: pd.DataFrame) -> dict[str, Any]:
    columns = []
    for name in df.columns:
        series = df[name]
        non_null = series.dropna()
        item: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "nulls": int(series.isna().sum()),
            "null_pct": round(float(series.isna().mean() * 100), 1) if len(series) else 0.0,
            "unique": int(series.nunique(dropna=True)),
            "empty": bool(series.isna().all()),
        }
        if pd.api.types.is_numeric_dtype(series):
            item.update(
                {
                    "min": float(series.min()) if not series.dropna().empty else None,
                    "max": float(series.max()) if not series.dropna().empty else None,
                    "mean": round(float(series.mean()), 2) if not series.dropna().empty else None,
                    "numeric_like": False,
                    "boolean_like": False,
                }
            )
        else:
            normalized = non_null.astype("string").str.strip().str.casefold()
            numeric_values = pd.to_numeric(
                normalized.str.replace("$", "", regex=False).str.replace(",", "", regex=False),
                errors="coerce",
            )
            boolean_values = {"true", "false", "yes", "no", "y", "n", "1", "0"}
            item["boolean_like"] = bool(len(non_null) and normalized.isin(boolean_values).all())
            item["numeric_like"] = bool(
                len(non_null) and numeric_values.notna().all() and not item["boolean_like"]
            )
            if item["numeric_like"]:
                item.update(
                    {
                        "min": float(numeric_values.min()),
                        "max": float(numeric_values.max()),
                        "mean": round(float(numeric_values.mean()), 2),
                    }
                )
            item["top_values"] = non_null.astype(str).value_counts().head(3).to_dict()
        columns.append(item)
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "empty_rows": int(df.isna().all(axis=1).sum()) if len(df.columns) else len(df),
        "empty_columns": [str(name) for name in df.columns if df[name].isna().all()],
        "column_stats": columns,
    }


def remove_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="all").copy()


def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    empty_columns = df.isna().all()
    if empty_columns.all():
        return df.copy()
    return df.loc[:, ~empty_columns].copy()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.select_dtypes(include="object"):
        values = cleaned[column].astype("string").str.strip()
        numeric = pd.to_numeric(
            values.str.replace("$", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce",
        )
        non_null = cleaned[column].notna()
        if non_null.any() and numeric[non_null].notna().all():
            cleaned[column] = numeric
    return cleaned


def normalize_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    true_values = {"true", "yes", "y", "1"}
    false_values = {"false", "no", "n", "0"}
    for column in cleaned.select_dtypes(include="object"):
        values = cleaned[column].dropna().astype(str).str.strip().str.casefold()
        if values.empty or not values.isin(true_values | false_values).all():
            continue
        cleaned[column] = cleaned[column].map(
            lambda value: "Yes" if isinstance(value, str) and value.strip().casefold() in true_values
            else "No" if isinstance(value, str) and value.strip().casefold() in false_values
            else value
        )
    return cleaned


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.columns:
        if cleaned[column].isna().any():
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())
            else:
                mode = cleaned[column].mode()
                cleaned[column] = cleaned[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return cleaned


def correct_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.select_dtypes(include="number"):
        values = cleaned[column]
        negative = values < 0
        if not negative.any():
            continue
        cleaned.loc[negative, column] = 0
    return cleaned


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.select_dtypes(include="object"):
        column_name = str(column).lower()
        is_region_column = column_name in {"region", "region_code", "country", "country_code"}

        def normalize(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            value = value.strip()
            if is_region_column and value.isalpha() and len(value) <= 3:
                return value.upper()
            return value.title()

        cleaned[column] = cleaned[column].map(normalize)
    return cleaned


def standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.columns:
        if "date" not in str(column).lower():
            continue
        values = cleaned[column]
        non_null = values.notna()
        if not non_null.any():
            continue
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        if parsed[non_null].notna().all():
            cleaned[column] = parsed.dt.strftime("%Y-%m-%d")
    return cleaned


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    names: list[str] = []
    used: set[str] = set()
    for original in cleaned.columns:
        base = re.sub(r"[^a-z0-9]+", "_", str(original).strip().lower()).strip("_") or "column"
        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        names.append(name)
    cleaned.columns = names
    return cleaned


def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    date_columns = [column for column in df.columns if "date" in str(column).lower()]
    date_values_valid = all(
        pd.to_datetime(df[column].dropna(), errors="coerce", format="mixed").notna().all()
        for column in date_columns
    )
    checks = {
        "no_missing_values": not df.isna().any().any(),
        "no_duplicate_rows": not df.duplicated().any(),
        "no_negative_numeric_values": not df.select_dtypes(include="number").lt(0).any().any(),
        "no_empty_rows": not df.isna().all(axis=1).any() if len(df.columns) else not len(df),
        "no_empty_columns": not df.isna().all().any(),
        "date_values_parseable": date_values_valid,
    }
    issues = [name for name, passed in checks.items() if not passed]
    return {
        "validated": not issues,
        "quality": quality(df),
        "rows": len(df),
        "columns": len(df.columns),
        "checks": checks,
        "issues": issues,
    }


def persist_dataframe(df: pd.DataFrame, output_path: str | Path) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return str(target)


def build_plan(profile: dict[str, Any]) -> list[dict[str, str]]:
    steps = []
    if profile.get("empty_rows"):
        steps.append({"operation": "remove_empty_rows", "reason": f"{profile['empty_rows']} fully empty rows detected"})
    empty_columns = profile.get("empty_columns", [])
    if empty_columns and len(empty_columns) < profile.get("columns", 0):
        steps.append({"operation": "remove_empty_columns", "reason": f"Remove fully empty columns: {', '.join(empty_columns)}"})
    if profile.get("duplicates"):
        steps.append({"operation": "remove_duplicates", "reason": f"{profile['duplicates']} exact duplicate rows detected"})
    numeric_columns = [column["name"] for column in profile.get("column_stats", []) if column.get("numeric_like")]
    if numeric_columns:
        steps.append({"operation": "coerce_numeric_columns", "reason": f"Convert numeric text to numbers in: {', '.join(numeric_columns)}"})
    boolean_columns = [column["name"] for column in profile.get("column_stats", []) if column.get("boolean_like")]
    if boolean_columns:
        steps.append({"operation": "normalize_boolean_columns", "reason": f"Normalize boolean-like values in: {', '.join(boolean_columns)}"})
    if profile.get("missing_values"):
        steps.append({"operation": "fill_missing", "reason": f"{profile['missing_values']} missing values detected", "method": "median/mode"})
    negative_columns = [
        column["name"]
        for column in profile.get("column_stats", [])
        if column.get("min") is not None and column["min"] < 0
    ]
    if negative_columns:
        steps.append({"operation": "correct_negative_values", "reason": f"Set invalid negative values in {', '.join(negative_columns)} to zero"})
    if any(column.get("dtype") == "object" for column in profile.get("column_stats", [])):
        steps.append({"operation": "normalize_strings", "reason": "Text columns may contain whitespace or casing drift"})
    date_columns = [
        column["name"]
        for column in profile.get("column_stats", [])
        if "date" in column.get("name", "").lower()
    ]
    if date_columns:
        steps.append({"operation": "standardize_dates", "reason": f"Normalize date formats in {', '.join(date_columns)} to ISO 8601"})
    column_names = [column["name"] for column in profile.get("column_stats", [])]
    normalized_names = [re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "column" for name in column_names]
    if column_names != normalized_names or len(set(normalized_names)) != len(normalized_names):
        steps.append({"operation": "standardize_column_names", "reason": "Normalize column names to unique snake_case identifiers"})
    steps.append({"operation": "validate", "reason": "Verify quality and schema after deterministic transformations"})
    steps.append({"operation": "store", "reason": "Persist the processed dataset and run metadata"})
    return steps


def plan_with_llm(profile: dict[str, Any], model: str = "llama3.2") -> list[dict[str, str]]:
    provider = os.getenv("LLM_PROVIDER", "demo").lower()
    if provider in {"demo", "none", "disabled"}:
        return build_plan(profile)
    if provider != "ollama":
        return build_plan(profile)

    llm_model = os.getenv("LLM_MODEL", model)
    parser = JsonOutputParser()
    prompt = (
        "You are a data engineering planner. Return a JSON array of objects with operation and reason fields. "
        "Choose only from the available operations below, and include every operation because each is required by observed profile evidence. "
        "Do not invent columns, values, or operations. Use validation to confirm missing values, duplicates, empty rows/columns, and negative numbers are handled.\n"
        "Available operations: " + json.dumps(build_plan(profile), default=str) + "\n"
        "Detailed dataset profile: " + json.dumps(profile, default=str) + "\n"
        + parser.get_format_instructions()
    )
    try:
        llm = ChatOllama(model=llm_model, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        response = llm.invoke([HumanMessage(content=prompt)])
        text = getattr(response, "content", str(response)).strip()
        parsed = parser.parse(text)
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            proposed = {
                item.get("operation"): item
                for item in parsed
                if isinstance(item.get("operation"), str)
            }
            required_plan = build_plan(profile)
            allowed_operations = {step["operation"] for step in required_plan}
            if set(proposed).issubset(allowed_operations):
                return [
                    {
                        **step,
                        "reason": proposed.get(step["operation"], {}).get("reason", step["reason"]),
                    }
                    for step in required_plan
                ]
    except Exception:
        pass
    return build_plan(profile)


def execute_tool(name: str, *args: Any, **kwargs: Any) -> Any:
    registry = {
        "load_frame": load_frame,
        "profile_frame": profile_frame,
        "remove_empty_rows": remove_empty_rows,
        "remove_empty_columns": remove_empty_columns,
        "remove_duplicates": remove_duplicates,
        "coerce_numeric_columns": coerce_numeric_columns,
        "normalize_boolean_columns": normalize_boolean_columns,
        "fill_missing": fill_missing_values,
        "correct_negative_values": correct_negative_values,
        "normalize_strings": normalize_strings,
        "standardize_dates": standardize_dates,
        "standardize_column_names": standardize_column_names,
        "validate": validate_dataframe,
        "store": lambda frame, file_path: persist_dataframe(frame, file_path),
        "quality": quality,
    }
    if name not in registry:
        raise KeyError(f"Unsupported tool: {name}")
    return registry[name](*args, **kwargs)


SAFE_TOOL_REGISTRY = {
    "load_frame": load_frame,
    "profile_frame": profile_frame,
    "remove_empty_rows": remove_empty_rows,
    "remove_empty_columns": remove_empty_columns,
    "remove_duplicates": remove_duplicates,
    "coerce_numeric_columns": coerce_numeric_columns,
    "normalize_boolean_columns": normalize_boolean_columns,
    "fill_missing": fill_missing_values,
    "correct_negative_values": correct_negative_values,
    "normalize_strings": normalize_strings,
    "standardize_dates": standardize_dates,
    "standardize_column_names": standardize_column_names,
    "validate": validate_dataframe,
    "store": persist_dataframe,
    "quality": quality,
}
