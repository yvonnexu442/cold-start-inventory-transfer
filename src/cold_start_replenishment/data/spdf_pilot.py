"""Leakage-conscious SPDF MAN/BRAF parsing and metadata-only analog construction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

from cold_start_replenishment.paths import resolve_repo_path

SPDF_DATA_ROOT = Path(
    "data/interim/dataset_audit/spdf/Spare-Part-Demand-Forecasting-main/All Data sets"
)


@dataclass(frozen=True)
class PilotDataset:
    name: str
    metadata: pd.DataFrame
    demand: pd.DataFrame
    frequency: str


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_text(value: object) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    text = re.sub(r"[^A-Z0-9]+", " ", str(value).upper())
    return re.sub(r"\s+", " ", text).strip()


def redacted_id(dataset: str, item_id: object) -> str:
    digest = hashlib.sha256(f"{dataset}:{item_id}".encode()).hexdigest()[:10]
    return f"{dataset.lower()}-{digest}"


def parse_man(path: Path | None = None) -> PilotDataset:
    source = path or resolve_repo_path(SPDF_DATA_ROOT / "MAN.xlsx")
    raw = pd.read_excel(source, sheet_name="Data", header=None)
    demand_columns = list(range(9, 159))
    data = raw.iloc[6:].reset_index(drop=True)
    metadata = pd.DataFrame(
        {
            "item_id": _numeric(data[0]),
            "product_group": data[1].map(normalize_text),
            "cost_price": _numeric(data[2]),
            "fixed_order_cost": _numeric(data[3]),
            "inventory_cost": _numeric(data[4]),
            "lead_time": _numeric(data[5]),
            "moq": _numeric(data[7]),
        }
    )
    demand = data[demand_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    demand.columns = [f"week_{index + 1}" for index in range(len(demand.columns))]
    valid = metadata["item_id"].notna()
    metadata = metadata.loc[valid].reset_index(drop=True)
    metadata["item_id"] = metadata["item_id"].astype(int).astype(str)
    return PilotDataset("MAN", metadata, demand.loc[valid].reset_index(drop=True), "weekly")


def parse_braf(path: Path | None = None) -> PilotDataset:
    source = path or resolve_repo_path(SPDF_DATA_ROOT / "BRAF.xls")
    raw = pd.read_excel(source, sheet_name="Results___7_years_demand_patter")
    demand_columns = list(raw.columns[4:88])
    metadata = pd.DataFrame(
        {
            "item_id": _numeric(raw.iloc[:, 0]),
            "description": raw.iloc[:, 1].map(normalize_text),
            "lead_time": _numeric(raw.iloc[:, 2]),
            "price": _numeric(raw.iloc[:, 3]),
        }
    )
    demand = raw[demand_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    demand.columns = [f"month_{index + 1}" for index in range(len(demand.columns))]
    valid = metadata["item_id"].notna()
    metadata = metadata.loc[valid].reset_index(drop=True)
    metadata["item_id"] = metadata["item_id"].astype(int).astype(str)
    return PilotDataset("BRAF", metadata, demand.loc[valid].reset_index(drop=True), "monthly")


def field_semantics(dataset: str) -> list[dict[str, Any]]:
    common = {
        "static_or_time_varying": "static in workbook",
        "transformation": "numeric coercion; invalid values become missing",
        "leakage_risk": "low if used without target demand or benchmark outputs",
    }
    if dataset == "MAN":
        specifications = [
            (
                "MAN.xlsx / Data",
                "A / row 1 'SKU nr'",
                "item_id",
                "identifier",
                "none",
                True,
                False,
                "Row identifier.",
            ),
            (
                "MAN.xlsx / Data",
                "B / rows 1-3 'products (Sub packaging)'",
                "product_group",
                "categorical text",
                "sub-packaging code",
                True,
                False,
                "Numeric-looking group semantics require source confirmation.",
            ),
            (
                "MAN.xlsx / Data",
                "C / 'cost price'",
                "cost_price",
                "float",
                "Euro per unit (header)",
                True,
                True,
                "Used only after median normalization.",
            ),
            (
                "MAN.xlsx / Data",
                "D / 'fixed cost'",
                "fixed_order_cost",
                "float",
                "Euro per order (header)",
                True,
                True,
                "Normalized proxy; enterprise semantics not independently verified.",
            ),
            (
                "MAN.xlsx / Data",
                "E / 'inventory cost'",
                "inventory_cost",
                "float",
                "Euro per unit per year (header)",
                True,
                True,
                "Appears related to cost price; used as holding exposure.",
            ),
            (
                "MAN.xlsx / Data",
                "F / 'lead time'",
                "lead_time",
                "float",
                "weeks (header)",
                True,
                True,
                "Static snapshot; capped in sensitivity analysis.",
            ),
            (
                "MAN.xlsx / Data",
                "H / 'min. order quantity'",
                "moq",
                "float",
                "units (header)",
                True,
                True,
                "Treated as an order multiple/minimum in pilot.",
            ),
            (
                "MAN.xlsx / Data",
                "J:FC / week 1..150",
                "demand",
                "float array",
                "units per week",
                False,
                False,
                "Target demand is hidden; donor demand is cutoff-truncated.",
            ),
        ]
    else:
        specifications = [
            (
                "BRAF.xls / Results___7_years_demand_patter",
                "A / 'Item Ref no'",
                "item_id",
                "identifier",
                "none",
                True,
                False,
                "Row identifier.",
            ),
            (
                "BRAF.xls / Results___7_years_demand_patter",
                "B / 'DESCRIPTION'",
                "description",
                "text",
                "none",
                True,
                False,
                "May contain sensitive source text; committed outputs are redacted.",
            ),
            (
                "BRAF.xls / Results___7_years_demand_patter",
                "C / 'Lead Time (months)'",
                "lead_time",
                "float",
                "months",
                True,
                True,
                "Static snapshot; capped in sensitivity analysis.",
            ),
            (
                "BRAF.xls / Results___7_years_demand_patter",
                "D / 'PRICE (£)'",
                "price",
                "float",
                "GBP per unit (header)",
                True,
                True,
                "Holding-exposure proxy only; not shortage cost.",
            ),
            (
                "BRAF.xls / Results___7_years_demand_patter",
                "E:CJ / JAN96..DEC02",
                "demand",
                "float array",
                "units per month",
                False,
                False,
                "Target demand is hidden; donor demand is cutoff-truncated.",
            ),
        ]
    return [
        {
            "dataset": dataset,
            "raw_source": source,
            "raw_header_position": position,
            "parsed_field": field,
            "inferred_type": inferred_type,
            "unit": unit,
            **common,
            "safe_for_analog_construction": safe_analog,
            "safe_for_or_decision_layer": safe_decision,
            "possible_ambiguity": ambiguity,
            "semantic_class": "raw operational metadata"
            if field != "demand"
            else "raw demand history",
            "confidence": "medium" if "require" in ambiguity or "proxy" in ambiguity else "high",
        }
        for source, position, field, inferred_type, unit, safe_analog, safe_decision, ambiguity in specifications
    ]


def man_similarity(metadata: pd.DataFrame, target_indices: np.ndarray | None = None) -> np.ndarray:
    indices = np.arange(len(metadata)) if target_indices is None else target_indices
    product = metadata["product_group"].astype(str).to_numpy()
    exact = (product[indices, None] == product[None, :]).astype(float)
    numeric = metadata[["lead_time", "cost_price", "inventory_cost", "moq"]].copy()
    numeric = np.log1p(numeric.clip(lower=0))
    scale = numeric.std().replace(0, 1).fillna(1)
    values = ((numeric - numeric.median()) / scale).fillna(0).to_numpy()
    distance = np.sqrt(((values[indices, None, :] - values[None, :, :]) ** 2).mean(axis=2))
    return 0.45 * exact + 0.55 * np.exp(-distance)


def braf_similarity(metadata: pd.DataFrame, target_indices: np.ndarray | None = None) -> np.ndarray:
    indices = np.arange(len(metadata)) if target_indices is None else target_indices
    texts = metadata["description"].fillna("").astype(str)
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform(texts)
    cosine = cosine_similarity(tfidf[indices], tfidf)
    tokens = [set(text.split()) for text in texts]
    jaccard = np.zeros((len(indices), len(tokens)))
    for output_row, left in enumerate(indices):
        left_tokens = tokens[left]
        for right in range(len(tokens)):
            union = left_tokens | tokens[right]
            value = len(left_tokens & tokens[right]) / len(union) if union else 0.0
            jaccard[output_row, right] = value
    text_values = texts.to_numpy()
    exact = (text_values[indices, None] == text_values[None, :]).astype(float)
    numeric = metadata[["lead_time", "price"]].copy()
    numeric = np.log1p(numeric.clip(lower=0))
    scale = numeric.std().replace(0, 1).fillna(1)
    values = ((numeric - numeric.median()) / scale).fillna(0).to_numpy()
    distance = np.sqrt(((values[indices, None, :] - values[None, :, :]) ** 2).mean(axis=2))
    return 0.50 * cosine + 0.25 * jaccard + 0.15 * exact + 0.10 * np.exp(-distance)


def assign_regimes(
    top_scores: pd.Series, strong_threshold: float, medium_threshold: float, minimum_close: float
) -> pd.Series:
    conditions = [
        top_scores < minimum_close,
        top_scores >= strong_threshold,
        top_scores >= medium_threshold,
    ]
    return pd.Series(
        np.select(conditions, ["no-close-analog", "strong", "medium"], default="weak"),
        index=top_scores.index,
    )
