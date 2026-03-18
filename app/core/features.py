"""Definições de features e utilitários de validação."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

BASE_FEATURES: List[str] = [
    "População",
    "Área_km",
    "Receita.Anual",
    "Orçamento.anual",
    "Danos.e.Prejuízos",
    "Afetados",
]
DIRECT_INPUT_FEATURES: List[str] = BASE_FEATURES + ["Precipitação.pluviométrica"]
DERIVED_FEATURES: List[str] = [
    "Densidade.populacional",
    "Capacidade.de.investimento.na.resposta.ao.desastre",
    "Capacidade.de.projeção.financeira",
    "Densidade.populacional.de.afetados",
]
FEATURES: List[str] = DIRECT_INPUT_FEATURES + DERIVED_FEATURES

DERIVED_FEATURE_SPECS: Dict[str, Dict[str, str]] = {
    "Densidade.populacional": {
        "numerator": "População",
        "denominator": "Área_km",
    },
    "Capacidade.de.investimento.na.resposta.ao.desastre": {
        "numerator": "Receita.Anual",
        "denominator": "Danos.e.Prejuízos",
    },
    "Capacidade.de.projeção.financeira": {
        "numerator": "Receita.Anual",
        "denominator": "Orçamento.anual",
    },
    "Densidade.populacional.de.afetados": {
        "numerator": "Afetados",
        "denominator": "População",
    },
}

LEVEL_MAP = {
    "1": "nivel_I",
    "2": "nivel_II",
    "3": "nivel_III",
}
RARE_CLASSES = {"4", "5", "6", "7", "8", "9", "10"}
CLASS_LABELS: List[str] = list(LEVEL_MAP.values()) + ["raro"]
DISPLAY_LABELS = {
    "nivel_I": "Nível I",
    "nivel_II": "Nível II",
    "nivel_III": "Nível III",
    "raro": "Raro",
}


class MissingColumnsError(ValueError):
    """Disparado quando o payload não contém todas as features obrigatórias."""

    def __init__(self, missing_columns: List[str]) -> None:
        self.missing_columns = missing_columns
        super().__init__(f"Colunas faltando: {missing_columns}")


class DerivedFeaturesError(ValueError):
    """Disparado quando não é possível calcular as variáveis derivadas."""


def _missing_row_indexes(series: pd.Series) -> List[int]:
    return series[series.isna()].index.tolist()


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula variáveis derivadas sem sobrescrever payloads antigos já completos."""
    enriched = df.copy()
    relevant_columns = {
        spec["numerator"] for spec in DERIVED_FEATURE_SPECS.values()
    } | {
        spec["denominator"] for spec in DERIVED_FEATURE_SPECS.values()
    } | set(DERIVED_FEATURES)

    for column in relevant_columns:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")

    errors: List[str] = []

    for derived_feature, spec in DERIVED_FEATURE_SPECS.items():
        numerator = spec["numerator"]
        denominator = spec["denominator"]

        if derived_feature not in enriched.columns:
            enriched[derived_feature] = pd.Series(pd.NA, index=enriched.index, dtype="Float64")

        missing_mask = enriched[derived_feature].isna()
        if not missing_mask.any():
            continue

        if numerator not in enriched.columns or denominator not in enriched.columns:
            errors.append(
                f"Não foi possível calcular '{derived_feature}' sem as colunas '{numerator}' e '{denominator}'."
            )
            continue

        missing_inputs = missing_mask & (enriched[numerator].isna() | enriched[denominator].isna())
        if missing_inputs.any():
            errors.append(
                f"Não foi possível calcular '{derived_feature}' por valores ausentes nas linhas "
                f"{missing_inputs[missing_inputs].index.tolist()}."
            )

        zero_division = missing_mask & ~missing_inputs & enriched[denominator].eq(0)
        if zero_division.any():
            errors.append(
                f"Não foi possível calcular '{derived_feature}' por divisão por zero em '{denominator}' nas linhas "
                f"{zero_division[zero_division].index.tolist()}."
            )

        calculable_mask = missing_mask & ~missing_inputs & ~zero_division

        enriched.loc[calculable_mask, derived_feature] = (
            enriched.loc[calculable_mask, numerator] / enriched.loc[calculable_mask, denominator]
        )

        unresolved_rows = _missing_row_indexes(enriched.loc[calculable_mask, derived_feature])
        if unresolved_rows:
            errors.append(
                f"Não foi possível calcular '{derived_feature}' nas linhas {unresolved_rows}."
            )

    if errors:
        raise DerivedFeaturesError(" ".join(errors))

    return enriched


def check_columns(df) -> List[str]:
    """Retorna as features obrigatórias ausentes."""
    return [column for column in FEATURES if column not in df.columns]


def map_cluster_label(value) -> str:
    """Converte o valor original de `cluster_H` para a taxonomia agregada."""
    if value is None or value != value:
        raise ValueError("Valor da coluna cluster_H inválido para mapeamento.")
    value_str = str(int(value))
    if value_str in LEVEL_MAP:
        return LEVEL_MAP[value_str]
    if value_str in RARE_CLASSES:
        return "raro"
    raise ValueError(f"Valor de cluster_H desconhecido: {value}")


def encode_target(series) -> List[str]:
    """Codifica a coluna alvo original nas classes agregadas."""
    return [map_cluster_label(v) for v in series]


def ensure_valid_features(df) -> None:
    """Valida a presença das features obrigatórias."""
    missing = check_columns(df)
    if missing:
        raise MissingColumnsError(missing)


def select_feature_frame(df):
    """Seleciona apenas as features usadas pelo pipeline."""
    ensure_valid_features(df)
    return df[FEATURES].copy()


def to_display_label(label: str) -> str:
    """Converte o rótulo interno para exibição amigável."""
    return DISPLAY_LABELS.get(label, label)
