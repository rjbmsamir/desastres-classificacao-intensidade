"""Definições de features e utilitários de validação."""
from __future__ import annotations

from typing import List

FEATURES: List[str] = [
    "População",
    "Área_km",
    "Receita.Anual",
    "Orçamento.anual",
    "Danos.e.Prejuízos",
    "Afetados",
    "Precipitação.pluviométrica",
    "Densidade.populacional",
    "Capacidade.de.investimento.na.resposta.ao.desastre",
    "Capacidade.de.projeção.financeira",
    "Densidade.populacional.de.afetados",
]

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
        raise ValueError(f"Colunas faltando: {missing}")


def select_feature_frame(df):
    """Seleciona apenas as features usadas pelo pipeline."""
    ensure_valid_features(df)
    return df[FEATURES].copy()


def to_display_label(label: str) -> str:
    """Converte o rótulo interno para exibição amigável."""
    return DISPLAY_LABELS.get(label, label)
