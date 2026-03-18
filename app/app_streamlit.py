from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import DATA_PATH, SHEET_NAME
from app.db import fetch_recent_runs, get_conn, init_db
from app.features import CLASS_LABELS, to_display_label
from app.predict import ModelNotFoundError, predict_from_df

st.set_page_config(page_title="Classificação de Desastres", layout="wide")

st.title("Classificação de intensidade de desastres")
st.caption("Treine o modelo, registre execuções e gere predições Nível I/II/III ou Raro.")

init_db()


@st.cache_data(show_spinner=False)
def load_recent_runs(limit: int = 10) -> pd.DataFrame:
    rows = fetch_recent_runs(limit)
    if not rows:
        return pd.DataFrame()
    data = []
    for row in rows:
        metrics = json.loads(row["metrics_summary_json"] or "{}")
        data.append(
            {
                "id": row["id"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "algorithm": row["algorithm"],
                "n_samples": row["n_samples"],
                "n_features": row["n_features"],
                **metrics,
            }
        )
    return pd.DataFrame(data)


with st.expander("📚 Histórico de execuções recentes", expanded=True):
    recent_runs = load_recent_runs()
    if recent_runs.empty:
        st.info("Nenhuma execução registrada ainda.")
    else:
        st.dataframe(recent_runs, use_container_width=True)


st.header("Treinar / re-treinar o modelo")

train_col_left, train_col_right = st.columns([2, 1])

with train_col_left:
    default_path = Path(DATA_PATH)
    st.write(f"Base padrão: `{default_path}`")
    data_path_text = st.text_input("Caminho da base (CSV/XLSX)", value=str(default_path))
    sheet_name = st.text_input("Nome da aba (para Excel)", value=SHEET_NAME)

with train_col_right:
    train_clicked = st.button("Executar treinamento", use_container_width=True)

if train_clicked:
    cmd = [
        sys.executable,
        "-m",
        "app.train",
        "--data-path",
        data_path_text,
        "--sheet-name",
        sheet_name,
    ]
    with st.spinner("Executando treinamento..."):
        completed = subprocess.run(cmd, capture_output=True, text=True)
    st.session_state.setdefault("train_logs", []).append(
        {"cmd": cmd, "stdout": completed.stdout, "stderr": completed.stderr, "returncode": completed.returncode}
    )
    if completed.returncode == 0:
        st.success("Treinamento finalizado com sucesso.")
        load_recent_runs.clear()
    else:
        st.error("Treinamento terminou com erros. Veja os logs abaixo.")

if st.session_state.get("train_logs"):
    with st.expander("Logs de treinamento", expanded=False):
        for item in reversed(st.session_state["train_logs"]):
            st.code("$ " + " ".join(item["cmd"]))
            if item["stdout"]:
                st.text(item["stdout"])
            if item["stderr"]:
                st.text(item["stderr"])
            st.text(f"returncode: {item['returncode']}")


st.header("Predição de novos dados")

uploaded = st.file_uploader("Carregue dados para inferência (CSV ou XLSX)", type=["csv", "xlsx", "xls"])

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_input = pd.read_csv(uploaded, sep=";", engine="python")
        else:
            df_input = pd.read_excel(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Falha ao ler o arquivo: {exc}")
        df_input = None

    if df_input is not None:
        st.subheader("Pré-visualização")
        st.dataframe(df_input.head(), use_container_width=True)

        try:
            results = predict_from_df(df_input, source="streamlit_upload")
        except ModelNotFoundError:
            st.warning("Modelo ainda não treinado. Execute o treinamento antes de prever.")
            results = None
        except Exception as exc:  # noqa: BLE001
            st.error(f"Falha ao gerar predições: {exc}")
            results = None

        if results is not None:
            st.subheader("Resultados")
            display_df = results.copy()
            if "y_pred_display" in display_df.columns:
                display_df["y_pred"] = display_df.pop("y_pred_display")
            if "y_true_display" in display_df.columns:
                display_df["y_true"] = display_df.pop("y_true_display")
            st.dataframe(display_df, use_container_width=True)

            if "y_true" in results.columns:
                report = classification_report(
                    results["y_true"],
                    results["y_pred"],
                    labels=CLASS_LABELS,
                    zero_division=0,
                    output_dict=True,
                )
                st.write("Métricas vs verdadeiro (se disponível):")
                st.dataframe(pd.DataFrame(report).T)

            csv_buffer = io.StringIO()
            display_df.to_csv(csv_buffer, index=False, sep=";")
            st.download_button(
                label="Baixar CSV com predições",
                data=csv_buffer.getvalue(),
                file_name="predicoes.csv",
                mime="text/csv",
            )


with st.expander("🔎 Últimas predições registradas", expanded=False):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT run_id, source, y_true, y_pred, probabilities_json, payload_json "
            "FROM predictions ORDER BY id DESC LIMIT 10"
        ).fetchall()
    if not rows:
        st.info("Nenhuma predição registrada ainda.")
    else:
        table = []
        for row in rows:
            probs = {}
            if row["probabilities_json"]:
                probs = json.loads(row["probabilities_json"])
            entry = {
                "run_id": row["run_id"],
                "source": row["source"],
                "y_true": to_display_label(row["y_true"]),
                "y_pred": to_display_label(row["y_pred"]),
            }
            for label in CLASS_LABELS:
                entry[f"proba_{label}"] = probs.get(label)
            table.append(entry)
        st.table(pd.DataFrame(table))
