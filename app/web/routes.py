"""Rotas web simples para operação manual do sistema."""
from __future__ import annotations

from pathlib import Path
import io
from typing import Any, Dict
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from app.core.config import SHEET_NAME
from app.core.features import FEATURES, MissingColumnsError, to_display_label
from app.core.model import ModelNotFoundError
from app.services.prediction_service import (
    build_prediction_items,
    get_model_info,
    predict_from_dataframe,
)
from app.utils.dataframe_io import (
    EmptyFileError,
    FileParsingError,
    UnsupportedFileFormatError,
    load_uploaded_dataframe,
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
router = APIRouter(include_in_schema=False)
_DOWNLOAD_CACHE: dict[str, bytes] = {}


def _base_context(request: Request, **extra: Any) -> Dict[str, Any]:
    """Cria o contexto base compartilhado pelos templates."""
    return {
        "request": request,
        "features": FEATURES,
        **extra,
    }


def _parse_manual_form(form_data) -> pd.DataFrame:
    """Converte os campos do formulário manual em DataFrame."""
    payload: Dict[str, float] = {}
    errors = []
    for feature in FEATURES:
        raw_value = str(form_data.get(feature, "")).strip()
        if raw_value == "":
            errors.append(f"O campo '{feature}' é obrigatório.")
            continue
        normalized = raw_value.replace(",", ".")
        try:
            payload[feature] = float(normalized)
        except ValueError:
            errors.append(f"O campo '{feature}' deve ser numérico.")
    if errors:
        raise ValueError(" ".join(errors))
    return pd.DataFrame([payload])


def _build_probabilities_for_view(item: Dict[str, Any]) -> list[dict[str, Any]]:
    """Transforma o mapa de probabilidades em lista amigável ao template."""
    probabilities = item.get("probabilities", {}) or {}
    return [
        {
            "label": label,
            "display": to_display_label(label),
            "value": probabilities.get(label),
        }
        for label in probabilities
    ]


@router.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    """Exibe a página inicial da interface web."""
    return templates.TemplateResponse("index.html", _base_context(request))


@router.get("/web", response_class=HTMLResponse)
def web_home() -> RedirectResponse:
    """Redireciona o atalho `/web` para a home."""
    return RedirectResponse(url="/", status_code=302)


@router.get("/web/model-status", response_class=HTMLResponse)
def model_status(request: Request) -> HTMLResponse:
    """Exibe o estado atual do artefato do modelo."""
    try:
        model_info = get_model_info()
        return templates.TemplateResponse(
            "model_status.html",
            _base_context(request, model_info=model_info, error_message=None),
        )
    except ModelNotFoundError as exc:
        return templates.TemplateResponse(
            "model_status.html",
            _base_context(
                request,
                model_info=None,
                error_message=str(exc),
            ),
            status_code=404,
        )


@router.get("/web/predict-form", response_class=HTMLResponse)
def predict_form_page(request: Request) -> HTMLResponse:
    """Exibe o formulário manual de predição."""
    return templates.TemplateResponse(
        "predict_form.html",
        _base_context(
            request,
            form_values={feature: "" for feature in FEATURES},
            prediction=None,
            error_message=None,
        ),
    )


@router.post("/web/predict-form", response_class=HTMLResponse)
async def predict_form_submit(request: Request) -> HTMLResponse:
    """Processa a submissão manual de features pelo navegador."""
    form = await request.form()
    form_values = {feature: str(form.get(feature, "")) for feature in FEATURES}
    try:
        df = _parse_manual_form(form)
        result = predict_from_dataframe(
            df,
            source="web_form",
            tracker=None,
        )
        item = build_prediction_items(result, input_columns=list(df.columns))[0]
        prediction = {
            "class_label": item["y_pred"],
            "class_display": item["y_pred_display"],
            "probabilities": _build_probabilities_for_view(item),
        }
        return templates.TemplateResponse(
            "predict_form.html",
            _base_context(
                request,
                form_values=form_values,
                prediction=prediction,
                error_message=None,
            ),
        )
    except (ValueError, MissingColumnsError, ModelNotFoundError) as exc:
        status_code = 404 if isinstance(exc, ModelNotFoundError) else 400
        return templates.TemplateResponse(
            "predict_form.html",
            _base_context(
                request,
                form_values=form_values,
                prediction=None,
                error_message=str(exc),
            ),
            status_code=status_code,
        )


@router.get("/web/predict-file", response_class=HTMLResponse)
def predict_file_page(request: Request) -> HTMLResponse:
    """Exibe a página de upload para predição em lote."""
    return templates.TemplateResponse(
        "predict_file.html",
        _base_context(
            request,
            result=None,
            preview_rows=[],
            download_token=None,
            error_message=None,
        ),
    )


@router.post("/web/predict-file", response_class=HTMLResponse)
async def predict_file_submit(request: Request, file: UploadFile) -> HTMLResponse:
    """Processa upload de CSV/XLSX e renderiza o resultado."""
    try:
        content = await file.read()
        df = load_uploaded_dataframe(file.filename or "", content, sheet_name=SHEET_NAME)
        result_df = predict_from_dataframe(
            df,
            source="web_file_upload",
            tracker=None,
        )

        preview_rows = build_prediction_items(result_df.head(20), input_columns=list(df.columns))
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False, sep=";")
        download_token = uuid4().hex
        _DOWNLOAD_CACHE[download_token] = csv_buffer.getvalue().encode("utf-8")

        result = {
            "filename": file.filename or "",
            "total_rows": len(df),
            "total_processed": len(result_df),
            "columns_received": list(df.columns),
            "required_columns": list(FEATURES),
        }
        return templates.TemplateResponse(
            "predict_file.html",
            _base_context(
                request,
                result=result,
                preview_rows=preview_rows,
                download_token=download_token,
                error_message=None,
            ),
        )
    except (
        EmptyFileError,
        UnsupportedFileFormatError,
        FileParsingError,
        MissingColumnsError,
        ModelNotFoundError,
        ValueError,
    ) as exc:
        status_code = 404 if isinstance(exc, ModelNotFoundError) else 400
        if isinstance(exc, MissingColumnsError):
            status_code = 422
        return templates.TemplateResponse(
            "predict_file.html",
            _base_context(
                request,
                result=None,
                preview_rows=[],
                download_token=None,
                error_message=str(exc),
            ),
            status_code=status_code,
        )


@router.get("/web/download/{token}")
def download_predictions(token: str) -> Response:
    """Disponibiliza o CSV processado para download."""
    content = _DOWNLOAD_CACHE.pop(token, None)
    if content is None:
        return Response(status_code=404, content="Arquivo não encontrado para download.")
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="predicoes_processadas.csv"'},
    )
