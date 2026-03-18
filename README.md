# Projeto Desastres

Pipeline completo para classificação de intensidade de desastres transformando as 10 classes originais (`cluster_H`) em quatro categorias agregadas: `nivel_I` (classe 1), `nivel_II` (classe 2), `nivel_III` (classe 3) e `raro` (classes 4-10). Inclui treino com validação cruzada estratificada repetida, registro de execuções/metrics/predictions em SQLite e interface web no Streamlit para reutilização.

## Pré-requisitos

- Python 3.10 ou superior.
- A base `data/desastres.xlsx` (ou CSV equivalente) deve estar presente com as colunas de features "cruas" abaixo:
  - `População`
  - `Área_km`
  - `Receita.Anual`
  - `Orçamento.anual`
  - `Danos.e.Prejuízos`
  - `Afetados`
  - `Precipitação.pluviométrica`
  - `Densidade.populacional`
  - `Capacidade.de.investimento.na.resposta.ao.desastre`
  - `Capacidade.de.projeção.financeira`
  - `Densidade.populacional.de.afetados`
  - Mais a coluna alvo `cluster_H`.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Variáveis de ambiente (opcional)

Copie `.env.example` para `.env` e ajuste caminhos se desejar sobrescrever os padrões:

```env
DATA_PATH=data/desastres.xlsx
DB_PATH=projeto-desastres.db
```

Outras variáveis: `MODEL_PATH`, `SHEET_NAME`, `CV_SPLITS`, `CV_REPEATS` e `RANDOM_STATE`.

## Treinamento do modelo

```bash
python -m app.train
```

Parâmetros opcionais:

- `--data-path PATH` para usar outro arquivo (CSV com `;`, XLSX, XLS ou Parquet).
- `--sheet-name NOME` ao usar XLSX com aba diferente.
- `--model-path PATH` para salvar o pipeline em outro local.

Durante o treinamento o sistema:

1. Valida se todas as features exigidas estão disponíveis.
2. Executa `RepeatedStratifiedKFold` (padrão 5×5) com `RandomForestClassifier` e armazena os scores por dobra.
3. Salva as métricas agregadas na tabela `runs` e as dobras na tabela `cv_metrics`.
4. Ajusta o pipeline com todos os dados, persiste em `app/model.joblib` (inclui `run_id` do treino) e registra as predições no banco (`predictions`, `source='train'`).

## Inferência em lote

Use o módulo `app.predict` para aplicar o modelo salvo:

```python
from pathlib import Path
from app.predict import predict_from_file

df_preds = predict_from_file(Path("data/novos_dados.xlsx"), source="script")
print(df_preds.head())
```

As predições (y, probabilidades por classe, payload) ficam registradas automaticamente na tabela `predictions`. Caso o arquivo possua `cluster_H` ou rótulos equivalentes (`Nível I/II/III` ou `raro`), as métricas permitem comparação na própria planilha retornada.

## Relatórios via CLI

```bash
python -m app.metrics_report --limit 5 --show-predictions
```

Mostra as execuções mais recentes e, opcionalmente, as últimas predições logadas.

## Interface web (Streamlit)

```bash
streamlit run app/app_streamlit.py
```

Funcionalidades:

- Botão para executar (re)treinamentos apontando para qualquer base local.
- Upload de arquivos CSV/XLSX para gerar novas predições, com download do resultado imediatamente.
- Visualização das métricas agregadas e histórico de predições recentes gravadas no banco.

## Aplicação web (FastAPI + Jinja2)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Acesse:

- `http://localhost:8000/` para a interface web
- `http://localhost:8000/docs` para a documentação Swagger
- `http://localhost:8000/health` para healthcheck
- `http://localhost:8000/model-info` para metadados do modelo

## Docker

Build local:

```bash
docker build -t projeto-desastres .
```

Execução local:

```bash
docker run --rm -p 8000:8000 projeto-desastres
```

Com variáveis de ambiente:

```bash
docker run --rm -p 8000:8000 \
  -e PORT=8000 \
  -e MODEL_PATH=/app/app/model.joblib \
  projeto-desastres
```

Decisão desta etapa:

- `app/model.joblib` vai dentro da imagem para simplificar o runtime do MVP.
- `data/` não vai para a imagem porque não é necessário para servir a API/web em produção básica.
- Se for preciso treinar dentro do container no futuro, a pasta de dados pode ser montada por volume ou copiada em uma imagem específica de treino.

## Banco de dados

O arquivo SQLite padrão (`projeto-desastres.db`) possui as tabelas:

- `runs`: metadados de cada treino e resumo de métricas.
- `cv_metrics`: scores por dobra da validação cruzada.
- `predictions`: registro de todas as predições (treino e inferências), incluindo probabilidades e payload das features. Há uma restrição para evitar duplicidade (`run_id + source + row_hash`).

## Estrutura de diretórios

```
projeto-desastres/
├─ data/
│  └─ desastres.xlsx
├─ app/
│  ├─ __init__.py
│  ├─ app_streamlit.py
│  ├─ config.py
│  ├─ db.py
│  ├─ features.py
│  ├─ metrics_report.py
│  ├─ predict.py
│  ├─ schema.sql
│  ├─ train.py
│  └─ model.joblib  # gerado após o primeiro treino
├─ requirements.txt
├─ README.md
└─ .env.example
```

## Próximos passos sugeridos

- Versionar múltiplos modelos e permitir comparação direta via Streamlit.
- Adicionar testes automatizados para funções críticas (`train`, `predict`).
- Integrar gráficos de distribuição das classes previstas na interface.
