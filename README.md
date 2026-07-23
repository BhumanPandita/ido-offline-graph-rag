# Offline IFS Graph RAG prototype

This prototype uses Azure OpenAI only while building the knowledge graph. The
resulting bundle is queried without a network connection using:

- a SQLite knowledge graph with source/page provenance;
- `BAAI/bge-small-en-v1.5` through ONNX-based FastEmbed;
- optional GGUF answer generation through llama.cpp.

LlamaIndex is intentionally not part of the phone bundle. It is useful for
server and laptop orchestration, but the mobile artifact needs a much smaller,
portable runtime.

## 1. Install

Use Python 3.10–3.12 in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, fill in the values, and keep the file local:

```bash
cp .env.example .env
```

The CLI loads `.env` automatically. Alternatively, export the four variables
in Git Bash with `export NAME=value`. Do not commit `.env` or company
documents; use your approved secret-management process for the values.

## Azure connectivity test

Before running a full graph build, test the Azure resource and deployment:

```bash
python test_azure_openai.py
```

The deployment value must be the custom deployment name shown in Azure, not
necessarily the underlying model name such as `gpt-4o`. You can also pass the
values explicitly:

```bash
python test_azure_openai.py \
  --endpoint "https://your-resource.openai.azure.com/" \
  --api-key "YOUR_KEY" \
  --deployment "YOUR_DEPLOYMENT_NAME" \
  --api-version "2024-10-21"
```

## 2. Build the graph

Put `.pdf`, `.txt`, or `.md` IFS documentation under `data/`, then run:

```bash
python offline_graph_rag.py build \
  --input data \
  --output ifs_mobile_bundle
```

The command sends document chunks to the configured Azure GPT-4o deployment.
It rejects extracted relationships whose evidence quote cannot be found
verbatim in the source chunk.

The output contains:

```text
ifs_mobile_bundle/
  graph.sqlite3
  embeddings.npy
  manifest.json
  models/embeddings/
  graph_dashboard.html
```

The build also creates `graph_dashboard.html` automatically. It is a
self-contained offline dashboard for inspecting communities, entities,
relationships, verbatim evidence, source files, and pages. Add
`--skip-dashboard` only when you do not want this laptop inspection artifact in
the bundle. If Azure extracts no grounded relationships, the retrieval bundle
is still created and the dashboard is skipped with a clear message.

### If Hugging Face downloads are blocked

Use the built-in `local-hash-v1` embedding option. It downloads no model and
adds no model files to the bundle:

```bash
python offline_graph_rag.py build \
  --input data \
  --output ifs_mobile_bundle \
  --embedding-model local-hash-v1
```

This is a lexical matcher, not a semantic language model. It works best when a
crew member uses the IFS terms found in the documentation. Use it to validate
the end-to-end Graph RAG pipeline on a restricted network; compare it with BGE
embeddings later if company-approved model access becomes available.

### Use a model downloaded manually

This project uses FastEmbed's ONNX version of BGE, not the original PyTorch
model files. From a browser on a machine that can access Hugging Face, download
every file from the `Qdrant/bge-small-en-v1.5-onnx-Q` repository into one local
folder, then transfer that whole folder to the company laptop. Do not download
only one `.onnx` file: the tokenizer and configuration files are also required.

For example, if the transferred folder is `models/bge-small-en-v1.5-onnx-Q/`:

```bash
python offline_graph_rag.py build \
  --input data \
  --output ifs_mobile_bundle \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --embedding-path models/bge-small-en-v1.5-onnx-Q
```

`--embedding-path` copies the model into `ifs_mobile_bundle/models/embeddings/`.
The build and every later offline query use that copied local model, so they do
not contact Hugging Face.

## 3. Query offline

Evidence-only mode is the safest and smallest first mobile experience:

```bash
python offline_graph_rag.py query \
  --bundle ifs_mobile_bundle \
  --question "How do I resolve an open task in the operating module?"
```

For a laptop proof-of-concept with a local GGUF model:

```bash
python -m pip install llama-cpp-python
python offline_graph_rag.py query \
  --bundle ifs_mobile_bundle \
  --llm-model models/smollm2-360m-instruct-q4_k_m.gguf \
  --question "How do I resolve an open task in the operating module?"
```

Check the actual package budget:

```bash
python offline_graph_rag.py inspect \
  --bundle ifs_mobile_bundle \
  --llm-model models/smollm2-360m-instruct-q4_k_m.gguf
```

## 4. Visualize the knowledge graph

Every new build already contains `ifs_mobile_bundle/graph_dashboard.html`.
Open it directly in Chrome or Edge; it makes no Azure or other network call.

You can also generate the dashboard from an existing bundle without rebuilding
the index or calling Azure:

```bash
python offline_graph_rag.py visualize \
  --bundle ifs_mobile_bundle \
  --output ifs_graph_dashboard.html
```

The dashboard detects Louvain communities within the displayed graph and
colors entities by community. It includes entity search; community, entity
type, relationship, and source filters; a cross-community view; zoom and pan;
clickable nodes and connections; community summaries; and an evidence table
with document/page provenance.

The default view shows up to 500 of the highest-connected entities. For a
focused view around a particular module, screen, process, task, or role:

```bash
python offline_graph_rag.py visualize \
  --bundle ifs_mobile_bundle \
  --entity "work order" \
  --depth 2 \
  --output work_order_dashboard.html
```

Communities are calculated for the entities included in that dashboard, so a
focused or truncated view can have different community boundaries from the
full graph. Treat the HTML as company data because it contains extracted names
and source quotations.

## Mobile boundary

The Python CLI proves indexing and retrieval logic. For an Android/iOS release,
ship the generated SQLite database and vectors, run the embedding ONNX model
with the platform's ONNX Runtime Mobile binding, and run the GGUF model with
llama.cpp's native mobile binding. Embedding and generation must be tested on
the target crew devices before fixing the 600 MB allocation.

For a first budget, reserve roughly 67 MB for the embedding model, about
270 MB for a Q4-quantized SmolLM2-360M model, and the remainder for the graph,
native runtimes, tokenizer files, UI, and application code. The graph and model
licenses still need company legal/security approval before distribution.
