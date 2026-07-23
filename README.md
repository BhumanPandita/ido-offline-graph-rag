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
```

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

Generate a self-contained HTML visualization with no Azure or other network
call:

```bash
python offline_graph_rag.py visualize \
  --bundle ifs_mobile_bundle \
  --output ifs_graph.html
```

The default view shows up to 150 of the highest-connected entities. For a
focused view around a particular module, screen, process, task, or role:

```bash
python offline_graph_rag.py visualize \
  --bundle ifs_mobile_bundle \
  --entity "work order" \
  --depth 2 \
  --output work_order_graph.html
```

Open the resulting HTML file in a browser. Nodes are colored by entity type.
Hover over a relationship line's midpoint to see its direction, predicate,
verbatim evidence, source document, and page. Treat the HTML as company data
because it contains extracted names and source quotations.

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
