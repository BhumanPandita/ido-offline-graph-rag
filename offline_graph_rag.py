#!/usr/bin/env python3
"""Build an IFS knowledge graph online and query it fully offline.

Build time:
    Documents -> Azure OpenAI structured extraction -> SQLite graph
              -> open-source FastEmbed vectors

Query time:
    Question -> local FastEmbed retrieval -> graph expansion
             -> optional local GGUF answer model

Azure is imported and called only by the ``build`` command.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from pydantic import BaseModel, Field


DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LOCAL_HASH_EMBED_MODEL = "local-hash-v1"
SUPPORTED_SUFFIXES = {".md", ".pdf", ".txt"}

EntityType = Literal[
    "MODULE",
    "SCREEN",
    "PROCESS",
    "TASK",
    "ROLE",
    "FIELD",
    "STATUS",
    "ERROR",
    "DOCUMENT",
    "SYSTEM",
    "EQUIPMENT",
    "LOCATION",
    "OTHER",
]

RelationType = Literal[
    "PART_OF",
    "REQUIRES",
    "PERFORMED_BY",
    "USES",
    "OPENS",
    "UPDATES",
    "CREATES",
    "RESOLVES",
    "TRIGGERS",
    "DEPENDS_ON",
    "APPLIES_TO",
    "LOCATED_IN",
    "RELATED_TO",
]


class ExtractedRelation(BaseModel):
    subject: str = Field(description="Exact or canonical subject name")
    subject_type: EntityType
    predicate: RelationType
    object: str = Field(description="Exact or canonical object name")
    object_type: EntityType
    evidence_quote: str = Field(
        description="Short verbatim quote from the supplied text proving the relation"
    )


class ChunkExtraction(BaseModel):
    relations: list[ExtractedRelation] = Field(default_factory=list)


class Chunk(BaseModel):
    id: str
    source: str
    page: int | None
    text: str


EXTRACTION_SYSTEM_PROMPT = """You extract a grounded knowledge graph from IFS
operating-module documentation for airline crew.

Rules:
1. Extract only facts explicitly supported by the supplied text.
2. Every relation must include a short VERBATIM evidence quote copied from it.
3. Prefer operational entities: modules, screens, processes, tasks, roles,
   fields, statuses, errors, documents, systems, equipment, and locations.
4. Use only the entity types and predicates allowed by the response schema.
5. Do not infer missing steps, causes, permissions, or safety instructions.
6. Use RELATED_TO only when no more precise allowed predicate applies.
7. Return no relation when the text does not support one.
"""


def stable_id(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def normalized_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def canonical_name(value: str) -> str:
    return normalized_space(value).casefold()


def normalize_azure_endpoint(endpoint: str) -> str:
    """Return the Azure resource base URL expected by AzureOpenAI."""
    endpoint = endpoint.strip().rstrip("/")
    for suffix in ("/openai/v1", "/openai"):
        if endpoint.lower().endswith(suffix):
            endpoint = endpoint[: -len(suffix)].rstrip("/")
    if "/deployments/" in endpoint.lower():
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT must be the resource URL only; "
            "remove /openai/deployments/..."
        )
    return endpoint


def iter_source_pages(input_dir: Path) -> Iterable[tuple[str, int | None, str]]:
    files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise ValueError(f"No .txt, .md, or .pdf files found under {input_dir}")

    for path in files:
        source = path.relative_to(input_dir).as_posix()
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("Install pypdf to read PDF files") from exc
            for page_number, page in enumerate(PdfReader(path).pages, start=1):
                text = page.extract_text() or ""
                if normalized_space(text):
                    yield source, page_number, text
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            if normalized_space(text):
                yield source, None, text


def split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if max_chars <= 0 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("Require max_chars > overlap_chars >= 0")

    text = normalized_space(text)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = hard_end
        if hard_end < len(text):
            boundary = max(
                text.rfind(". ", start, hard_end),
                text.rfind("; ", start, hard_end),
                text.rfind(" ", start, hard_end),
            )
            if boundary > start + max_chars // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def load_chunks(input_dir: Path, max_chars: int, overlap_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, page, text in iter_source_pages(input_dir):
        for position, chunk_text in enumerate(
            split_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
        ):
            chunks.append(
                Chunk(
                    id=stable_id(source, page, position, chunk_text),
                    source=source,
                    page=page,
                    text=chunk_text,
                )
            )
    return chunks


class AzureGraphExtractor:
    def __init__(self) -> None:
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError(
                "Install python-dotenv or export the Azure variables in Git Bash"
            ) from exc
        load_dotenv()

        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to use Azure extraction") from exc

        required = {
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
            "AZURE_OPENAI_DEPLOYMENT": (
                os.getenv("AZURE_OPENAI_DEPLOYMENT")
                or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
                or os.getenv("AZURE_OPENAI_MODEL_NAME")
            ),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                f"Missing environment variables: {', '.join(missing)}. "
                "Create a .env file from .env.example or export these variables."
            )

        self.deployment = required["AZURE_OPENAI_DEPLOYMENT"]
        self.client = AzureOpenAI(
            azure_endpoint=normalize_azure_endpoint(required["AZURE_OPENAI_ENDPOINT"]),
            api_key=required["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            max_retries=5,
        )

    def extract(self, chunk: Chunk) -> ChunkExtraction:
        completion = self.client.beta.chat.completions.parse(
            model=self.deployment,
            temperature=0,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Source: {chunk.source}\n"
                        f"Page: {chunk.page or 'not available'}\n\n"
                        f"Text:\n{chunk.text}"
                    ),
                },
            ],
            response_format=ChunkExtraction,
        )
        message = completion.choices[0].message
        if message.parsed is None:
            raise RuntimeError(f"Azure extraction failed: {message.refusal or 'no parsed output'}")
        return message.parsed


def relation_is_grounded(relation: ExtractedRelation, chunk_text: str) -> bool:
    quote = normalized_space(relation.evidence_quote)
    return bool(
        canonical_name(relation.subject)
        and canonical_name(relation.object)
        and canonical_name(relation.subject) != canonical_name(relation.object)
        and len(quote) >= 8
        and quote.casefold() in normalized_space(chunk_text).casefold()
    )


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = DELETE;
        PRAGMA foreign_keys = ON;

        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            page INTEGER,
            text TEXT NOT NULL
        );

        CREATE TABLE entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            type TEXT NOT NULL,
            UNIQUE(canonical_name, type)
        );

        CREATE TABLE relations (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES entities(id),
            target_id TEXT NOT NULL REFERENCES entities(id),
            predicate TEXT NOT NULL,
            chunk_id TEXT NOT NULL REFERENCES chunks(id),
            evidence_quote TEXT NOT NULL,
            UNIQUE(source_id, target_id, predicate, chunk_id, evidence_quote)
        );

        CREATE TABLE retrieval_items (
            position INTEGER PRIMARY KEY,
            kind TEXT NOT NULL CHECK(kind IN ('chunk', 'relation')),
            ref_id TEXT NOT NULL,
            text TEXT NOT NULL
        );

        CREATE INDEX relations_source_idx ON relations(source_id);
        CREATE INDEX relations_target_idx ON relations(target_id);
        CREATE INDEX relations_chunk_idx ON relations(chunk_id);
        """
    )


def upsert_entity(
    connection: sqlite3.Connection, name: str, entity_type: str
) -> str:
    name = normalized_space(name)
    entity_id = stable_id(entity_type, canonical_name(name))
    connection.execute(
        """
        INSERT OR IGNORE INTO entities(id, name, canonical_name, type)
        VALUES (?, ?, ?, ?)
        """,
        (entity_id, name, canonical_name(name), entity_type),
    )
    return entity_id


def add_extraction(
    connection: sqlite3.Connection, chunk: Chunk, extraction: ChunkExtraction
) -> tuple[int, int]:
    accepted = 0
    discarded = 0
    for relation in extraction.relations:
        if not relation_is_grounded(relation, chunk.text):
            discarded += 1
            continue
        source_id = upsert_entity(connection, relation.subject, relation.subject_type)
        target_id = upsert_entity(connection, relation.object, relation.object_type)
        evidence = normalized_space(relation.evidence_quote)
        relation_id = stable_id(
            source_id, relation.predicate, target_id, chunk.id, evidence
        )
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO relations(
                id, source_id, target_id, predicate, chunk_id, evidence_quote
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                relation_id,
                source_id,
                target_id,
                relation.predicate,
                chunk.id,
                evidence,
            ),
        )
        accepted += cursor.rowcount
    return accepted, discarded


def create_retrieval_items(connection: sqlite3.Connection) -> list[str]:
    items: list[tuple[str, str, str]] = []
    for row in connection.execute("SELECT id, source, page, text FROM chunks ORDER BY id"):
        location = f"{row[1]} page {row[2]}" if row[2] else row[1]
        items.append(("chunk", row[0], f"Source {location}: {row[3]}"))

    relation_sql = """
        SELECT r.id, s.name, s.type, r.predicate, t.name, t.type,
               r.evidence_quote, c.source, c.page
        FROM relations r
        JOIN entities s ON s.id = r.source_id
        JOIN entities t ON t.id = r.target_id
        JOIN chunks c ON c.id = r.chunk_id
        ORDER BY r.id
    """
    for row in connection.execute(relation_sql):
        location = f"{row[7]} page {row[8]}" if row[8] else row[7]
        text = (
            f"{row[1]} ({row[2]}) {row[3]} {row[4]} ({row[5]}). "
            f"Evidence from {location}: {row[6]}"
        )
        items.append(("relation", row[0], text))

    connection.executemany(
        "INSERT INTO retrieval_items(position, kind, ref_id, text) VALUES (?, ?, ?, ?)",
        [(position, kind, ref_id, text) for position, (kind, ref_id, text) in enumerate(items)],
    )
    return [item[2] for item in items]


class LocalHashEmbedder:
    """Zero-download lexical embeddings for restricted/offline environments."""

    dimensions = 384

    def embed(self, texts: list[str]) -> Iterable[np.ndarray]:
        for text in texts:
            vector = np.zeros(self.dimensions, dtype=np.float32)
            tokens = re.findall(r"[a-z0-9][a-z0-9_/-]*", text.casefold())
            features = tokens + [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, byteorder="little")
                index = value % self.dimensions
                vector[index] += 1.0 if value & (1 << 63) else -1.0
            yield vector


def get_embedder(model_name: str, cache_dir: Path, offline: bool):
    if model_name == LOCAL_HASH_EMBED_MODEL:
        return LocalHashEmbedder()
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("Install fastembed to create or query embeddings") from exc
    return TextEmbedding(
        model_name=model_name,
        cache_dir=str(cache_dir),
        local_files_only=offline,
    )


def embed_texts(embedder, texts: list[str]) -> np.ndarray:
    if not texts:
        raise ValueError("Cannot embed an empty graph")
    matrix = np.asarray(list(embedder.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def build_bundle(args: argparse.Namespace) -> None:
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Choose a new directory."
        )

    chunks = load_chunks(input_dir, args.chunk_chars, args.overlap_chars)
    extractor = AzureGraphExtractor()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Keep build files outside locations such as Downloads that may be scanned
    # or synchronized by Windows while SQLite is being finalized.
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", ignore_cleanup_errors=True
    ) as temp_name:
        temp_dir = Path(temp_name)
        database_path = temp_dir / "graph.sqlite3"
        model_cache = temp_dir / "models" / "embeddings"
        # Download the local embedding model before any paid Azure extraction.
        # A corporate-network/Hugging Face failure should fail fast here.
        print("Preparing the local embedding model...", flush=True)
        embedder = get_embedder(args.embedding_model, model_cache, offline=False)

        connection = sqlite3.connect(database_path)
        initialize_database(connection)
        connection.executemany(
            "INSERT INTO chunks(id, source, page, text) VALUES (?, ?, ?, ?)",
            [(chunk.id, chunk.source, chunk.page, chunk.text) for chunk in chunks],
        )

        accepted = 0
        discarded = 0
        for number, chunk in enumerate(chunks, start=1):
            extraction = extractor.extract(chunk)
            added, rejected = add_extraction(connection, chunk, extraction)
            accepted += added
            discarded += rejected
            print(
                f"\rExtracting {number}/{len(chunks)} "
                f"(relations: {accepted}, rejected: {discarded})",
                end="",
                flush=True,
            )
        print()

        retrieval_texts = create_retrieval_items(connection)
        connection.commit()
        # Close SQLite before moving the completed temporary directory.
        connection.close()
        del connection

        embeddings = embed_texts(embedder, retrieval_texts)
        # FastEmbed/ONNX can keep model files open on Windows. Release it
        # before moving the directory into the final bundle.
        del embedder
        gc.collect()
        np.save(temp_dir / "embeddings.npy", embeddings.astype(np.float16))

        manifest = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": args.embedding_model,
            "chunks": len(chunks),
            "relations": accepted,
            "rejected_ungrounded_relations": discarded,
            "embedding_dimensions": int(embeddings.shape[1]),
            "embedding_dtype": "float16",
            "azure_used_at_query_time": False,
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        shutil.move(str(temp_dir), str(output_dir))

    size_mb = directory_size(output_dir) / (1024 * 1024)
    print(f"Built {output_dir} ({size_mb:.1f} MiB)")


def top_positions(embeddings: np.ndarray, query_vector: np.ndarray, top_k: int) -> list[int]:
    scores = embeddings.astype(np.float32) @ query_vector.astype(np.float32)
    top_k = min(max(top_k, 1), len(scores))
    return np.argsort(scores)[-top_k:][::-1].tolist()


def collect_context(
    connection: sqlite3.Connection, positions: list[int], graph_depth: int
) -> list[sqlite3.Row]:
    connection.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in positions)
    seed_items = list(
        connection.execute(
            f"SELECT * FROM retrieval_items WHERE position IN ({placeholders})",
            positions,
        )
    )
    relation_ids = {row["ref_id"] for row in seed_items if row["kind"] == "relation"}
    chunk_ids = {row["ref_id"] for row in seed_items if row["kind"] == "chunk"}

    if chunk_ids:
        chunk_marks = ",".join("?" for _ in chunk_ids)
        relation_ids.update(
            row[0]
            for row in connection.execute(
                f"SELECT id FROM relations WHERE chunk_id IN ({chunk_marks})",
                tuple(chunk_ids),
            )
        )

    frontier: set[str] = set()
    if relation_ids:
        relation_marks = ",".join("?" for _ in relation_ids)
        for row in connection.execute(
            f"""
            SELECT source_id, target_id FROM relations
            WHERE id IN ({relation_marks})
            """,
            tuple(relation_ids),
        ):
            frontier.update(row)

    visited_entities: set[str] = set()
    for _ in range(max(graph_depth, 0)):
        frontier -= visited_entities
        if not frontier:
            break
        visited_entities.update(frontier)
        entity_marks = ",".join("?" for _ in frontier)
        new_frontier: set[str] = set()
        for row in connection.execute(
            f"""
            SELECT id, source_id, target_id FROM relations
            WHERE source_id IN ({entity_marks}) OR target_id IN ({entity_marks})
            """,
            tuple(frontier) * 2,
        ):
            relation_ids.add(row["id"])
            new_frontier.update((row["source_id"], row["target_id"]))
        frontier = new_frontier

    result: list[sqlite3.Row] = []
    if chunk_ids:
        marks = ",".join("?" for _ in chunk_ids)
        result.extend(
            connection.execute(
                f"""
                SELECT 'chunk' AS kind, id, NULL AS subject,
                       NULL AS predicate, NULL AS object, text AS evidence_quote,
                       source, page, text
                FROM chunks
                WHERE id IN ({marks})
                """,
                tuple(chunk_ids),
            )
        )
    if relation_ids:
        marks = ",".join("?" for _ in relation_ids)
        result.extend(
            connection.execute(
                f"""
                SELECT 'relation' AS kind, r.id, s.name AS subject,
                       r.predicate, t.name AS object, r.evidence_quote,
                       c.source, c.page, c.text
                FROM relations r
                JOIN entities s ON s.id = r.source_id
                JOIN entities t ON t.id = r.target_id
                JOIN chunks c ON c.id = r.chunk_id
                WHERE r.id IN ({marks})
                """,
                tuple(relation_ids),
            )
        )
    return result


def format_evidence(rows: list[sqlite3.Row], max_chars: int = 6500) -> str:
    blocks: list[str] = []
    used: set[tuple[str, int | None, str]] = set()
    total = 0
    for row in rows:
        key = (row["source"], row["page"], row["evidence_quote"])
        if key in used:
            continue
        used.add(key)
        location = row["source"]
        if row["page"]:
            location += f", page {row['page']}"
        if row["kind"] == "relation":
            block = (
                f"[{location}]\n"
                f"Graph fact: {row['subject']} --{row['predicate']}--> {row['object']}\n"
                f"Evidence: {row['evidence_quote']}"
            )
        else:
            block = f"[{location}]\nSource text: {row['evidence_quote']}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def generate_local_answer(model_path: Path, question: str, evidence: str) -> str:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "Install llama-cpp-python to generate with a local GGUF model"
        ) from exc
    if not model_path.is_file():
        raise FileNotFoundError(f"GGUF model not found: {model_path}")

    llm = Llama(model_path=str(model_path), n_ctx=4096, verbose=False)
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied IFS evidence. Cite sources in "
                    "square brackets exactly as shown. If the evidence is insufficient, "
                    "say so. Never invent an operational or safety step."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nIFS evidence:\n{evidence}",
            },
        ],
        temperature=0.1,
        max_tokens=350,
    )
    return response["choices"][0]["message"]["content"].strip()


def query_bundle(args: argparse.Namespace) -> None:
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    embeddings = np.load(bundle / "embeddings.npy", mmap_mode="r")
    embedder = get_embedder(
        manifest["embedding_model"],
        bundle / "models" / "embeddings",
        offline=True,
    )
    query_vector = embed_texts(embedder, [args.question])[0]
    positions = top_positions(embeddings, query_vector, args.top_k)

    connection = sqlite3.connect(bundle / "graph.sqlite3")
    rows = collect_context(connection, positions, args.graph_depth)
    connection.close()
    evidence = format_evidence(rows)
    if not evidence:
        print("No grounded evidence was retrieved.")
        return

    if args.llm_model:
        print(generate_local_answer(args.llm_model.resolve(), args.question, evidence))
    else:
        print(evidence)


def inspect_bundle(args: argparse.Namespace) -> None:
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["bundle_size_mib"] = round(directory_size(bundle) / (1024 * 1024), 1)
    if args.llm_model:
        manifest["bundle_plus_llm_mib"] = round(
            (directory_size(bundle) + args.llm_model.stat().st_size) / (1024 * 1024),
            1,
        )
    print(json.dumps(manifest, indent=2))


def select_visual_subgraph(
    connection: sqlite3.Connection,
    entity_query: str | None,
    depth: int,
    max_nodes: int,
) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    connection.row_factory = sqlite3.Row
    if max_nodes < 1:
        raise ValueError("max_nodes must be at least 1")

    selected: set[str] = set()
    if entity_query:
        matches = connection.execute(
            """
            SELECT id FROM entities
            WHERE canonical_name LIKE ?
            ORDER BY name
            LIMIT ?
            """,
            (f"%{canonical_name(entity_query)}%", max_nodes),
        )
        selected.update(row["id"] for row in matches)
        if not selected:
            raise ValueError(f"No entity name contains: {entity_query}")

        frontier = set(selected)
        for _ in range(max(depth, 0)):
            marks = ",".join("?" for _ in frontier)
            rows = connection.execute(
                f"""
                SELECT source_id, target_id FROM relations
                WHERE source_id IN ({marks}) OR target_id IN ({marks})
                ORDER BY id
                """,
                tuple(frontier) * 2,
            )
            next_frontier: set[str] = set()
            for row in rows:
                for entity_id in (row["source_id"], row["target_id"]):
                    if len(selected) >= max_nodes:
                        break
                    if entity_id not in selected:
                        selected.add(entity_id)
                        next_frontier.add(entity_id)
            frontier = next_frontier
            if not frontier or len(selected) >= max_nodes:
                break
    else:
        degree_rows = connection.execute(
            """
            SELECT entity_id, COUNT(*) AS degree
            FROM (
                SELECT source_id AS entity_id FROM relations
                UNION ALL
                SELECT target_id AS entity_id FROM relations
            )
            GROUP BY entity_id
            ORDER BY degree DESC, entity_id
            LIMIT ?
            """,
            (max_nodes,),
        )
        selected.update(row["entity_id"] for row in degree_rows)

    if not selected:
        raise ValueError("The graph contains no connected entities")

    marks = ",".join("?" for _ in selected)
    entity_rows = list(
        connection.execute(
            f"""
            SELECT id, name, type FROM entities
            WHERE id IN ({marks})
            ORDER BY type, name
            """,
            tuple(selected),
        )
    )
    relation_rows = list(
        connection.execute(
            f"""
            SELECT r.id, r.source_id, r.target_id, r.predicate,
                   r.evidence_quote, s.name AS subject, t.name AS object,
                   c.source, c.page
            FROM relations r
            JOIN entities s ON s.id = r.source_id
            JOIN entities t ON t.id = r.target_id
            JOIN chunks c ON c.id = r.chunk_id
            WHERE r.source_id IN ({marks}) AND r.target_id IN ({marks})
            ORDER BY r.predicate, s.name, t.name
            """,
            tuple(selected) * 2,
        )
    )
    return entity_rows, relation_rows


def visualize_bundle(args: argparse.Namespace) -> None:
    try:
        import networkx as nx
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Install networkx and plotly to generate the graph visualization"
        ) from exc

    bundle = args.bundle.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Visualization already exists: {output}")

    connection = sqlite3.connect(bundle / "graph.sqlite3")
    entities, relations = select_visual_subgraph(
        connection,
        entity_query=args.entity,
        depth=args.depth,
        max_nodes=args.max_nodes,
    )
    connection.close()

    graph = nx.MultiDiGraph()
    for entity in entities:
        graph.add_node(entity["id"], name=entity["name"], type=entity["type"])
    for relation in relations:
        graph.add_edge(
            relation["source_id"],
            relation["target_id"],
            key=relation["id"],
            predicate=relation["predicate"],
        )

    positions = nx.spring_layout(graph, seed=42, iterations=100)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    midpoint_x: list[float] = []
    midpoint_y: list[float] = []
    edge_hover: list[str] = []
    for relation in relations:
        x0, y0 = positions[relation["source_id"]]
        x1, y1 = positions[relation["target_id"]]
        edge_x.extend((x0, x1, None))
        edge_y.extend((y0, y1, None))
        midpoint_x.append((x0 + x1) / 2)
        midpoint_y.append((y0 + y1) / 2)
        location = relation["source"]
        if relation["page"]:
            location += f", page {relation['page']}"
        edge_hover.append(
            "<b>"
            + html.escape(relation["subject"])
            + " —"
            + html.escape(relation["predicate"])
            + "→ "
            + html.escape(relation["object"])
            + "</b><br>Evidence: "
            + html.escape(relation["evidence_quote"])
            + "<br>Source: "
            + html.escape(location)
        )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 0.8, "color": "#94a3b8"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=midpoint_x,
            y=midpoint_y,
            mode="markers",
            marker={"size": 9, "color": "rgba(0,0,0,0)"},
            text=edge_hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    palette = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
        "#65a30d",
        "#db2777",
        "#0f766e",
        "#7c3aed",
        "#ca8a04",
        "#475569",
        "#64748b",
    ]
    entity_types = sorted({entity["type"] for entity in entities})
    colors = {
        entity_type: palette[index % len(palette)]
        for index, entity_type in enumerate(entity_types)
    }
    for entity_type in entity_types:
        type_entities = [entity for entity in entities if entity["type"] == entity_type]
        figure.add_trace(
            go.Scatter(
                x=[positions[entity["id"]][0] for entity in type_entities],
                y=[positions[entity["id"]][1] for entity in type_entities],
                mode="markers+text",
                text=[entity["name"] for entity in type_entities],
                textposition="top center",
                textfont={"size": 9},
                marker={
                    "size": 15,
                    "color": colors[entity_type],
                    "line": {"width": 1, "color": "white"},
                },
                customdata=[
                    [
                        entity["name"],
                        entity["type"],
                        graph.degree(entity["id"]),
                    ]
                    for entity in type_entities
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Type: %{customdata[1]}<br>"
                    "Connections: %{customdata[2]}<extra></extra>"
                ),
                name=entity_type,
            )
        )

    focus = f' near “{args.entity}”' if args.entity else " by highest connectivity"
    figure.update_layout(
        title=(
            f"IFS knowledge graph{focus} — "
            f"{len(entities)} entities, {len(relations)} relationships"
        ),
        template="plotly_white",
        hovermode="closest",
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        legend={"title": {"text": "Entity type"}},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    print(
        f"Wrote {output} with {len(entities)} entities and "
        f"{len(relations)} relationships"
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build the portable graph bundle")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    build.add_argument("--chunk-chars", type=int, default=3500)
    build.add_argument("--overlap-chars", type=int, default=300)
    build.set_defaults(func=build_bundle)

    query = subparsers.add_parser("query", help="Query without any network call")
    query.add_argument("--bundle", type=Path, required=True)
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--graph-depth", type=int, default=1)
    query.add_argument("--llm-model", type=Path, help="Optional local GGUF model")
    query.set_defaults(func=query_bundle)

    inspect = subparsers.add_parser("inspect", help="Show bundle size and metadata")
    inspect.add_argument("--bundle", type=Path, required=True)
    inspect.add_argument("--llm-model", type=Path)
    inspect.set_defaults(func=inspect_bundle)

    visualize = subparsers.add_parser(
        "visualize", help="Create an offline interactive HTML graph"
    )
    visualize.add_argument("--bundle", type=Path, required=True)
    visualize.add_argument("--output", type=Path, default=Path("ifs_graph.html"))
    visualize.add_argument(
        "--entity", help="Show entities whose names contain this text and their neighbors"
    )
    visualize.add_argument("--depth", type=int, default=2)
    visualize.add_argument("--max-nodes", type=int, default=150)
    visualize.set_defaults(func=visualize_bundle)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
