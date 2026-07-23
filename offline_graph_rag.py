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


def get_embedder(
    model_name: str,
    cache_dir: Path,
    offline: bool,
    model_path: Path | None = None,
):
    if model_name == LOCAL_HASH_EMBED_MODEL:
        if model_path:
            raise ValueError("local-hash-v1 does not use an embedding model folder")
        return LocalHashEmbedder()
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("Install fastembed to create or query embeddings") from exc
    options = {
        "model_name": model_name,
        "cache_dir": str(cache_dir),
        "local_files_only": offline,
    }
    if model_path:
        options["specific_model_path"] = str(model_path)
    return TextEmbedding(**options)


def embed_texts(embedder, texts: list[str]) -> np.ndarray:
    if not texts:
        raise ValueError("Cannot embed an empty graph")
    matrix = np.asarray(list(embedder.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def validate_dashboard_configuration(args: argparse.Namespace) -> None:
    if getattr(args, "skip_dashboard", False):
        return
    if getattr(args, "dashboard_max_nodes", 500) < 1:
        raise ValueError("dashboard-max-nodes must be at least 1")
    template_path = Path(__file__).with_name("graph_dashboard_template.html")
    if not template_path.is_file():
        raise FileNotFoundError(f"Dashboard template not found: {template_path}")
    try:
        import networkx  # noqa: F401
        from plotly.offline import get_plotlyjs  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Install networkx and plotly to generate the graph dashboard"
        ) from exc


def generate_build_dashboard(
    bundle_dir: Path,
    args: argparse.Namespace,
    accepted_relations: int,
) -> bool:
    if getattr(args, "skip_dashboard", False):
        return False
    if not accepted_relations:
        print(
            "Dashboard skipped because the build produced no grounded relationships.",
            flush=True,
        )
        return False
    visualize_bundle(
        argparse.Namespace(
            bundle=bundle_dir,
            output=bundle_dir / "graph_dashboard.html",
            entity=None,
            depth=2,
            max_nodes=getattr(args, "dashboard_max_nodes", 500),
        )
    )
    return True


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
    validate_dashboard_configuration(args)
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
        manual_model_path = (
            args.embedding_path.resolve() if args.embedding_path else None
        )
        if manual_model_path:
            if not manual_model_path.is_dir():
                raise FileNotFoundError(
                    f"Embedding model folder not found: {manual_model_path}"
                )
            shutil.copytree(manual_model_path, model_cache)
        # Download the local embedding model before any paid Azure extraction.
        # A corporate-network/Hugging Face failure should fail fast here.
        print("Preparing the local embedding model...", flush=True)
        embedder = get_embedder(
            args.embedding_model,
            model_cache,
            offline=bool(manual_model_path),
            model_path=model_cache if manual_model_path else None,
        )

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
            "dashboard_generated": bool(
                accepted and not getattr(args, "skip_dashboard", False)
            ),
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        generate_build_dashboard(temp_dir, args, accepted)
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
        model_path=(bundle / "models" / "embeddings")
        if (bundle / "models" / "embeddings").is_dir()
        else None,
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
            SELECT r.id, r.source_id, r.target_id, r.predicate, r.chunk_id,
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


def detect_visual_communities(
    entities: list[sqlite3.Row], relations: list[sqlite3.Row]
) -> tuple[dict[str, int], object]:
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("Install networkx to detect graph communities") from exc

    graph = nx.Graph()
    graph.add_nodes_from(entity["id"] for entity in entities)
    for relation in relations:
        source_id = relation["source_id"]
        target_id = relation["target_id"]
        if graph.has_edge(source_id, target_id):
            graph[source_id][target_id]["weight"] += 1
        else:
            graph.add_edge(source_id, target_id, weight=1)

    if graph.number_of_edges():
        groups = nx.community.louvain_communities(graph, weight="weight", seed=42)
    else:
        groups = [{entity_id} for entity_id in graph.nodes]
    groups = sorted(groups, key=lambda group: (-len(group), min(group)))

    community_by_entity: dict[str, int] = {}
    for community_id, group in enumerate(groups, start=1):
        for entity_id in group:
            community_by_entity[entity_id] = community_id
    return community_by_entity, graph


def build_visual_payload(
    entities: list[sqlite3.Row], relations: list[sqlite3.Row]
) -> dict:
    community_by_entity, graph = detect_visual_communities(entities, relations)
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("Install networkx to lay out the graph") from exc

    if graph.number_of_nodes() == 1:
        only_id = next(iter(graph.nodes))
        positions = {only_id: np.asarray([0.0, 0.0])}
    else:
        positions = nx.spring_layout(
            graph,
            seed=42,
            iterations=200,
            weight="weight",
            k=max(0.12, 1.8 / np.sqrt(graph.number_of_nodes())),
        )

    relation_count = {entity["id"]: 0 for entity in entities}
    incoming_count = {entity["id"]: 0 for entity in entities}
    outgoing_count = {entity["id"]: 0 for entity in entities}
    for relation in relations:
        relation_count[relation["source_id"]] += 1
        relation_count[relation["target_id"]] += 1
        outgoing_count[relation["source_id"]] += 1
        incoming_count[relation["target_id"]] += 1

    nodes = []
    entity_name_by_id = {}
    entity_type_by_id = {}
    for entity in entities:
        entity_id = entity["id"]
        entity_name_by_id[entity_id] = entity["name"]
        entity_type_by_id[entity_id] = entity["type"]
        nodes.append(
            {
                "id": entity_id,
                "name": entity["name"],
                "type": entity["type"],
                "community": community_by_entity[entity_id],
                "degree": relation_count[entity_id],
                "incoming": incoming_count[entity_id],
                "outgoing": outgoing_count[entity_id],
                "x": round(float(positions[entity_id][0]), 6),
                "y": round(float(positions[entity_id][1]), 6),
            }
        )

    relation_items = []
    edge_groups: dict[tuple[str, str], dict] = {}
    for relation in relations:
        source_id = relation["source_id"]
        target_id = relation["target_id"]
        relation_items.append(
            {
                "id": relation["id"],
                "source_id": source_id,
                "target_id": target_id,
                "subject": relation["subject"],
                "predicate": relation["predicate"],
                "object": relation["object"],
                "evidence": relation["evidence_quote"],
                "source": relation["source"],
                "page": relation["page"],
                "chunk_id": relation["chunk_id"],
                "cross_community": (
                    community_by_entity[source_id] != community_by_entity[target_id]
                ),
            }
        )
        pair = tuple(sorted((source_id, target_id)))
        if pair not in edge_groups:
            edge_groups[pair] = {
                "id": stable_id("visual-edge", *pair),
                "source_id": pair[0],
                "target_id": pair[1],
                "relation_ids": [],
            }
        edge_groups[pair]["relation_ids"].append(relation["id"])

    communities = []
    for community_id in sorted(set(community_by_entity.values())):
        member_ids = {
            entity_id
            for entity_id, assigned_id in community_by_entity.items()
            if assigned_id == community_id
        }
        incident_relations = [
            relation
            for relation in relation_items
            if relation["source_id"] in member_ids
            or relation["target_id"] in member_ids
        ]
        top_ids = sorted(
            member_ids,
            key=lambda entity_id: (
                -relation_count[entity_id],
                entity_name_by_id[entity_id].casefold(),
            ),
        )[:3]
        communities.append(
            {
                "id": community_id,
                "label": " · ".join(entity_name_by_id[entity_id] for entity_id in top_ids),
                "entity_count": len(member_ids),
                "relationship_count": len(incident_relations),
                "top_entities": [
                    entity_name_by_id[entity_id] for entity_id in top_ids
                ],
                "types": sorted(
                    {entity_type_by_id[entity_id] for entity_id in member_ids}
                ),
                "sources": sorted(
                    {relation["source"] for relation in incident_relations}
                ),
            }
        )

    return {
        "nodes": nodes,
        "relations": relation_items,
        "edge_groups": list(edge_groups.values()),
        "communities": communities,
        "filters": {
            "entity_types": sorted({node["type"] for node in nodes}),
            "predicates": sorted(
                {relation["predicate"] for relation in relation_items}
            ),
            "sources": sorted({relation["source"] for relation in relation_items}),
        },
    }


def render_visual_dashboard(payload: dict, title: str) -> str:
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise RuntimeError("Install plotly to generate the graph dashboard") from exc

    template_path = Path(__file__).with_name("graph_dashboard_template.html")
    if not template_path.is_file():
        raise FileNotFoundError(f"Dashboard template not found: {template_path}")
    template = template_path.read_text(encoding="utf-8")
    graph_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return (
        template.replace("__DASHBOARD_TITLE__", html.escape(title))
        .replace("__PLOTLY_JAVASCRIPT__", get_plotlyjs())
        .replace("__GRAPH_DATA__", graph_json)
    )


def visualize_bundle(args: argparse.Namespace) -> None:
    bundle = args.bundle.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Visualization already exists: {output}")

    connection = sqlite3.connect(bundle / "graph.sqlite3")
    try:
        entities, relations = select_visual_subgraph(
            connection,
            entity_query=args.entity,
            depth=args.depth,
            max_nodes=args.max_nodes,
        )
    finally:
        connection.close()

    focus = f' near “{args.entity}”' if args.entity else " by highest connectivity"
    payload = build_visual_payload(entities, relations)
    payload["meta"] = {
        "title": "IFS Knowledge Graph",
        "scope": f"Communities detected in this view{focus}",
    }
    document = render_visual_dashboard(payload, "IFS Knowledge Graph Dashboard")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(
        f"Wrote interactive dashboard {output} with {len(entities)} entities, "
        f"{len(relations)} relationships, and {len(payload['communities'])} communities"
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build the portable graph bundle")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    build.add_argument(
        "--embedding-path",
        type=Path,
        help="Local FastEmbed-compatible ONNX model folder; avoids any model download",
    )
    build.add_argument("--chunk-chars", type=int, default=3500)
    build.add_argument("--overlap-chars", type=int, default=300)
    build.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Do not generate graph_dashboard.html after building",
    )
    build.add_argument("--dashboard-max-nodes", type=int, default=500)
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
        "visualize", help="Create an offline interactive graph dashboard"
    )
    visualize.add_argument("--bundle", type=Path, required=True)
    visualize.add_argument(
        "--output", type=Path, default=Path("ifs_graph_dashboard.html")
    )
    visualize.add_argument(
        "--entity", help="Show entities whose names contain this text and their neighbors"
    )
    visualize.add_argument("--depth", type=int, default=2)
    visualize.add_argument("--max-nodes", type=int, default=500)
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
