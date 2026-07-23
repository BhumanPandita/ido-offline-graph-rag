import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from offline_graph_rag import (
    Chunk,
    ChunkExtraction,
    ExtractedRelation,
    LOCAL_HASH_EMBED_MODEL,
    LocalHashEmbedder,
    add_extraction,
    build_visual_payload,
    collect_context,
    detect_visual_communities,
    generate_build_dashboard,
    initialize_database,
    relation_is_grounded,
    normalize_azure_endpoint,
    select_visual_subgraph,
    split_text,
    top_positions,
    make_parser,
)


class OfflineGraphRagTests(unittest.TestCase):
    def test_split_text_overlaps_without_losing_content(self):
        text = "First operational sentence. Second operational sentence. Third one."
        chunks = split_text(text, max_chars=40, overlap_chars=8)
        self.assertGreater(len(chunks), 1)
        self.assertIn("First operational sentence.", chunks[0])
        self.assertIn("Third one.", chunks[-1])

    def test_relation_requires_verbatim_evidence(self):
        relation = ExtractedRelation(
            subject="Task",
            subject_type="TASK",
            predicate="REQUIRES",
            object="Approval",
            object_type="STATUS",
            evidence_quote="The task requires approval.",
        )
        self.assertTrue(
            relation_is_grounded(relation, "The task requires approval. Continue.")
        )
        relation.evidence_quote = "This quote was invented."
        self.assertFalse(relation_is_grounded(relation, "The task requires approval."))

    def test_only_grounded_relation_is_stored(self):
        connection = sqlite3.connect(":memory:")
        initialize_database(connection)
        chunk = Chunk(id="chunk-1", source="guide.pdf", page=2, text="Task uses IFS.")
        connection.execute(
            "INSERT INTO chunks(id, source, page, text) VALUES (?, ?, ?, ?)",
            (chunk.id, chunk.source, chunk.page, chunk.text),
        )
        extraction = ChunkExtraction(
            relations=[
                ExtractedRelation(
                    subject="Task",
                    subject_type="TASK",
                    predicate="USES",
                    object="IFS",
                    object_type="SYSTEM",
                    evidence_quote="Task uses IFS.",
                )
            ]
        )
        accepted, discarded = add_extraction(connection, chunk, extraction)
        self.assertEqual((accepted, discarded), (1, 0))
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM relations").fetchone()[0], 1)

    def test_vector_search_returns_best_position(self):
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
        query = np.asarray([0.9, 0.1], dtype=np.float32)
        self.assertEqual(top_positions(embeddings, query, top_k=1), [0])

    def test_local_hash_embedder_needs_no_model_download(self):
        matrix = np.asarray(
            list(
                LocalHashEmbedder().embed(
                    ["Work order approval", "Aircraft maintenance"]
                )
            )
        )
        self.assertEqual(matrix.shape, (2, 384))
        self.assertEqual(LOCAL_HASH_EMBED_MODEL, "local-hash-v1")
        self.assertFalse(np.array_equal(matrix[0], matrix[1]))

    def test_build_accepts_a_manually_downloaded_embedding_folder(self):
        args = make_parser().parse_args(
            [
                "build",
                "--input",
                "data",
                "--output",
                "bundle",
                "--embedding-path",
                "models/bge-small-en-v1.5-onnx-q",
            ]
        )
        self.assertEqual(args.embedding_path, Path("models/bge-small-en-v1.5-onnx-q"))

    def test_azure_endpoint_normalization_matches_diagnostic_script(self):
        self.assertEqual(
            normalize_azure_endpoint(
                "https://example.openai.azure.com/openai/v1/"
            ),
            "https://example.openai.azure.com",
        )

    def test_retrieved_chunk_is_returned_without_a_graph_relation(self):
        connection = sqlite3.connect(":memory:")
        initialize_database(connection)
        connection.execute(
            "INSERT INTO chunks(id, source, page, text) VALUES (?, ?, ?, ?)",
            ("chunk-1", "guide.pdf", 4, "A directly relevant source sentence."),
        )
        connection.execute(
            """
            INSERT INTO retrieval_items(position, kind, ref_id, text)
            VALUES (?, ?, ?, ?)
            """,
            (0, "chunk", "chunk-1", "A directly relevant source sentence."),
        )
        rows = collect_context(connection, [0], graph_depth=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "chunk")

    def test_visual_subgraph_can_focus_on_an_entity(self):
        connection = sqlite3.connect(":memory:")
        initialize_database(connection)
        connection.execute(
            "INSERT INTO chunks(id, source, page, text) VALUES (?, ?, ?, ?)",
            ("chunk-1", "guide.pdf", 3, "Work Order uses IFS."),
        )
        extraction = ChunkExtraction(
            relations=[
                ExtractedRelation(
                    subject="Work Order",
                    subject_type="PROCESS",
                    predicate="USES",
                    object="IFS",
                    object_type="SYSTEM",
                    evidence_quote="Work Order uses IFS.",
                )
            ]
        )
        chunk = Chunk(
            id="chunk-1", source="guide.pdf", page=3, text="Work Order uses IFS."
        )
        add_extraction(connection, chunk, extraction)
        entities, relations = select_visual_subgraph(
            connection, entity_query="work order", depth=1, max_nodes=10
        )
        self.assertEqual({entity["name"] for entity in entities}, {"Work Order", "IFS"})
        self.assertEqual(len(relations), 1)

    def test_visual_communities_group_dense_clusters_and_keep_isolates(self):
        entities = [
            {"id": entity_id, "name": entity_id.upper(), "type": "OTHER"}
            for entity_id in ("a", "b", "c", "d", "e", "f", "isolated")
        ]
        pairs = [
            ("a", "b"),
            ("b", "c"),
            ("c", "a"),
            ("d", "e"),
            ("e", "f"),
            ("f", "d"),
            ("c", "d"),
        ]
        relations = [
            {"source_id": source_id, "target_id": target_id}
            for source_id, target_id in pairs
        ]
        communities, _ = detect_visual_communities(entities, relations)
        self.assertEqual(communities["a"], communities["b"])
        self.assertEqual(communities["d"], communities["e"])
        self.assertNotEqual(communities["a"], communities["d"])
        self.assertNotIn(
            communities["isolated"],
            {communities["a"], communities["d"]},
        )

    def test_visual_payload_groups_parallel_edges_without_losing_evidence(self):
        entities = [
            {"id": "a", "name": "Work Order", "type": "PROCESS"},
            {"id": "b", "name": "Approval", "type": "STATUS"},
        ]
        relations = [
            {
                "id": "r1",
                "source_id": "a",
                "target_id": "b",
                "subject": "Work Order",
                "predicate": "REQUIRES",
                "object": "Approval",
                "evidence_quote": "A work order requires approval.",
                "source": "guide.pdf",
                "page": 3,
                "chunk_id": "chunk-1",
            },
            {
                "id": "r2",
                "source_id": "b",
                "target_id": "a",
                "subject": "Approval",
                "predicate": "APPLIES_TO",
                "object": "Work Order",
                "evidence_quote": "Approval applies to the work order.",
                "source": "guide.pdf",
                "page": 4,
                "chunk_id": "chunk-2",
            },
        ]
        payload = build_visual_payload(entities, relations)
        self.assertEqual(len(payload["edge_groups"]), 1)
        self.assertEqual(set(payload["edge_groups"][0]["relation_ids"]), {"r1", "r2"})
        self.assertEqual(payload["relations"][0]["evidence"], relations[0]["evidence_quote"])
        self.assertEqual(payload["relations"][0]["chunk_id"], "chunk-1")

    def test_visualization_writes_self_contained_html(self):
        with tempfile.TemporaryDirectory() as temp_name:
            bundle = Path(temp_name) / "bundle"
            bundle.mkdir()
            connection = sqlite3.connect(bundle / "graph.sqlite3")
            initialize_database(connection)
            chunk = Chunk(
                id="chunk-1",
                source="guide.pdf",
                page=3,
                text="Work Order uses IFS.",
            )
            connection.execute(
                "INSERT INTO chunks(id, source, page, text) VALUES (?, ?, ?, ?)",
                (chunk.id, chunk.source, chunk.page, chunk.text),
            )
            add_extraction(
                connection,
                chunk,
                ChunkExtraction(
                    relations=[
                        ExtractedRelation(
                            subject="Work Order",
                            subject_type="PROCESS",
                            predicate="USES",
                            object="IFS",
                            object_type="SYSTEM",
                            evidence_quote="Work Order uses IFS.",
                        )
                    ]
                ),
            )
            connection.commit()
            connection.close()
            output = bundle / "graph_dashboard.html"
            generated = generate_build_dashboard(
                bundle,
                SimpleNamespace(
                    skip_dashboard=False,
                    dashboard_max_nodes=10,
                ),
                accepted_relations=1,
            )
            self.assertTrue(generated)
            html_text = output.read_text(encoding="utf-8")
            self.assertIn("plotly.js", html_text)
            self.assertIn("Work Order", html_text)
            self.assertIn("Communities", html_text)
            self.assertIn('id="relationship-table"', html_text)
            self.assertNotIn("<script src=", html_text)

    def test_zero_relation_build_skips_dashboard_without_failing(self):
        with tempfile.TemporaryDirectory() as temp_name:
            bundle = Path(temp_name) / "bundle"
            bundle.mkdir()
            generated = generate_build_dashboard(
                bundle,
                SimpleNamespace(
                    skip_dashboard=False,
                    dashboard_max_nodes=10,
                ),
                accepted_relations=0,
            )
            self.assertFalse(generated)
            self.assertFalse((bundle / "graph_dashboard.html").exists())


if __name__ == "__main__":
    unittest.main()
