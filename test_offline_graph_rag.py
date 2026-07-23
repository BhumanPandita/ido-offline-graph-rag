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
    add_extraction,
    collect_context,
    initialize_database,
    relation_is_grounded,
    normalize_azure_endpoint,
    select_visual_subgraph,
    split_text,
    top_positions,
    visualize_bundle,
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
            output = Path(temp_name) / "graph.html"
            visualize_bundle(
                SimpleNamespace(
                    bundle=bundle,
                    output=output,
                    entity=None,
                    depth=2,
                    max_nodes=10,
                )
            )
            html_text = output.read_text(encoding="utf-8")
            self.assertIn("plotly.js", html_text)
            self.assertIn("Work Order", html_text)


if __name__ == "__main__":
    unittest.main()
