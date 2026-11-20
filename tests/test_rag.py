"""
test_rag.py - Unit and Integration Tests for Project 9
Tests PDF ingestion, chunking strategy, ChromaDB persistence,
and strict grounded question answering for assessment questions Q1-Q5.
"""

import os
import unittest
from ingest import extract_pages_from_pdf, chunk_pages, get_embedding_function, DEFAULT_PDF_PATH
from rag_engine import answer, get_collection


class TestArchivalRAGProject(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure collection is loaded
        cls.collection = get_collection()

    def test_pdf_ingestion_page_count_and_indexing(self):
        """Technical Req 1: Ingest page-by-page preserving 1-indexed page numbers."""
        self.assertTrue(os.path.exists(DEFAULT_PDF_PATH), f"PDF file not found at {DEFAULT_PDF_PATH}")
        pages = extract_pages_from_pdf(DEFAULT_PDF_PATH)
        self.assertEqual(len(pages), 50, "PDF must contain exactly 50 pages")

        # Verify 1-indexed page numbers
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[-1]["page_number"], 50)
        for i, page in enumerate(pages):
            self.assertEqual(page["page_number"], i + 1)
            self.assertGreater(len(page["text"]), 100, f"Page {i+1} text should not be empty")

    def test_text_chunking_deterministic_ids_and_metadata(self):
        """Technical Req 2: Chunking with RecursiveCharacterTextSplitter and deterministic IDs."""
        pages = extract_pages_from_pdf(DEFAULT_PDF_PATH)
        chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=200)

        self.assertGreater(len(chunks), 50, "Chunks should be generated for all pages")
        for chunk in chunks:
            cid = chunk["id"]
            # Verify deterministic format: page_<N>_chunk_<M>
            parts = cid.split("_")
            self.assertEqual(len(parts), 4, f"Chunk ID '{cid}' does not match page_<N>_chunk_<M>")
            self.assertEqual(parts[0], "page")
            self.assertEqual(parts[2], "chunk")
            self.assertTrue(parts[1].isdigit(), f"Page number in '{cid}' must be integer")
            self.assertTrue(parts[3].isdigit(), f"Chunk index in '{cid}' must be integer")

            # Verify metadata
            meta = chunk["metadata"]
            self.assertIn("page_number", meta)
            self.assertIn("chunk_id", meta)
            self.assertIn("source", meta)
            self.assertEqual(meta["chunk_id"], cid)
            self.assertEqual(meta["page_number"], int(parts[1]))

    def test_chroma_persistence_and_reloading(self):
        """Technical Req 3 & 4: ChromaDB PersistentClient at ./chroma_db with local embeddings."""
        self.assertTrue(os.path.isdir("./chroma_db"), "./chroma_db directory must exist on disk")
        count = self.collection.count()
        self.assertEqual(count, 100, "Collection should persist 100 indexed chunks")

        # Test local embedding function output dimension (all-MiniLM-L6-v2 produces 384-dim vectors)
        ef = get_embedding_function()
        vecs = ef(["Test local embedding function"])
        self.assertEqual(len(vecs[0]), 384, "Chroma default embedding function should yield 384 dimensions")

    def test_required_function_signature(self):
        """Technical Req 5: Exact function signature answer(question: str) -> dict."""
        res = answer("What is the ruling gradient ratio negotiated at Battery Hill?")
        self.assertIsInstance(res, dict, "answer() must return a dictionary")
        self.assertIn("answer", res, "Result dict must have 'answer' key")
        self.assertIn("sources", res, "Result dict must have 'sources' key")
        self.assertIsInstance(res["answer"], str, "'answer' must be a string")
        self.assertIsInstance(res["sources"], list, "'sources' must be a list")

    def test_q1_guaranteed_interest_rate_1849(self):
        """Assessment Q1: 1849 guaranteed interest rate."""
        q1 = "What exact annual guaranteed interest rate, expressed as a percentage, did the colonial administration promise private British railway companies under the 1849 contract terms?"
        res1 = answer(q1)
        self.assertNotEqual(res1["answer"], "[not found]")
        self.assertIn("5%", res1["answer"])
        self.assertIn("page_13_chunk_1", res1["sources"])

    def test_q2_bhor_ghat_reversing_station_and_gradient(self):
        """Assessment Q2: Bhor Ghat engineering obstacle and gradient ratio."""
        q2 = "Which engineering obstacle on the Bhor Ghat incline made the reversing station described in Chapter 3 necessary, and what was the gradient ratio?"
        res2 = answer(q2)
        self.assertNotEqual(res2["answer"], "[not found]")
        self.assertIn("Battery Hill", res2["answer"])
        self.assertIn("1 in 37", res2["answer"])
        self.assertIn("page_26_chunk_1", res2["sources"])

    def test_q3_vande_bharat_ticket_price_unsupported(self):
        """Assessment Q3: Vande Bharat ticket price (not in 19th century corpus -> [not found])."""
        q3 = "What is the ticket price for an Executive Chair Car seat on the Vande Bharat Express between New Delhi and Varanasi?"
        res3 = answer(q3)
        self.assertEqual(res3["answer"], "[not found]")
        self.assertEqual(res3["sources"], [])

    def test_q4_diamond_harbour_telegraph_supervisor(self):
        """Assessment Q4: 1851 Diamond Harbour telegraph supervisor."""
        q4 = "According to the archival records in the text, who supervised the laying of the experimental Diamond Harbour underwater telegraph cable in 1851?"
        res4 = answer(q4)
        self.assertNotEqual(res4["answer"], "[not found]")
        self.assertTrue("O’Shaughnessy" in res4["answer"] or "O'Shaughnessy" in res4["answer"])
        self.assertIn("page_43_chunk_1", res4["sources"])

    def test_q5_fifa_world_cup_unsupported(self):
        """Assessment Q5: 2022 FIFA World Cup (out of domain -> [not found])."""
        q5 = "Which country won the 2022 FIFA World Cup?"
        res5 = answer(q5)
        self.assertEqual(res5["answer"], "[not found]")
        self.assertEqual(res5["sources"], [])


if __name__ == "__main__":
    unittest.main()
