import unittest
import json
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import guardrails
from guardrails import FileContextDecision, ChartType, ChartIntentDecision
import backend
import rag_engine


class TestFileContextIsolation(unittest.TestCase):
    """
    Phase 2: Comprehensive automated tests for Strict File Context Isolation.
    """

    def setUp(self):
        self.files_pool = [
            {"id": "file_2023_id", "original_name": "sales_2023.csv"},
            {"id": "file_2024_id", "original_name": "sales_2024.csv"},
            {"id": "file_summary_id", "original_name": "annual_summary.pdf"},
        ]

    def test_new_file_isolation_over_older_files(self):
        """
        When user uploads a new file (payload_file_ids provided), it must be used
        instead of older files already in conversation_files.
        """
        eval_res = guardrails.resolve_file_context(
            user_prompt="Create a bar chart of the sales",
            payload_file_ids=["file_2024_id"],
            conversation_files=self.files_pool,
            is_chart_request=True
        )
        self.assertEqual(eval_res["decision"], FileContextDecision.RESOLVED)
        self.assertEqual(eval_res["active_file_ids"], ["file_2024_id"])
        self.assertNotIn("file_2023_id", eval_res["active_file_ids"])
        self.assertNotIn("file_summary_id", eval_res["active_file_ids"])

    def test_two_files_conflicting_values_explicit_source_selection(self):
        """
        When multiple files exist with conflicting figures, explicit reference to one file
        scopes strictly to that file, completely excluding the other.
        """
        # User explicitly references sales_2023.csv
        eval_res_2023 = guardrails.resolve_file_context(
            user_prompt="Please generate a bar chart using sales_2023.csv",
            payload_file_ids=None,
            conversation_files=self.files_pool[:2],
            is_chart_request=True
        )
        self.assertEqual(eval_res_2023["decision"], FileContextDecision.RESOLVED)
        self.assertEqual(eval_res_2023["active_file_ids"], ["file_2023_id"])
        self.assertNotIn("file_2024_id", eval_res_2023["active_file_ids"])

        # User explicitly references sales_2024.csv
        eval_res_2024 = guardrails.resolve_file_context(
            user_prompt="Make a line chart using sales_2024",
            payload_file_ids=None,
            conversation_files=self.files_pool[:2],
            is_chart_request=True
        )
        self.assertEqual(eval_res_2024["decision"], FileContextDecision.RESOLVED)
        self.assertEqual(eval_res_2024["active_file_ids"], ["file_2024_id"])
        self.assertNotIn("file_2023_id", eval_res_2024["active_file_ids"])

    def test_explicit_request_to_use_older_file(self):
        """
        Respects ordinal or explicit requests to use older / first file.
        """
        eval_res = guardrails.resolve_file_context(
            user_prompt="Create a pie chart from the older file",
            payload_file_ids=None,
            conversation_files=self.files_pool[:2],
            is_chart_request=True
        )
        self.assertEqual(eval_res["decision"], FileContextDecision.RESOLVED)
        self.assertEqual(eval_res["active_file_ids"], ["file_2023_id"])

        eval_res_first = guardrails.resolve_file_context(
            user_prompt="Make a bar chart using the first file",
            payload_file_ids=None,
            conversation_files=self.files_pool[:2],
            is_chart_request=True
        )
        self.assertEqual(eval_res_first["decision"], FileContextDecision.RESOLVED)
        self.assertEqual(eval_res_first["active_file_ids"], ["file_2023_id"])

    def test_explicit_comparison_and_combination_of_multiple_files(self):
        """
        Detects explicit comparison or combination requests and includes both files.
        """
        # Explicit comparison
        eval_compare = guardrails.resolve_file_context(
            user_prompt="Compare sales_2023.csv and sales_2024.csv in a bar chart",
            payload_file_ids=None,
            conversation_files=self.files_pool[:2],
            is_chart_request=True
        )
        self.assertEqual(eval_compare["decision"], FileContextDecision.RESOLVED)
        self.assertIn("file_2023_id", eval_compare["active_file_ids"])
        self.assertIn("file_2024_id", eval_compare["active_file_ids"])

        # Explicit combination
        eval_combine = guardrails.resolve_file_context(
            user_prompt="Combine data from both files into a single bar chart",
            payload_file_ids=None,
            conversation_files=self.files_pool[:2],
            is_chart_request=True
        )
        self.assertEqual(eval_combine["decision"], FileContextDecision.RESOLVED)
        self.assertEqual(len(eval_combine["active_file_ids"]), 2)

    def test_ambiguous_file_reference_triggers_clarification(self):
        """
        When multiple files are attached and user requests a chart without specifying which file,
        clarification is triggered asking the user to specify or compare.
        """
        eval_ambiguous = guardrails.resolve_file_context(
            user_prompt="Create a bar chart of the quarterly revenue",
            payload_file_ids=None,
            conversation_files=self.files_pool[:2],
            is_chart_request=True
        )
        self.assertEqual(eval_ambiguous["decision"], FileContextDecision.AMBIGUOUS)
        self.assertIsNotNone(eval_ambiguous["response"])
        self.assertIn("sales_2023.csv", eval_ambiguous["response"])
        self.assertIn("sales_2024.csv", eval_ambiguous["response"])
        self.assertIn("compare or combine", eval_ambiguous["response"])

    def test_rag_retrieval_respects_file_provenance(self):
        """
        Verifies RAG retrieve_file_context strictly limits chunks to the resolved file IDs
        and format_context_block includes File ID and filename provenance.
        """
        mock_chunks = [
            {"id": "c1", "file_id": "file_2023_id", "original_name": "sales_2023.csv", "chunk_index": 0, "content": "2023 Rev: $100"},
            {"id": "c2", "file_id": "file_2024_id", "original_name": "sales_2024.csv", "chunk_index": 0, "content": "2024 Rev: $500"},
        ]

        with patch("db.get_chunks_by_file_ids") as mock_db_chunks:
            # Simulate DB returning chunks strictly matching the requested file_ids
            mock_db_chunks.side_effect = lambda fids: [c for c in mock_chunks if c["file_id"] in fids]

            results_2023 = rag_engine.retrieve_file_context(["file_2023_id"], "sales")
            self.assertEqual(len(results_2023), 1)
            self.assertEqual(results_2023[0]["chunk"]["file_id"], "file_2023_id")
            self.assertIn("2023 Rev", results_2023[0]["chunk"]["content"])

            # Verify format_context_block output contains file provenance
            formatted = rag_engine.format_context_block(results_2023)
            self.assertIn("File ID: file_2023_id", formatted)
            self.assertIn("sales_2023.csv", formatted)

    def test_consecutive_chart_requests_history_sanitization(self):
        """
        Verifies that stale chart blocks in conversation history are sanitized
        to prevent earlier chart numbers leaking into a new chart, unless explicitly requested.
        """
        history_with_chart = [
            {"role": "user", "content": "Create a pie chart for 2023"},
            {"role": "assistant", "content": '```chart\n{"type": "pie", "data": [{"name": "OldProd", "value": 999}]}\n```\nHere is 2023 data.'}
        ]

        # Scenario 1: New chart request without explicit reuse -> sanitize
        sanitized = guardrails.sanitize_conversation_history_for_charts(
            history=history_with_chart,
            user_prompt="Now create a bar chart for 2024",
            is_chart_request=True
        )
        self.assertNotIn('"OldProd"', sanitized[1]["content"])
        self.assertNotIn('```chart', sanitized[1]["content"])
        self.assertIn("Earlier chart visualization omitted", sanitized[1]["content"])

        # Scenario 2: Explicit request to reuse/update earlier chart -> preserve
        preserved = guardrails.sanitize_conversation_history_for_charts(
            history=history_with_chart,
            user_prompt="Update the previous chart with new percentages",
            is_chart_request=True
        )
        self.assertIn('"OldProd"', preserved[1]["content"])
        self.assertIn('```chart', preserved[1]["content"])

    def test_missing_or_insufficient_data_fastpath_stream(self):
        """
        When active file is selected for chart generation but RAG returns no content,
        fast-path explains the limitation rather than hallucinating dummy numbers.
        """
        from run import app
        client = app.test_client()

        with patch("db.get_conversation_files") as mock_conv_files, \
             patch("rag_engine.retrieve_file_context") as mock_rag:
            mock_conv_files.return_value = [{"id": "empty_file_id", "original_name": "empty_data.csv"}]
            mock_rag.return_value = []  # No readable chunks

            payload = {
                "conversation_id": "conv_empty_file_test",
                "model": "mistral:latest",
                "jailbreak": "default",
                "meta": {
                    "content": {
                        "conversation": [],
                        "internet_access": False,
                        "parts": [{"role": "user", "content": "Create a bar chart from empty_data.csv"}]
                    }
                }
            }
            res = client.post('/backend-api/v2/conversation', json=payload)
            self.assertEqual(res.status_code, 200)
            output = res.get_data(as_text=True)

            self.assertIn("does not contain readable data or content to generate a chart", output)
            self.assertIn("empty_data.csv", output)
            self.assertIn("data: [DONE]", output)

    def test_ambiguous_file_clarification_sse_stream(self):
        """
        In Standard Mode, an ambiguous file reference during a chart request triggers
        immediate SSE clarification without calling external LLM.
        """
        from run import app
        client = app.test_client()

        with patch("db.get_conversation_files") as mock_conv_files:
            mock_conv_files.return_value = self.files_pool[:2]

            payload = {
                "conversation_id": "conv_ambiguous_files_test",
                "model": "mistral:latest",
                "jailbreak": "default",
                "meta": {
                    "content": {
                        "conversation": [],
                        "internet_access": False,
                        "parts": [{"role": "user", "content": "Create a bar chart of the sales"}]
                    }
                }
            }
            res = client.post('/backend-api/v2/conversation', json=payload)
            self.assertEqual(res.status_code, 200)
            output = res.get_data(as_text=True)

            self.assertIn("You have multiple files available", output)
            self.assertIn("sales_2023.csv", output)
            self.assertIn("sales_2024.csv", output)
            self.assertIn("data: [DONE]", output)

    def test_dan_mode_preserves_legacy_file_behavior(self):
        """
        DAN Mode ('gpt-dan-11.0') bypasses Standard Mode file context isolation and clarification.
        """
        req_mock = MagicMock()
        req_mock.json = {
            "meta": {
                "content": {
                    "conversation": [],
                    "internet_access": False,
                    "parts": [{"content": "Create a visual plot"}]
                }
            },
            "conversation_id": "conv_dan_files_test"
        }

        with patch("backend.request", req_mock), \
             patch("db.get_conversation_files") as mock_conv_files:
            mock_conv_files.return_value = self.files_pool[:2]

            messages = backend.build_messages(jailbreak="gpt-dan-11.0")
            # Should have DAN instructions and not trigger Standard Mode clarification
            has_dan = any("DAN" in m.get("content", "") for m in messages)
            self.assertTrue(has_dan)
            has_clarify = any("You have multiple files available" in m.get("content", "") for m in messages)
            self.assertFalse(has_clarify)


if __name__ == '__main__':
    unittest.main()
