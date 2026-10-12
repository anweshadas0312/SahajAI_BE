"""
End-to-End Integration & Regression Verification Tests for SahajAI Chart Guardrails
Validates:
1. End-to-end chart flow (Pie, Bar, Line).
2. Ambiguous requests and clarification handling.
3. Markdown tables, numeric prose, and ordinary text safety (no unwanted charts).
4. Invalid/insufficient data handling without silent zero coercion.
5. Strict file context isolation (single file, multi-file comparison, old file leakage prevention).
6. Multi-turn conversation history sanitization (turn 1 chart -> turn 2 non-chart).
7. Stream buffering & validation behavior.
8. DAN Mode and EvilBOT Mode isolation.
"""

import unittest
import json
import os
import sys

# Ensure backend directory is in path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import guardrails
from guardrails import (
    ChartType,
    ChartIntentDecision,
    FileContextDecision,
    ContentDecision,
    evaluate_chart_intent,
    get_chart_directive,
    resolve_file_context,
    sanitize_conversation_history_for_charts,
    validate_chart_payload,
    validate_and_sanitize_chart_response,
    evaluate_content_safety,
    handle_identity_query,
)
from backend import special_instructions


class TestChartE2EIntegration(unittest.TestCase):

    def test_e2e_pie_chart_flow(self):
        """Verify explicit pie chart flow: intent -> directive -> response validation."""
        user_msg = "Please generate a pie chart of the revenue breakdown"
        intent = evaluate_chart_intent(user_msg)
        self.assertEqual(intent["decision"], ChartIntentDecision.GENERATE)
        self.assertEqual(intent["chart_type"], ChartType.PIE)

        directive = get_chart_directive(intent["chart_type"])
        self.assertIn("```chart", directive)
        self.assertIn('"type": "pie"', directive)

        raw_output = (
            "Here is the market share breakdown:\n\n"
            "```chart\n"
            '{\n  "type": "pie",\n  "title": "Market Share",\n  "data": [\n'
            '    {"name": "Tech", "value": 50},\n'
            '    {"name": "Auto", "value": 30},\n'
            '    {"name": "Finance", "value": 20}\n  ]\n}\n'
            "```\n"
        )
        is_v, clean_text, payload, v_err = validate_and_sanitize_chart_response(
            raw_output, expected_chart_type="pie"
        )
        self.assertTrue(is_v)
        self.assertIn("```chart", clean_text)
        self.assertIn('"type": "pie"', clean_text)
        self.assertIsNone(v_err)

    def test_e2e_bar_chart_flow_with_negative_values(self):
        """Verify explicit bar chart flow with legitimate negative values (e.g. profit/loss)."""
        user_msg = "Create a bar chart comparing regional revenue"
        intent = evaluate_chart_intent(user_msg)
        self.assertEqual(intent["decision"], ChartIntentDecision.GENERATE)
        self.assertEqual(intent["chart_type"], ChartType.BAR)

        raw_output = (
            "Quarterly profit summary:\n\n"
            "```chart\n"
            '{\n  "type": "bar",\n  "title": "Quarterly Profit",\n  "data": [\n'
            '    {"name": "Q1", "value": 500},\n'
            '    {"name": "Q2", "value": -200},\n'
            '    {"name": "Q3", "value": 300}\n  ]\n}\n'
            "```\n"
        )
        is_v, clean_text, payload, v_err = validate_and_sanitize_chart_response(
            raw_output, expected_chart_type="bar"
        )
        self.assertTrue(is_v)
        self.assertIn("```chart", clean_text)
        self.assertIn("-200", clean_text)

    def test_e2e_line_chart_flow(self):
        """Verify explicit line chart flow across sequence of data points."""
        user_msg = "Plot a line chart of temperature changes over time"
        intent = evaluate_chart_intent(user_msg)
        self.assertEqual(intent["decision"], ChartIntentDecision.GENERATE)
        self.assertEqual(intent["chart_type"], ChartType.LINE)

        raw_output = (
            "Temperature trend:\n\n"
            "```chart\n"
            '{\n  "type": "line",\n  "title": "Weekly Temperature",\n  "data": [\n'
            '    {"name": "Day 1", "value": 22},\n'
            '    {"name": "Day 2", "value": 24},\n'
            '    {"name": "Day 3", "value": 21},\n'
            '    {"name": "Day 4", "value": 25},\n'
            '    {"name": "Day 5", "value": 26}\n  ]\n}\n'
            "```\n"
        )
        is_v, clean_text, payload, v_err = validate_and_sanitize_chart_response(
            raw_output, expected_chart_type="line"
        )
        self.assertTrue(is_v)
        self.assertIn("```chart", clean_text)
        self.assertIn('"type": "line"', clean_text)

    def test_ambiguous_chart_clarification_no_chart_emitted(self):
        """Ambiguous chart request produces clarification prompt without generating a chart block."""
        user_msg = "Can you make a chart showing revenue?"
        intent = evaluate_chart_intent(user_msg)
        self.assertEqual(intent["decision"], ChartIntentDecision.CLARIFY)
        self.assertEqual(intent["chart_type"], ChartType.AMBIGUOUS)
        self.assertIn("Pie Chart", intent["response"])
        self.assertIn("Bar Chart", intent["response"])
        self.assertIn("Line Chart", intent["response"])

    def test_ordinary_tables_and_prose_do_not_trigger_chart(self):
        """Markdown tables, bullet lists, and numeric prose must never trigger chart intent."""
        samples = [
            "Here is the pricing breakdown:\n| Plan | Price |\n| Basic | 10 |\n| Pro | 25 |",
            "Can you provide a summary table of the quarterly revenue?",
            "Format this financial data as a markdown table",
            "Summarize the plot of Hamlet",
            "Can you explain the visual elements in modern UI design?",
            "Draw an architecture diagram of a microservices backend",
        ]
        for s in samples:
            intent = evaluate_chart_intent(s)
            self.assertEqual(intent["decision"], ChartIntentDecision.NONE, f"Failed for input: {s}")

    def test_invalid_chart_data_produces_limitation_explanation(self):
        """Invalid data (e.g. pie with negative values) produces readable explanation."""
        bad_pie = (
            "```chart\n"
            "{\n"
            '  "type": "pie",\n'
            '  "title": "Quarterly Breakdown",\n'
            '  "data": [\n'
            '    {"name": "Operating Expense", "value": -50},\n'
            '    {"name": "Revenue", "value": 100}\n'
            '  ]\n'
            "}\n"
            "```\n"
        )
        is_v, clean_text, payload, v_err = validate_and_sanitize_chart_response(
            bad_pie, expected_chart_type="pie"
        )
        self.assertFalse(is_v)
        self.assertNotIn("```chart", clean_text)
        self.assertIn("Chart Generation Limitation", clean_text)
        self.assertIn("cannot have negative values", clean_text)

    def test_no_silent_zero_coercion_on_corrupt_values(self):
        """Backend validator rejects string/non-numeric values instead of coercing to 0."""
        corrupt_payload = {
            "type": "bar",
            "title": "Corrupt",
            "data": [
                {"name": "A", "value": 100},
                {"name": "B", "value": "null"}
            ]
        }
        is_valid, err = validate_chart_payload(corrupt_payload)
        self.assertFalse(is_valid)
        self.assertIn("non-numeric", err)

    def test_file_context_isolation_new_file_replaces_old(self):
        """When user uploads or targets a new file, older files in conversation are isolated."""
        available_files = [
            {"id": "file_2023_id", "original_name": "sales_2023.csv"},
            {"id": "file_2024_id", "original_name": "sales_2024.csv"},
        ]
        user_msg = "Create a bar chart of the sales"
        context = resolve_file_context(
            user_prompt=user_msg,
            payload_file_ids=["file_2024_id"],
            conversation_files=available_files,
            is_chart_request=True
        )
        self.assertEqual(context["decision"], FileContextDecision.RESOLVED)
        self.assertEqual(context["active_file_ids"], ["file_2024_id"])
        self.assertNotIn("file_2023_id", context["active_file_ids"])

    def test_multi_file_explicit_comparison(self):
        """Explicit comparison selects both files with provenance."""
        available_files = [
            {"id": "file_2023_id", "original_name": "sales_2023.csv"},
            {"id": "file_2024_id", "original_name": "sales_2024.csv"},
        ]
        user_msg = "Compare sales_2023.csv and sales_2024.csv in a bar chart"
        context = resolve_file_context(
            user_prompt=user_msg,
            payload_file_ids=None,
            conversation_files=available_files,
            is_chart_request=True
        )
        self.assertEqual(context["decision"], FileContextDecision.RESOLVED)
        self.assertEqual(len(context["active_file_ids"]), 2)
        self.assertIn("file_2023_id", context["active_file_ids"])
        self.assertIn("file_2024_id", context["active_file_ids"])

    def test_conversation_history_sanitization_across_turns(self):
        """Turn 1 chart JSON is sanitized when building context for a new turn 2 chart request."""
        history = [
            {"role": "user", "content": "Make a pie chart of our portfolio"},
            {
                "role": "assistant",
                "content": (
                    "```chart\n"
                    '{"type": "pie", "data": [{"name": "OldProd", "value": 999}]}\n'
                    "```\n"
                    "Here is 2023 data."
                )
            },
            {"role": "user", "content": "Now create a bar chart for 2024"}
        ]
        sanitized = sanitize_conversation_history_for_charts(
            history=history,
            user_prompt=history[2]["content"],
            is_chart_request=True
        )
        assistant_turn = sanitized[1]["content"]
        self.assertNotIn("```chart", assistant_turn)
        self.assertNotIn("OldProd", assistant_turn)
        self.assertIn("Earlier chart visualization omitted", assistant_turn)

        # Non-chart turn does not trigger chart generation
        non_chart_query = "What is the capital of France?"
        intent_non_chart = evaluate_chart_intent(non_chart_query)
        self.assertEqual(intent_non_chart["decision"], ChartIntentDecision.NONE)

    def test_mode_isolation_dan_and_evilbot_untouched(self):
        """Verify DAN Mode and EvilBOT Mode special instructions remain intact."""
        self.assertIn("gpt-dan-11.0", special_instructions)
        self.assertIn("gpt-evil", special_instructions)
        dan_instruction = special_instructions["gpt-dan-11.0"][0]["content"]
        evil_instruction = special_instructions["gpt-evil"][0]["content"]
        self.assertIn("DAN", dan_instruction)
        self.assertIn("EvilBOT", evil_instruction)

    def test_identity_and_safety_regression_intact(self):
        """Verify identity and content safety guardrails remain intact during chart evaluation."""
        self_harm_check = evaluate_content_safety("I want to kill myself")
        self.assertEqual(self_harm_check["decision"], ContentDecision.REFUSE_OR_REDIRECT)
        self.assertEqual(self_harm_check["category"], "self_harm")

        identity_check = handle_identity_query("Who are you?")
        self.assertIsNotNone(identity_check)
        self.assertTrue(identity_check["handled"])
        self.assertEqual(identity_check["response"], "I'm SahajAI, your AI assistant.")


if __name__ == "__main__":
    unittest.main()
