import unittest
import json
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import guardrails
from guardrails import ChartType, ChartIntentDecision
import backend


class TestChartIntentDetection(unittest.TestCase):
    """
    Phase 1: Unit tests for evaluate_chart_intent()
    Verifies single-turn explicit intent detection, 3 supported chart types,
    false-positive prevention, and clarification on ambiguous requests.
    """

    def test_explicit_pie_chart_requests(self):
        prompts = [
            "Please create a pie chart of the market shares",
            "Can you make a pie chart showing expenses?",
            "Pie chart for the quarterly sales",
            "Show me a pie graph of user distribution",
            "I need a piechart",
            "visualize this data as a pie chart",
            "Please generate a pie chart of the revenue breakdown",
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                res = guardrails.evaluate_chart_intent(p)
                self.assertEqual(res["decision"], ChartIntentDecision.GENERATE, f"Failed for: {p}")
                self.assertEqual(res["chart_type"], ChartType.PIE, f"Wrong chart type for: {p}")

    def test_explicit_bar_chart_requests(self):
        prompts = [
            "Create a bar chart comparing regional revenue",
            "Generate a bar graph for monthly expenses",
            "Show me a bar chart of headcount",
            "Make a barchart for sales",
            "Can you plot a bar chart comparing 2023 and 2024?",
            "A column chart of product sales",
            "Please render a bar chart of the quarterly profit",
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                res = guardrails.evaluate_chart_intent(p)
                self.assertEqual(res["decision"], ChartIntentDecision.GENERATE, f"Failed for: {p}")
                self.assertEqual(res["chart_type"], ChartType.BAR, f"Wrong chart type for: {p}")

    def test_explicit_line_chart_requests(self):
        prompts = [
            "Plot a line chart of temperature changes over time",
            "Show me a line graph tracking website visits",
            "Make a linechart for the stock price trend",
            "Can you draw a line chart for the last 12 months?",
            "Trend line chart of revenue growth",
            "Generate a line chart showing daily active users",
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                res = guardrails.evaluate_chart_intent(p)
                self.assertEqual(res["decision"], ChartIntentDecision.GENERATE, f"Failed for: {p}")
                self.assertEqual(res["chart_type"], ChartType.LINE, f"Wrong chart type for: {p}")

    def test_ambiguous_chart_requests_trigger_clarification(self):
        prompts = [
            "Create a chart for this data",
            "Can you make a graph showing revenue?",
            "Show me a chart",
            "I need a graph of these numbers",
            "Visualize this data as a chart",
            "Please draw a chart",
            "Can you plot the sales data?",
            "Create a pie chart or a bar chart",  # Multiple conflicting types
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                res = guardrails.evaluate_chart_intent(p)
                self.assertEqual(res["decision"], ChartIntentDecision.CLARIFY, f"Failed for: {p}")
                self.assertEqual(res["chart_type"], ChartType.AMBIGUOUS, f"Wrong type for: {p}")
                self.assertIsNotNone(res["response"], f"Missing response for: {p}")
                self.assertIn("Pie Chart", res["response"])
                self.assertIn("Bar Chart", res["response"])
                self.assertIn("Line Chart", res["response"])

    def test_false_positive_suppression(self):
        """
        Ensures non-chart queries (summaries, explanations, tables, or words like
        'plot', 'visual', 'diagram', 'bar', 'pie') DO NOT trigger charts.
        """
        non_chart_prompts = [
            "Summarize the plot of Hamlet",
            "Can you explain the visual elements in modern UI design?",
            "Draw an architecture diagram of a microservices backend",
            "How do I bake an apple pie from scratch?",
            "I went to a juice bar with my friends yesterday",
            "Explain the plot line of Inception",
            "Can you provide a summary table of the quarterly revenue?",
            "Format this financial data as a markdown table",
            "What is a pie chart and how does it work?",
            "Explain the difference between a bar chart and a line chart",
            "Define a line graph",
            "When should I use a bar chart?",
            "Summarize this document",
            "Explain these numbers from the previous response",
            "Can you chart a course for our software architecture?",
            "What are the pros and cons of line charts?",
            "Walk me through the pie chart mentioned earlier",
        ]
        for p in non_chart_prompts:
            with self.subTest(prompt=p):
                res = guardrails.evaluate_chart_intent(p)
                self.assertEqual(res["decision"], ChartIntentDecision.NONE, f"False positive triggered for: {p}")
                self.assertEqual(res["chart_type"], ChartType.NONE, f"Wrong type for: {p}")

    def test_chart_directive_formatting(self):
        pie_dir = guardrails.get_chart_directive(ChartType.PIE)
        bar_dir = guardrails.get_chart_directive(ChartType.BAR)
        line_dir = guardrails.get_chart_directive(ChartType.LINE)

        self.assertIn('"type": "pie"', pie_dir)
        self.assertIn('"type": "bar"', bar_dir)
        self.assertIn('"type": "line"', line_dir)

        # All must mandate ```chart start and real data extraction
        for d in [pie_dir, bar_dir, line_dir]:
            self.assertIn("```chart", d)
            self.assertIn("REAL DATA ONLY", d)
            self.assertIn("START WITH ```chart DIRECTLY", d)


class TestConsecutiveTurnIndependence(unittest.TestCase):
    """
    Verifies that previous chart turns do not cause chart generation in later turns.
    Intent detection is strictly derived from the current user prompt only.
    """

    def test_consecutive_turn_eval(self):
        turn1_prompt = "Generate a bar chart of the sales by region."
        eval_t1 = guardrails.evaluate_chart_intent(turn1_prompt)
        self.assertEqual(eval_t1["decision"], ChartIntentDecision.GENERATE)
        self.assertEqual(eval_t1["chart_type"], ChartType.BAR)

        turn2_prompt = "Explain these regional numbers in detail and summarize the key trends."
        eval_t2 = guardrails.evaluate_chart_intent(turn2_prompt)
        self.assertEqual(eval_t2["decision"], ChartIntentDecision.NONE)
        self.assertEqual(eval_t2["chart_type"], ChartType.NONE)

    def test_build_messages_turn_independence(self):
        """
        Verifies build_messages attaches [NO CHART REQUESTED] when current turn is non-chart,
        even if history contains a chart response from the assistant.
        """
        history_with_chart = [
            {"role": "user", "content": "Create a bar chart of revenue."},
            {"role": "assistant", "content": '```chart\n{"type": "bar", "title": "Rev", "data": [{"name": "A", "value": 10}]}\n```\nHere is the chart.'}
        ]
        req_mock = MagicMock()
        req_mock.json = {
            "meta": {
                "content": {
                    "conversation": history_with_chart,
                    "internet_access": False,
                    "parts": [{"content": "What does this data mean for Q3 planning?"}]
                }
            },
            "conversation_id": "test_conv_turn2"
        }

        with patch("backend.request", req_mock):
            messages = backend.build_messages(jailbreak="default")
            last_msg = messages[-1]
            self.assertEqual(last_msg["role"], "user")
            self.assertIn("[NO CHART REQUESTED]", last_msg["content"])
            self.assertNotIn("[MANDATORY INSTRUCTION FOR", last_msg["content"])


class TestChartPipelineIntegration(unittest.TestCase):
    """
    Verifies backend pipeline streaming behavior for ambiguous chart intent
    and preservation of DAN/EvilBOT modes.
    """

    def setUp(self):
        from run import app
        self.app = app
        self.client = self.app.test_client()

    def test_ambiguous_chart_fastpath_stream(self):
        """
        In Standard Mode, an ambiguous chart query yields immediate clarification via SSE.
        """
        payload = {
            "conversation_id": "conv_clarify_test",
            "model": "mistral:latest",
            "jailbreak": "default",
            "meta": {
                "content": {
                    "conversation": [],
                    "internet_access": False,
                    "parts": [{"role": "user", "content": "Can you draw a chart for this data?"}]
                }
            }
        }
        res = self.client.post('/backend-api/v2/conversation', json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn('text/event-stream', res.content_type)
        output = res.get_data(as_text=True)

        self.assertIn("Which type of chart would you like me to generate?", output)
        self.assertIn("Pie Chart", output)
        self.assertIn("Bar Chart", output)
        self.assertIn("Line Chart", output)
        self.assertIn("data: [DONE]", output)

    def test_dan_mode_preservation(self):
        """
        Verifies that in DAN Mode ('gpt-dan-11.0'), Standard Mode chart intent guardrails
        are not applied, preserving original legacy prompt injection and behavior.
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
            "conversation_id": "conv_dan_test"
        }

        with patch("backend.request", req_mock):
            messages = backend.build_messages(jailbreak="gpt-dan-11.0")
            # In DAN mode, prompt should have legacy instructions and DAN instructions
            has_dan = any("DAN" in m.get("content", "") for m in messages)
            self.assertTrue(has_dan, "DAN mode instructions missing")
            # Should NOT contain Standard Mode system prompt
            has_sahaj_ident = any("You are SahajAI" in m.get("content", "") for m in messages)
            self.assertFalse(has_sahaj_ident, "Standard Mode identity leaked into DAN mode")


if __name__ == '__main__':
    unittest.main()
