import unittest
import json
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import guardrails
from guardrails import ChartType
import backend


class TestChartDataAndSchemaValidation(unittest.TestCase):
    """
    Phase 3: Automated tests for Chart Data & Schema Validation.
    """

    def test_valid_pie_bar_and_line_chart_payloads(self):
        valid_pie = {
            "type": "pie",
            "title": "Market Share Breakdown",
            "data": [
                {"name": "Chrome", "value": 65},
                {"name": "Safari", "value": 20},
                {"name": "Firefox", "value": 15}
            ]
        }
        is_valid, err = guardrails.validate_chart_payload(valid_pie, expected_type="pie")
        self.assertTrue(is_valid)
        self.assertIsNone(err)

        valid_bar = {
            "type": "bar",
            "title": "Quarterly Revenue",
            "data": [
                {"name": "Q1", "value": 100},
                {"name": "Q2", "value": 150}
            ]
        }
        is_valid, err = guardrails.validate_chart_payload(valid_bar, expected_type="bar")
        self.assertTrue(is_valid)
        self.assertIsNone(err)

        valid_line = {
            "type": "line",
            "title": "Annual Stock Trend",
            "data": [
                {"name": "2020", "value": 120},
                {"name": "2021", "value": 140},
                {"name": "2022", "value": 160}
            ]
        }
        is_valid, err = guardrails.validate_chart_payload(valid_line, expected_type="line")
        self.assertTrue(is_valid)
        self.assertIsNone(err)

    def test_malformed_json_and_missing_fields(self):
        # Missing 'title'
        missing_title = {
            "type": "bar",
            "data": [{"name": "A", "value": 10}]
        }
        is_valid, err = guardrails.validate_chart_payload(missing_title)
        self.assertFalse(is_valid)
        self.assertIn("missing required field: 'title'", err)

        # Missing 'data'
        missing_data = {
            "type": "bar",
            "title": "Some Chart"
        }
        is_valid, err = guardrails.validate_chart_payload(missing_data)
        self.assertFalse(is_valid)
        self.assertIn("missing required field: 'data'", err)

        # Malformed JSON in text response
        malformed_text = "```chart\n{\n  \"type\": \"pie\",\n  \"title\": \"Broken\",\n  \"data\": [ {\"name\": 10, value: } ]\n}\n```\nHere is the data."
        is_v, clean_text, payload, v_err = guardrails.validate_and_sanitize_chart_response(malformed_text, expected_chart_type="pie")
        self.assertFalse(is_v)
        self.assertIsNone(payload)
        self.assertIn("malformed JSON", clean_text)
        self.assertNotIn("```chart", clean_text)

    def test_unsupported_chart_types_and_invalid_numeric_values(self):
        # Unsupported chart type 'scatter'
        unsupported = {
            "type": "scatter",
            "title": "Scatter Plot",
            "data": [{"name": "P1", "value": 10}, {"name": "P2", "value": 20}]
        }
        is_valid, err = guardrails.validate_chart_payload(unsupported)
        self.assertFalse(is_valid)
        self.assertIn("Unsupported chart type 'scatter'", err)

        # String instead of numeric value (no silent coercion)
        str_val = {
            "type": "bar",
            "title": "Category Chart",
            "data": [{"name": "Cat A", "value": "one hundred"}]
        }
        is_valid, err = guardrails.validate_chart_payload(str_val)
        self.assertFalse(is_valid)
        self.assertIn("invalid non-numeric value", err)

        # Boolean instead of numeric value
        bool_val = {
            "type": "bar",
            "title": "Category Chart",
            "data": [{"name": "Cat A", "value": True}]
        }
        is_valid, err = guardrails.validate_chart_payload(bool_val)
        self.assertFalse(is_valid)
        self.assertIn("invalid non-numeric value", err)

    def test_negative_and_zero_total_pie_charts(self):
        # Negative value in Pie Chart
        neg_pie = {
            "type": "pie",
            "title": "Budget",
            "data": [
                {"name": "Marketing", "value": 500},
                {"name": "Deficit", "value": -200}
            ]
        }
        is_valid, err = guardrails.validate_chart_payload(neg_pie)
        self.assertFalse(is_valid)
        self.assertIn("cannot have negative values", err)

        # Zero total Pie Chart
        zero_pie = {
            "type": "pie",
            "title": "Zero Allocation",
            "data": [
                {"name": "Dept A", "value": 0},
                {"name": "Dept B", "value": 0}
            ]
        }
        is_valid, err = guardrails.validate_chart_payload(zero_pie)
        self.assertFalse(is_valid)
        self.assertIn("total value must be greater than zero", err)

        # Single category Pie Chart (cannot establish proportions)
        single_pie = {
            "type": "pie",
            "title": "Single Item",
            "data": [{"name": "Sole Slice", "value": 100}]
        }
        is_valid, err = guardrails.validate_chart_payload(single_pie)
        self.assertFalse(is_valid)
        self.assertIn("requires at least two categories", err)

    def test_valid_negative_bar_chart_values(self):
        # Bar charts legitimately allow negative numbers (e.g. profit/loss, temperature)
        valid_neg_bar = {
            "type": "bar",
            "title": "Net Profit and Loss",
            "data": [
                {"name": "Q1 Profit", "value": 500},
                {"name": "Q2 Loss", "value": -150},
                {"name": "Q3 Profit", "value": 300}
            ]
        }
        is_valid, err = guardrails.validate_chart_payload(valid_neg_bar, expected_type="bar")
        self.assertTrue(is_valid)
        self.assertIsNone(err)

    def test_invalid_line_chart_ordering_and_axes(self):
        # Out-of-order years
        out_of_order_years = {
            "type": "line",
            "title": "Stock Growth",
            "data": [
                {"name": "2022", "value": 150},
                {"name": "2020", "value": 100},
                {"name": "2021", "value": 120}
            ]
        }
        is_valid, err = guardrails.validate_chart_payload(out_of_order_years, expected_type="line")
        self.assertFalse(is_valid)
        self.assertIn("not in chronological order", err)

        # Duplicate dates on X-axis
        duplicate_dates = {
            "type": "line",
            "title": "Daily Users",
            "data": [
                {"name": "2024-01-01", "value": 100},
                {"name": "2024-01-01", "value": 120}
            ]
        }
        is_valid, err = guardrails.validate_chart_payload(duplicate_dates, expected_type="line")
        self.assertFalse(is_valid)
        self.assertIn("duplicate categories/time points", err)

    def test_empty_datasets_rejected(self):
        empty_data = {
            "type": "bar",
            "title": "Empty Data",
            "data": []
        }
        is_valid, err = guardrails.validate_chart_payload(empty_data)
        self.assertFalse(is_valid)
        self.assertIn("cannot be empty", err)

    def test_fabricated_cross_file_data_rejected(self):
        doc_context = "This file contains regional employee counts for HR: Engineering has 45 staff, Marketing has 20 staff, Sales has 35 staff."
        
        # Fabricated dummy categories ("Gold", "Silver") from legacy prompt defaults
        dummy_chart = {
            "type": "pie",
            "title": "Precious Metals Breakdown",
            "data": [
                {"name": "Gold", "value": 22956},
                {"name": "Silver", "value": 3308}
            ]
        }
        is_valid, err = guardrails.validate_chart_payload(dummy_chart, file_context=doc_context)
        self.assertFalse(is_valid)
        self.assertIn("fabricated dummy category 'Gold'", err)

        # Real extracted categories from active document
        real_chart = {
            "type": "pie",
            "title": "HR Staff Distribution",
            "data": [
                {"name": "Engineering", "value": 45},
                {"name": "Marketing", "value": 20},
                {"name": "Sales", "value": 35}
            ]
        }
        is_valid, err = guardrails.validate_chart_payload(real_chart, file_context=doc_context)
        self.assertTrue(is_valid)
        self.assertIsNone(err)

    def test_sse_behavior_when_chart_validation_fails(self):
        """
        Verifies that an invalid chart payload in an assistant response is cleanly converted
        into a readable explanation rather than a broken chart codeblock.
        """
        invalid_response = (
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
            "Here is the detailed explanation of operating expenses and revenue."
        )

        is_v, clean_text, payload, v_err = guardrails.validate_and_sanitize_chart_response(
            invalid_response,
            expected_chart_type="pie"
        )
        self.assertFalse(is_v)
        self.assertNotIn("```chart", clean_text)
        self.assertIn("Chart Generation Limitation", clean_text)
        self.assertIn("cannot have negative values", clean_text)
        self.assertIn("Here is the detailed explanation", clean_text)


if __name__ == '__main__':
    unittest.main()
