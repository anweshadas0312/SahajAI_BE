import unittest
import json
import re
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import guardrails

class TestLineChartBugReproduction(unittest.TestCase):
    def test_reproduce_new_date_and_space_colon_header(self):
        """
        Message 1919 in DB had:
        ``` chart:
        {
          "type": "line",
          "title": "Total Quantity Ordered by Order Date",
          "data": [
            {"name": new Date(2017, 0, 2), "value": 49384},
            {"name": new Date(2017, 0, 3), "value": 56908}
          ]
        }
        ```
        In the current implementation:
        validate_and_sanitize_chart_response ignores it because of the leading space and colon,
        returning is_valid=True and letting the malformed JSON through to the frontend.
        The frontend then fails with SyntaxError in JSON.parse.
        """
        raw_msg = (
            "``` chart:\n"
            "{\n"
            '  "type": "line",\n'
            '  "title": "Total Quantity Ordered by Order Date",\n'
            '  "data": [\n'
            '    {"name": new Date(2017, 0, 2), "value": 49384},\n'
            '    {"name": new Date(2017, 0, 3), "value": 56908}\n'
            "  ]\n"
            "}\n"
            "```"
        )
        is_v, clean, payload, err = guardrails.validate_and_sanitize_chart_response(
            raw_msg,
            expected_chart_type="line"
        )
        # In the fixed implementation, this should either:
        # 1. Sanitize new Date(...) into valid JSON string dates and return is_v=True with valid payload, OR
        # 2. If rejected, cleanly replace the block with limitation message (is_v=False, clean has limitation text, no broken ```chart block)
        # Under NO circumstances should raw_msg pass through unvalidated with new Date(...) intact!
        self.assertNotIn("new Date(", clean)

    def test_reproduce_prose_in_chart_block(self):
        """
        Message 1923 in DB had:
        ```chart: To create a line chart of total quantity ordered by order date using only the provided `order.csv` file:
        - **File Overview & Purpose**: ...
        ```
        Currently, because there is no '{', validate_and_sanitize_chart_response returns is_v=True,
        leaving the ```chart: ...``` block intact, which ReactMarkdown treats as language-chart
        and frontend JSON.parse throws 'Unable to parse chart specification payload.'
        """
        raw_msg = (
            "```chart: To create a line chart of total quantity ordered by order date using only the provided `order.csv` file:\n"
            "- **File Overview & Purpose**: transactional data.\n"
            "```"
        )
        is_v, clean, payload, err = guardrails.validate_and_sanitize_chart_response(
            raw_msg,
            expected_chart_type="line"
        )
        # Should detect chart block has no valid JSON, return is_v=False,
        # and replace the broken block with clear limitation explanation.
        self.assertFalse(is_v)
        self.assertIn("Chart Generation Limitation", clean)
        self.assertNotIn("```chart", clean)

    def test_order_csv_date_ordering_validation(self):
        """
        order.csv has dates like '01-Jan-17', '02-Jan-17', '03-Jan-17'.
        validate_line_chart_ordering should recognize these dates and enforce chronological ordering.
        """
        valid_order_dates = [
            {"name": "01-Jan-17", "value": 7},
            {"name": "02-Jan-17", "value": 121},
            {"name": "03-Jan-17", "value": 25}
        ]
        is_ord, err = guardrails.validate_line_chart_ordering(valid_order_dates)
        self.assertTrue(is_ord)
        self.assertIsNone(err)

        out_of_order_dates = [
            {"name": "03-Jan-17", "value": 25},
            {"name": "01-Jan-17", "value": 7},
            {"name": "02-Jan-17", "value": 121}
        ]
        is_ord2, err2 = guardrails.validate_line_chart_ordering(out_of_order_dates)
        self.assertFalse(is_ord2)
        self.assertIn("chronological", err2.lower())

    def test_derive_tabular_chart_data_from_order_csv(self):
        """
        Derives discrete daily quantities directly from order.csv:
        - 01-Jan-17: 7
        - 02-Jan-17: 121
        - 03-Jan-17: 25
        Chronologically ordered, discrete daily aggregation.
        """
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'uploads', '3e80ef2a-67db-4e67-a0ca-7b6a89d2117d.csv')
        if not os.path.exists(csv_path):
            # Fallback to any order.csv found in uploads
            uploads_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
            for f in os.listdir(uploads_dir):
                if f.endswith('.csv') and os.path.getsize(os.path.join(uploads_dir, f)) == 6638:
                    csv_path = os.path.join(uploads_dir, f)
                    break

        derived = guardrails.derive_tabular_chart_data(
            csv_path,
            "Using the earlier order.csv file, create a line chart of total quantity ordered by order date. Use only the order data and aggregate quantities by date.",
            "line"
        )
        self.assertIsNotNone(derived)
        self.assertEqual(derived['type'], 'line')
        self.assertEqual(len(derived['data']), 3)
        self.assertEqual(derived['data'][0], {"name": "01-Jan-17", "value": 7})
        self.assertEqual(derived['data'][1], {"name": "02-Jan-17", "value": 121})
        self.assertEqual(derived['data'][2], {"name": "03-Jan-17", "value": 25})

    def test_validate_and_sanitize_recovers_with_derived_chart(self):
        """
        When the LLM output has duplicate dates or malformed JSON,
        validate_and_sanitize_chart_response uses the verified derived_chart
        to produce a clean, valid chart specification payload.
        """
        derived_chart = {
            "type": "line",
            "title": "Total Quantity Ordered by Order Date",
            "data": [
                {"name": "01-Jan-17", "value": 7},
                {"name": "02-Jan-17", "value": 121},
                {"name": "03-Jan-17", "value": 25}
            ]
        }
        # Model produced duplicate X-axis points
        model_output_duplicate_dates = (
            "```chart\n"
            "{\n"
            '  "type": "line",\n'
            '  "title": "Total Quantity Ordered by Order Date",\n'
            '  "data": [\n'
            '    {"name": "02-Jan-17", "value": 10},\n'
            '    {"name": "02-Jan-17", "value": 20},\n'
            '    {"name": "03-Jan-17", "value": 25}\n'
            "  ]\n"
            "}\n"
            "```"
        )
        is_v, clean, payload, err = guardrails.validate_and_sanitize_chart_response(
            model_output_duplicate_dates,
            expected_chart_type="line",
            derived_chart=derived_chart
        )
        self.assertTrue(is_v)
        self.assertIsNone(err)
        self.assertIsNotNone(payload)
        self.assertEqual(payload['data'][0]['value'], 7)
        self.assertEqual(payload['data'][1]['value'], 121)
        self.assertEqual(payload['data'][2]['value'], 25)

if __name__ == "__main__":
    unittest.main()
