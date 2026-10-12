"""
End-to-End Integration Tests for SahajAI Backend Pipeline Stream
Tests SSE streaming outputs for Standard Mode identity queries,
model inquiry, and verification that DAN / EvilBOT bypass Standard Mode guardrails.
"""

import unittest
import json
import os
import sys

# Ensure backend directory is in path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from run import app


class TestPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.client = self.app.test_client()

    def _parse_sse(self, response_data: str):
        events = []
        for line in response_data.split('\n'):
            line = line.strip()
            if line.startswith('data: '):
                payload_str = line[6:].strip()
                if payload_str == '[DONE]':
                    events.append({'type': '[DONE]'})
                else:
                    try:
                        events.append(json.loads(payload_str))
                    except Exception:
                        events.append({'raw': payload_str})
        return events

    def test_standard_mode_who_are_you_sse_stream(self):
        """Standard Mode 'Who are you?' should return SahajAI identity through valid SSE."""
        payload = {
            "conversation_id": "test_conv_identity_1",
            "model": "mistral:latest",
            "jailbreak": "default",
            "meta": {
                "content": {
                    "conversation": [],
                    "internet_access": False,
                    "parts": [{"role": "user", "content": "Who are you?"}]
                }
            }
        }
        res = self.client.post('/backend-api/v2/conversation', json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn('text/event-stream', res.content_type)
        
        events = self._parse_sse(res.get_data(as_text=True))
        
        # Check presence of content event with expected text
        content_events = [e for e in events if e.get('type') == 'content']
        self.assertTrue(len(content_events) > 0)
        accumulated_text = "".join(e.get('delta', '') for e in content_events)
        self.assertEqual(accumulated_text, "I'm SahajAI, your AI assistant.")

        # Check stream terminates with [DONE]
        done_events = [e for e in events if e.get('type') == '[DONE]']
        self.assertEqual(len(done_events), 1)

    def test_standard_mode_who_created_you_sse_stream(self):
        """Standard Mode 'Who created you?' should attribute to the SahajAI Team."""
        payload = {
            "conversation_id": "test_conv_identity_2",
            "model": "mistral:latest",
            "jailbreak": "default",
            "meta": {
                "content": {
                    "conversation": [],
                    "internet_access": False,
                    "parts": [{"role": "user", "content": "Who created you?"}]
                }
            }
        }
        res = self.client.post('/backend-api/v2/conversation', json=payload)
        self.assertEqual(res.status_code, 200)
        
        events = self._parse_sse(res.get_data(as_text=True))
        content_events = [e for e in events if e.get('type') == 'content']
        accumulated_text = "".join(e.get('delta', '') for e in content_events)
        self.assertEqual(accumulated_text, "SahajAI was created by the SahajAI Team.")

    def test_standard_mode_are_you_qwen_sse_stream(self):
        """Standard Mode 'Are you Qwen?' must respond with SahajAI identity without leaking model names."""
        payload = {
            "conversation_id": "test_conv_identity_3",
            "model": "mistral:latest",
            "jailbreak": "default",
            "meta": {
                "content": {
                    "conversation": [],
                    "internet_access": False,
                    "parts": [{"role": "user", "content": "Are you Qwen?"}]
                }
            }
        }
        res = self.client.post('/backend-api/v2/conversation', json=payload)
        self.assertEqual(res.status_code, 200)
        
        events = self._parse_sse(res.get_data(as_text=True))
        content_events = [e for e in events if e.get('type') == 'content']
        accumulated_text = "".join(e.get('delta', '') for e in content_events)
        self.assertEqual(accumulated_text, "I'm SahajAI, your AI assistant.")
        self.assertNotIn("qwen", accumulated_text.lower())

    def test_standard_mode_model_inquiry_sse_stream(self):
        """Standard Mode 'Which AI model powers you right now?' must not disclose model name."""
        payload = {
            "conversation_id": "test_conv_identity_4",
            "model": "mistral:latest",
            "jailbreak": "default",
            "meta": {
                "content": {
                    "conversation": [],
                    "internet_access": False,
                    "parts": [{"role": "user", "content": "Which AI model powers you right now?"}]
                }
            }
        }
        res = self.client.post('/backend-api/v2/conversation', json=payload)
        self.assertEqual(res.status_code, 200)
        
        events = self._parse_sse(res.get_data(as_text=True))
        content_events = [e for e in events if e.get('type') == 'content']
        accumulated_text = "".join(e.get('delta', '') for e in content_events)
        expected_msg = "I'm SahajAI, an AI assistant created by the SahajAI Team. I can help you with questions, writing, coding, and more."
        self.assertEqual(accumulated_text, expected_msg)
        self.assertNotIn("mistral", accumulated_text.lower())

    def test_dan_mode_bypasses_standard_guardrail_fast_path(self):
        """DAN Mode (gpt-dan-11.0) must bypass standard identity fast path."""
        import guardrails
        self.assertFalse(guardrails.is_standard_mode("gpt-dan-11.0"))

    def test_evilbot_mode_bypasses_standard_guardrail_fast_path(self):
        """EvilBOT Mode (gpt-evil) must bypass standard identity fast path."""
        import guardrails
        self.assertFalse(guardrails.is_standard_mode("gpt-evil"))


if __name__ == '__main__':
    unittest.main()
