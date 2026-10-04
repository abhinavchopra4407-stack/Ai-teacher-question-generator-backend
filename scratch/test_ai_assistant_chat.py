import sys
import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Ensure backend path is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.database import Base, engine, get_db

class TestAIAssistantChat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)
        cls.client = TestClient(app)

    def test_complete_chat_workflow_and_intents(self):
        # 1. Register User A
        res_reg_a = self.client.post("/api/auth/register", json={
            "email": "teacher_chat_a@test.com",
            "password": "Password123!",
            "full_name": "Teacher Chat A"
        })
        if res_reg_a.status_code != 200:
            res_reg_a = self.client.post("/api/auth/login", json={
                "email": "teacher_chat_a@test.com",
                "password": "Password123!"
            })
        token_a = res_reg_a.json()["access_token"]
        headers_a = {"Authorization": f"Bearer {token_a}"}

        # Mock AI response generator to simulate real model responses based on intent
        def mock_ai_response(messages, user_api_key=None):
            last_msg = messages[-1]["content"].lower() if messages else ""
            if "india become independent" in last_msg:
                return "India became independent on 15 August 1947."
            elif "25 * 16" in last_msg or "25 × 16" in last_msg:
                return "25 × 16 = 400."
            elif "20 questions about photosynthesis" in last_msg:
                return "\n".join([f"{i}. Sample photosynthesis question {i}?" for i in range(1, 21)])
            elif "previous question" in last_msg:
                prev_user_msgs = [m["content"] for m in messages if m["role"] == "user"]
                if len(prev_user_msgs) >= 2:
                    return f"Your previous question was: '{prev_user_msgs[-2]}'"
                return "Your previous question was about photosynthesis."
            elif "mitosis and meiosis" in last_msg:
                return "Mitosis produces 2 identical diploid somatic cells, whereas meiosis produces 4 genetically unique haploid gametes."
            elif "photosynthesis" in last_msg:
                return "Photosynthesis is the biological process by which green plants use sunlight, water, and carbon dioxide to produce glucose and oxygen."
            return f"Answer for query: {messages[-1]['content']}"

        with patch("app.ai_engine.generate_chat_response", side_effect=mock_ai_response):
            # Test 1: Direct Factual Query
            res1 = self.client.post("/api/chat/send", json={
                "message": "When did India become independent?"
            }, headers=headers_a)
            self.assertEqual(res1.status_code, 200)
            conv_data1 = res1.json()
            conv_id = conv_data1["id"]
            ai_ans1 = conv_data1["messages"][-1]["content"]
            self.assertIn("15 August 1947", ai_ans1)
            self.assertNotIn("### Teaching Guide:", ai_ans1) # Verify no forced teaching guide!

            # Test 2: Math Direct Calculation
            res2 = self.client.post("/api/chat/send", json={
                "conversation_id": conv_id,
                "message": "What is 25 * 16?"
            }, headers=headers_a)
            self.assertEqual(res2.status_code, 200)
            ai_ans2 = res2.json()["messages"][-1]["content"]
            self.assertIn("400", ai_ans2)

            # Test 3: Specific Quantity Request (20 Questions)
            res3 = self.client.post("/api/chat/send", json={
                "conversation_id": conv_id,
                "message": "Generate 20 questions about photosynthesis."
            }, headers=headers_a)
            self.assertEqual(res3.status_code, 200)
            ai_ans3 = res3.json()["messages"][-1]["content"]
            self.assertIn("20.", ai_ans3)

            # Test 4: Multi-Turn Context Follow-Up
            res4 = self.client.post("/api/chat/send", json={
                "conversation_id": conv_id,
                "message": "What was my previous question?"
            }, headers=headers_a)
            self.assertEqual(res4.status_code, 200)
            ai_ans4 = res4.json()["messages"][-1]["content"]
            self.assertIn("Generate 20 questions about photosynthesis", ai_ans4)

            # Test 5: Concept Comparison
            res5 = self.client.post("/api/chat/send", json={
                "conversation_id": conv_id,
                "message": "Explain the difference between mitosis and meiosis."
            }, headers=headers_a)
            self.assertEqual(res5.status_code, 200)
            ai_ans5 = res5.json()["messages"][-1]["content"]
            self.assertIn("diploid", ai_ans5.lower())

        # Test Error Handling: When AI Key is Missing / Service Fails
        def mock_ai_failure(messages, user_api_key=None):
            raise ValueError("AI Assistant provider is currently unavailable. Please check GEMINI_API_KEY.")

        with patch("app.ai_engine.generate_chat_response", side_effect=mock_ai_failure):
            res_fail = self.client.post("/api/chat/send", json={
                "message": "Test question when key is missing"
            }, headers=headers_a)
            self.assertEqual(res_fail.status_code, 503)
            self.assertIn("AI Assistant provider is currently unavailable", res_fail.json()["detail"])

    def test_ai_diagnostics_endpoint(self):
        res = self.client.get("/api/ai/diagnostics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("providers", data)
        self.assertIn("groq", data["providers"])
        self.assertIn("gemini", data["providers"])
        # Ensure no secret keys are exposed
        raw_json = res.text
        self.assertNotIn("AIza", raw_json)
        self.assertNotIn("gsk_", raw_json)

    def test_voice_transcription_suite(self):
        # 1. Register User for Voice Testing
        res_reg = self.client.post("/api/auth/register", json={
            "email": "teacher_voice_test@test.com",
            "password": "Password123!",
            "full_name": "Teacher Voice Test"
        })
        if res_reg.status_code != 200:
            res_reg = self.client.post("/api/auth/login", json={
                "email": "teacher_voice_test@test.com",
                "password": "Password123!"
            })
        token = res_reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Mock successful Groq audio transcription response
        def mock_transcribe_success(file_bytes, filename="recording.webm", language=None, user_api_key=None):
            if language == "hi":
                return "प्रकाश संश्लेषण क्या है? (What is photosynthesis?)"
            return "Explain the process of photosynthesis for Class 8 students."

        # Test A: Successful Transcription (English & Hindi)
        with patch("app.ai_engine.transcribe_audio_with_groq", side_effect=mock_transcribe_success):
            dummy_audio = b"RIFF....WAVEfmt ....data...." + b"\x00" * 200
            
            res_en = self.client.post(
                "/api/ai/voice/transcribe",
                files={"file": ("recording.webm", dummy_audio, "audio/webm")},
                data={"language": "en"},
                headers=headers
            )
            self.assertEqual(res_en.status_code, 200)
            self.assertIn("photosynthesis", res_en.json()["transcript"].lower())

            res_hi = self.client.post(
                "/api/ai/voice/transcribe",
                files={"file": ("recording_hi.webm", dummy_audio, "audio/webm")},
                data={"language": "hi"},
                headers=headers
            )
            self.assertEqual(res_hi.status_code, 200)
            self.assertIn("प्रकाश संश्लेषण", res_hi.json()["transcript"])

        # Test B: Unauthenticated Request (401)
        res_unauth = self.client.post(
            "/api/ai/voice/transcribe",
            files={"file": ("recording.webm", b"1234567890", "audio/webm")}
        )
        self.assertEqual(res_unauth.status_code, 401)

        # Test C: Empty Audio Payload (400)
        res_empty = self.client.post(
            "/api/ai/voice/transcribe",
            files={"file": ("empty.webm", b"", "audio/webm")},
            headers=headers
        )
        self.assertEqual(res_empty.status_code, 400)
        self.assertIn("empty audio", res_empty.json()["detail"].lower())

        # Test D: Oversized Audio Payload > 10MB (400)
        oversized_bytes = b"0" * (10 * 1024 * 1024 + 1024)
        res_large = self.client.post(
            "/api/ai/voice/transcribe",
            files={"file": ("huge.webm", oversized_bytes, "audio/webm")},
            headers=headers
        )
        self.assertEqual(res_large.status_code, 400)
        self.assertIn("exceeds maximum limit", res_large.json()["detail"].lower())

        # Test E: Unsupported File Format (400)
        res_unsupported = self.client.post(
            "/api/ai/voice/transcribe",
            files={"file": ("document.exe", b"executable bytes", "application/x-msdownload")},
            headers=headers
        )
        self.assertEqual(res_unsupported.status_code, 400)
        self.assertIn("unsupported audio format", res_unsupported.json()["detail"].lower())

        # Test F: Missing API Key Error Handling (503)
        def mock_transcribe_nokey(*args, **kwargs):
            raise ValueError("Voice transcription provider is not configured. Please ensure GROQ_API_KEY is configured in server environment variables.")

        with patch("app.ai_engine.transcribe_audio_with_groq", side_effect=mock_transcribe_nokey):
            res_nokey = self.client.post(
                "/api/ai/voice/transcribe",
                files={"file": ("recording.webm", b"1234567890", "audio/webm")},
                headers=headers
            )
            self.assertEqual(res_nokey.status_code, 503)

        # Test G: Provider Authentication Failure (401)
        def mock_transcribe_401(*args, **kwargs):
            raise ValueError("Groq API authentication failed (HTTP 401). Please check server GROQ_API_KEY configuration.")

        with patch("app.ai_engine.transcribe_audio_with_groq", side_effect=mock_transcribe_401):
            res_401 = self.client.post(
                "/api/ai/voice/transcribe",
                files={"file": ("recording.webm", b"1234567890", "audio/webm")},
                headers=headers
            )
            self.assertEqual(res_401.status_code, 401)
            self.assertNotIn("gsk_", res_401.text) # Verify no API key leaked!

if __name__ == "__main__":
    unittest.main()

