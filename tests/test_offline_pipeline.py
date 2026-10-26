"""
Run:  python -m unittest tests.test_offline_pipeline -v      (from the project root)

No GPU, microphone, speaker or Ollama needed — a fake Ollama HTTP server and
fake STT/TTS/audio sink stand in. These tests exercise the real threading,
streaming, interruption and memory logic.
"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db_writer  # noqa: E402
from agent_offline import OfflineAurobot, Turn  # noqa: E402
from memory import EmotionMemory  # noqa: E402
from memory_store import MemoryStore  # noqa: E402
from offline.audio_io import NullAudioOut  # noqa: E402
from offline.config import OfflineConfig  # noqa: E402
from offline.llm import OllamaClient, iter_sentences  # noqa: E402
from offline.speaker import Speaker  # noqa: E402
from offline.user_control import UserControl  # noqa: E402
from offline.vad import FRAME_BYTES, FRAME_SAMPLES, FrameVAD, UtteranceSegmenter  # noqa: E402


# ----------------------------------------------------------------- fakes
class FakeOllama:
    """Streams `reply` token-by-token with a delay, like the real /api/chat."""

    def __init__(self, reply="Hi Siva, nice to see you. How was your day? Tell me everything.",
                 token_delay=0.03):
        self.reply = reply
        self.token_delay = token_delay
        self.stream_finished_at = None
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"models": [{"name": "fake:latest"}]}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n))
                outer.requests.append(req)
                self.send_response(200)
                self.end_headers()
                if not req.get("stream", True):
                    self.wfile.write(json.dumps({"done": True}).encode())
                    return
                try:
                    words = outer.reply.split(" ")
                    for i, w in enumerate(words):
                        tok = w + ("" if i == len(words) - 1 else " ")
                        self.wfile.write(json.dumps(
                            {"message": {"role": "assistant", "content": tok}, "done": False}
                        ).encode() + b"\n")
                        self.wfile.flush()
                        time.sleep(outer.token_delay)
                    self.wfile.write(json.dumps({"done": True}).encode() + b"\n")
                    outer.stream_finished_at = time.time()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


class FakeSTT:
    def __init__(self, text="hello there"):
        self.text = text

    def transcribe(self, audio):
        return self.text


class FakeTTS:
    backend = "fake"
    sample_rate = 16000

    def synth(self, text):
        return np.zeros(int(0.012 * self.sample_rate * len(text)), dtype=np.int16).tobytes()


def make_bot(fake, tmpdir, **cfg_overrides):
    cfg = OfflineConfig(**cfg_overrides)
    llm = OllamaClient(fake.url, "fake")
    out = NullAudioOut(delay_per_chunk=0.004)
    speaker = Speaker(FakeTTS(), out)
    speaker.start()
    db_writer.start()
    bot = OfflineAurobot(
        cfg, stt=FakeSTT(), llm=llm, speaker=speaker, mic=None,
        store=MemoryStore(os.path.join(tmpdir, "t.db")),
        memory=EmotionMemory(os.path.join(tmpdir, "t.db")),
    )
    return bot, out


# ----------------------------------------------------------------- tests
class SentenceSplitter(unittest.TestCase):
    def split(self, pieces):
        return list(iter_sentences(pieces))

    def test_basic_streaming(self):
        out = self.split(["Hello th", "ere, Siva. How are", " you? I'm fine", "!"])
        self.assertEqual(out, ["Hello there, Siva.", "How are you?", "I'm fine!"])

    def test_decimals_not_split(self):
        self.assertEqual(self.split(["You have 3.5 apples. Enjoy them."]),
                         ["You have 3.5 apples.", "Enjoy them."])

    def test_markdown_and_emoji_removed(self):
        out = self.split(["**Great** news 🎉! Here's `code`."])
        self.assertTrue(all("*" not in s and "`" not in s and "🎉" not in s for s in out))

    def test_early_clause_flush_speeds_first_audio(self):
        first = next(iter_sentences(
            ["Well, that is honestly a really interesting question to think about, because "]))
        self.assertTrue(first.endswith(","))

    def test_no_empty_sentences(self):
        self.assertEqual(self.split(["...", " ! ", "Hi."]), ["Hi."])


class VADSegmenter(unittest.TestCase):
    def frames(self, ms, loud):
        n = ms // 30
        rng = np.random.default_rng(0)
        for _ in range(n):
            amp = 4000 if loud else 20
            yield (rng.normal(0, amp, FRAME_SAMPLES)).astype(np.int16).tobytes()

    def run_seg(self, plan):
        seg = UtteranceSegmenter(FrameVAD(force_energy=True), end_silence_ms=600)
        results = []
        for ms, loud in plan:
            for f in self.frames(ms, loud):
                a = seg.feed(f)
                if a is not None:
                    results.append(a)
        return results

    def test_detects_one_utterance(self):
        res = self.run_seg([(600, False), (1200, True), (1000, False)])
        self.assertEqual(len(res), 1)
        self.assertGreater(len(res[0]) / 16000, 1.1)

    def test_short_blip_ignored(self):
        self.assertEqual(self.run_seg([(600, False), (120, True), (1000, False)]), [])

    def test_two_utterances(self):
        res = self.run_seg([(600, False), (900, True), (900, False), (900, True), (900, False)])
        self.assertEqual(len(res), 2)


class UserControlRules(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MemoryStore(os.path.join(self.tmp, "c.db"))
        self.calls = []
        self.uc = UserControl(self.store, lambda fn, *a, **k: self.calls.append((fn.__name__, a, k)))

    def test_stop_silences(self):
        self.assertEqual(self.uc.handle("please be quiet"), "quiet")
        self.assertTrue(self.uc.is_quiet)

    def test_long_sentence_with_stop_does_not_silence(self):
        self.assertIsNone(self.uc.handle("I can't stop thinking about my exam tomorrow morning"))
        self.assertFalse(self.uc.is_quiet)

    def test_word_boundary(self):
        self.assertIsNone(self.uc.handle("start the stopwatch"))

    def test_reengagement_clears_quiet(self):
        self.uc.handle("stop")
        self.assertTrue(self.uc.is_quiet)
        self.uc.handle("okay what is the weather like")
        self.assertFalse(self.uc.is_quiet)

    def test_memory_routing(self):
        self.uc.handle("remember I prefer tea")
        self.uc.handle("remind me about the demo")
        kinds = [c[1][0] for c in self.calls]
        self.assertEqual(kinds, ["semantic", "task"])


class TurnPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fake = FakeOllama()

    def tearDown(self):
        self.fake.close()

    def test_turn_speaks_and_records_reply(self):
        bot, out = make_bot(self.fake, self.tmp)
        t0 = time.time()
        bot._run_turn(Turn("user", "hello", t_speech_end=t0, t_stt_done=t0))
        self.assertGreater(len(out.chunks), 0)
        self.assertEqual(bot.history[-1]["role"], "assistant")
        self.assertIn("Tell me everything", bot.history[-1]["content"])

    def test_first_audio_precedes_stream_end(self):
        bot, out = make_bot(self.fake, self.tmp)
        first = {}
        orig_write = out.write

        def timed_write(samples):
            first.setdefault("t", time.time())
            orig_write(samples)
        out.write = timed_write

        bot._run_turn(Turn("user", "hello", t_speech_end=time.time(), t_stt_done=time.time()))
        self.assertLess(first["t"], self.fake.stream_finished_at,
                        "audio should begin before the LLM has finished generating")

    def test_history_is_bounded(self):
        bot, _ = make_bot(self.fake, self.tmp, max_history_messages=6)
        self.fake.reply = "Sure thing."
        for i in range(15):
            bot._run_turn(Turn("user", f"question {i}", t_speech_end=time.time(), t_stt_done=time.time()))
        self.assertLessEqual(len(bot.history), 6)
        self.assertEqual(bot.history[0]["role"], "user")
        sent = self.fake.requests[-1]["messages"]
        self.assertLessEqual(len(sent), 6 + 2)  # history + system + current user

    def test_interrupt_stops_quickly(self):
        self.fake.reply = " ".join(["This is a long sentence number %d." % i for i in range(40)])
        bot, out = make_bot(self.fake, self.tmp)
        th = threading.Thread(target=bot._run_turn,
                              args=(Turn("user", "talk a lot", time.time(), time.time()),))
        th.start()
        time.sleep(0.6)
        t = time.time()
        bot._interrupt_all()
        th.join(timeout=5)
        self.assertFalse(th.is_alive())
        self.assertLess(time.time() - t, 1.5)
        self.assertFalse(bot.speaker.is_speaking)

    def test_proactive_silent_token_speaks_nothing(self):
        self.fake.reply = "[SILENT]"
        bot, out = make_bot(self.fake, self.tmp)
        bot._run_turn(Turn("proactive", emotion="neutral"))
        self.assertEqual(len(out.chunks), 0)
        self.assertEqual(bot.history, [])

    def test_greeting_includes_recap_and_runs(self):
        bot, out = make_bot(self.fake, self.tmp)
        bot._recap = "user: I'm working on the UAV project"
        bot._run_turn(Turn("greeting"))
        last_user_prompt = self.fake.requests[-1]["messages"][-1]["content"]
        self.assertIn("UAV project", last_user_prompt)
        self.assertGreater(len(out.chunks), 0)

    def test_llm_down_gives_spoken_fallback(self):
        bot, out = make_bot(self.fake, self.tmp)
        bot.llm.base = "http://127.0.0.1:9"   # nothing listening
        bot._run_turn(Turn("user", "hello", time.time(), time.time()))
        self.assertGreater(len(out.chunks), 0)

    def test_emotion_hint_reaches_llm(self):
        bot, _ = make_bot(self.fake, self.tmp)
        bot.fusion.update_vision({"sad": 90.0, "happy": 5.0, "neutral": 5.0})
        bot._run_turn(Turn("user", "hi", time.time(), time.time()))
        self.assertIn("looks sad", self.fake.requests[-1]["messages"][-1]["content"])
        self.assertEqual(bot.history[-2]["content"], "hi")   # hint NOT stored in history

    def test_ollama_check_reports_missing_model(self):
        from offline.llm import OllamaError
        c = OllamaClient(self.fake.url, "not-installed")
        with self.assertRaises(OllamaError) as cm:
            c.check()
        self.assertIn("ollama pull", str(cm.exception))

    def test_memory_logged(self):
        bot, _ = make_bot(self.fake, self.tmp)
        bot._run_turn(Turn("user", "hello", time.time(), time.time()))
        deadline = time.time() + 3
        while time.time() < deadline:
            if bot.memory.recent_conversation(limit=5):
                break
            time.sleep(0.05)
        self.assertTrue(any(role == "assistant" for role, _, _ in bot.memory.recent_conversation(limit=5)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
