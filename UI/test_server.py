import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from UI import server


def log_line(frame_id):
    return json.dumps({"frame_id": frame_id , "elapsed": frame_id * 0.1 , "data": {"steering": 0.1}}).encode("utf-8") + b"\n"


class TestLogs(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix = "test-logs-" , dir = server.ui_dir)
        self.log_dir = Path(self.directory.name)
        self.original_dir = server.log_dir
        server.log_dir = self.log_dir
        server.replay_cache.clear()

    def tearDown(self):
        server.log_dir = self.original_dir
        server.replay_cache.clear()
        self.assertEqual(self.log_dir.resolve().parent , server.ui_dir)
        self.directory.cleanup()

    def test_latest_ignores_partial_and_malformed_lines(self):
        path = self.log_dir / "run.jsonl"
        path.write_bytes(log_line(0) + log_line(1) + b"broken\n" + log_line(2)[:-1])
        self.assertEqual(server.latest_record(path)["frame_id"] , 1)

    def test_tail_does_not_parse_partial_first_line(self):
        path = self.log_dir / "run.jsonl"
        path.write_bytes(log_line(0) + log_line(1) + log_line(2))
        with patch.object(server , "tail_size" , len(log_line(1)) + len(log_line(2)) - 1):
            self.assertEqual(server.latest_record(path)["frame_id"] , 2)

    def test_nonfinite_numbers_become_null(self):
        line = b'{"data":{"ttc":Infinity,"a":NaN,"b":-Infinity,"c":1e999,"nested":[0.2,Infinity]}}\n'
        record = server.read_record(line)
        self.assertEqual(record["data"] , {"ttc": None , "a": None , "b": None , "c": None , "nested": [0.2 , None]})
        json.dumps(record , allow_nan = False)

    def test_replay_skips_bad_lines_and_finishes_partial_line(self):
        path = self.log_dir / "run.jsonl"
        path.write_bytes(log_line(0) + b"broken\n" + log_line(1)[:-1])
        record , index , count = server.replay_record(path , 10)
        self.assertEqual((record["frame_id"] , index , count) , (0 , 0 , 1))
        with path.open("ab") as file:
            file.write(b"\n" + log_line(2))
        record , index , count = server.replay_record(path , 1)
        self.assertEqual((record["frame_id"] , index , count) , (1 , 1 , 3))
        record , index , count = server.replay_record(path , 99)
        self.assertEqual((record["frame_id"] , index , count) , (2 , 2 , 3))

    def test_replay_handles_truncation(self):
        path = self.log_dir / "run.jsonl"
        path.write_bytes(log_line(0) + log_line(1))
        server.replay_record(path , 1)
        path.write_bytes(log_line(3))
        record , index , count = server.replay_record(path , 1)
        self.assertEqual((record["frame_id"] , index , count) , (3 , 0 , 1))

    def test_live_stale_replay_and_missing(self):
        self.assertIsNone(server.get_state("latest")["record"])
        path = self.log_dir / "run.jsonl"
        path.write_bytes(log_line(0))
        state = server.get_state("latest")
        self.assertTrue(state["live"])
        old_time = time.time() - 60
        os.utime(path , (old_time , old_time))
        state = server.get_state("latest")
        self.assertFalse(state["live"])
        self.assertGreater(state["age_s"] , 50)
        state = server.get_state("run.jsonl")
        self.assertEqual(state["mode"] , "replay")
        self.assertFalse(state["live"])

    def test_rejects_parent_absolute_and_nested_paths(self):
        for name in ("../run.jsonl" , "..\\run.jsonl" , str(self.log_dir / "run.jsonl") , "sub/run.jsonl" , "run.txt"):
            with self.subTest(name = name):
                with self.assertRaises(ValueError):
                    server.get_log_path(name)

    def test_only_top_level_logs_are_listed(self):
        (self.log_dir / "run.jsonl").write_bytes(log_line(0))
        (self.log_dir / "note.txt").write_text("not a log")
        (self.log_dir / "sub").mkdir()
        (self.log_dir / "sub" / "hidden.jsonl").write_bytes(log_line(0))
        self.assertEqual([log["name"] for log in server.list_logs()] , ["run.jsonl"])


if __name__ == "__main__":
    unittest.main()
