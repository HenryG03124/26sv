import json
import time
from datetime import datetime
from pathlib import Path

class Logger():
    def __init__(self , log_dir = Path(__file__).resolve().parent / "logs"):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents = True , exist_ok = True)
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl"
        self.file = open(log_dir / filename , "w" , encoding = "utf-8" , buffering = 1)
        self.start_time = time.monotonic()
        self.frame_id = 0

    def write(self , timestamp , data):
        record = {
            "frame_id": self.frame_id ,
            "elapsed": timestamp - self.start_time ,
            "timestamp": timestamp ,
            "data": data ,
        }
        self.file.write(json.dumps(record , ensure_ascii = False , separators = ("," , ":")) + "\n")
        self.frame_id += 1

    def close(self):
        self.file.close()
