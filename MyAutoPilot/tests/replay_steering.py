"""Replay recorded perception, not a closed-loop driving test. No gamepad access."""
import argparse
import json
from pathlib import Path

import numpy as np

from MyAutoPilot.steering import Steering


def replay(records , speed_kmh = 45):
    Steering.reset()
    previous = 0
    output = []
    for record in records:
        data = record["data"]
        speed = data.get("telemetry" , {}).get("speed_kmh" , speed_kmh)
        previous = Steering.steering(data["center_points"] , data["y_far"] , 3 , previous ,
                                     speed_kmh = speed , timestamp = record["timestamp"] , y_near = data["y_near"])
        output.append(previous)
    return np.asarray(output)


def metrics(values):
    # Count meaningful lock reversals, not numerical zero crossings.
    signs = np.sign(values[np.abs(values) >= 0.015])
    return {"rms": float(np.sqrt(np.mean(values ** 2))) , "peak": float(np.max(np.abs(values))) ,
            "total_variation": float(np.sum(np.abs(np.diff(values)))) ,
            "reversals_above_0.015": int(np.sum(signs[1:] != signs[:-1]))}


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("log" , type = Path)
    parser.add_argument("--speed" , type = float , default = 45)
    parser.add_argument("--start" , type = float , default = 0)
    parser.add_argument("--end" , type = float , default = float("inf"))
    args = parser.parse_args()
    records = [json.loads(line) for line in args.log.read_text(encoding = "utf-8").splitlines()]
    current = replay(records , args.speed)
    selected = np.array([args.start <= record["elapsed"] <= args.end for record in records])
    recorded = np.array([record["data"]["steering"] for record in records])
    print(json.dumps({"recorded": metrics(recorded[selected]) , "new_replay": metrics(current[selected])} , indent = 2))


if __name__ == "__main__":
    main()
