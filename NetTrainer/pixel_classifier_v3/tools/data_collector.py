"""Collect ETS2 RGB images and SCS SDK revision-12 telemetry on Windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime , timezone

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0 , str(PROJECT_ROOT))

from MyAutoPilot.scs_telemetry import SCSTelemetry

capture_fullscreen = False #True: full monitor , False: window client area
monitor_index = 1
window_name = "Euro Truck Simulator 2"

DEFAULT_OUTPUT = PROJECT_ROOT / "NetTrainer" / "datasets" / "ets2_steering"
SDK_URL = "https://github.com/truckermudgeon/scs-sdk-plugin"
SDK_COMMIT = "c8910d6e9a5ca0c2fa018942054263292cab19a4"
COLUMNS = (
    "timestamp" , "image" , "userSteer" , "gameSteer" , "speed" , "throttle" , "brake" ,
    "sdkTimestamp" , "captureDurationMs" , "telemetryAgeMs" ,
)


def valid_telemetry(value):
    if value.get("status") != "ok":
        return False
    for field , bounds in (
        ("user_steer" , (-1 , 1)) , ("game_steer" , (-1 , 1)) ,
        ("throttle" , (0 , 1)) , ("brake" , (0 , 1)) ,
        ("speed_mps" , (-math.inf , math.inf)) ,
    ):
        number = value.get(field)
        if not isinstance(number , (int , float)) or not math.isfinite(number):
            return False
        if not bounds[0] <= number <= bounds[1]:
            return False
    return True


class SessionWriter:
    """A portable CSV with relative image paths; never overwrite a session."""

    def __init__(self , output , session , image_format = "jpg" , metadata = None):
        if not session or Path(session).name != session or session in ("." , ".."):
            raise ValueError("Session must be a single directory name")
        if image_format not in ("jpg" , "png"):
            raise ValueError("image_format must be jpg or png")
        self.directory = Path(output).expanduser().resolve() / session
        self.directory.mkdir(parents = True , exist_ok = False)
        (self.directory / "images").mkdir()
        self.image_format = image_format
        self.count = 0
        details = {
            "schema_version": 1 , "sdk_repository": SDK_URL , "sdk_commit": SDK_COMMIT ,
            "sdk_revision": 12 , "timestamp_unit": "Unix nanoseconds (UTC, capture midpoint)" ,
            "sdkTimestamp_unit": "microseconds since SDK start, not Unix time" ,
            "speed_unit": "m/s (signed)" , "steering_range": [-1 , 1] ,
            "steering_positive": "left" , "throttle_brake": "user input, [0, 1]" ,
            "image_color": "RGB" , "image_format": image_format ,
            "synchronization": "nearest read before/after screen capture; not hardware synchronized" ,
            **(metadata or {}) ,
        }
        (self.directory / "metadata.json").write_text(
            json.dumps(details , indent = 2 , ensure_ascii = False) , encoding = "utf-8"
        )
        self.file = (self.directory / "samples.csv").open("x" , newline = "" , encoding = "utf-8")
        self.writer = csv.DictWriter(self.file , fieldnames = COLUMNS)
        self.writer.writeheader()
        self.file.flush()

    def write(self , image , telemetry , timestamp , duration_ms , age_ms):
        if not valid_telemetry(telemetry):
            raise ValueError("Refusing to write invalid telemetry")
        relative = Path("images") / f"{self.count:08d}.{self.image_format}"
        destination = self.directory / relative
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        image = image.convert("RGB")
        try:
            if self.image_format == "jpg":
                image.save(temporary , format = "JPEG" , quality = 95 , subsampling = 0)
            else:
                image.save(temporary , format = "PNG")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok = True)
        # The CSV only references completed image files, and flushes every row.
        self.writer.writerow({
            "timestamp": timestamp , "image": relative.as_posix() ,
            "userSteer": telemetry["user_steer"] , "gameSteer": telemetry["game_steer"] ,
            "speed": telemetry["speed_mps"] , "throttle": telemetry["throttle"] ,
            "brake": telemetry["brake"] , "sdkTimestamp": telemetry["sdk_timestamp_us"] ,
            "captureDurationMs": duration_ms , "telemetryAgeMs": age_ms ,
        })
        self.file.flush()
        self.count += 1

    def close(self):
        self.file.close()


class Collector:
    """Injectable capture/telemetry interfaces also allow offline verification."""

    def __init__(self , telemetry , capture , writer , size = (640 , 352) , crop = None , max_age_ms = 100):
        self.telemetry = telemetry
        self.capture = capture  # returns a PIL RGB image
        self.writer = writer
        self.size = size
        self.crop = crop
        self.max_age_ms = max_age_ms
        self.last_sdk_timestamp = None

    def sample(self):
        before = self.telemetry.read()
        before_time = time.monotonic_ns()
        if before.get("status") == "unsupported":
            raise RuntimeError(f"Unsupported SDK revision {before.get('revision')}; install revision 12")
        if not valid_telemetry(before):
            return before.get("status" , "invalid") if before.get("status") != "ok" else "invalid"
        # Require an advancing SDK clock, including at startup after a crash.
        if self.last_sdk_timestamp is None:
            self.last_sdk_timestamp = before["sdk_timestamp_us"]
            return "warming up"
        start = time.monotonic_ns()
        wall_start = time.time_ns()
        try:
            image = self.capture()
        except RuntimeError as error:
            return str(error)
        end = time.monotonic_ns()
        after = self.telemetry.read()
        after_time = time.monotonic_ns()
        if not valid_telemetry(after):
            return "telemetry unavailable after capture"
        if after["sdk_timestamp_us"] < before["sdk_timestamp_us"]:
            self.last_sdk_timestamp = None
            return "SDK clock reset"
        midpoint = (start + end) // 2
        # Read time + observed SDK age is an estimate, not a frame-sync guarantee.
        candidates = ((before , before_time) , (after , after_time))
        selected , age_ms = min(
            ((data , abs(read_time - midpoint) / 1e6 + data.get("age_s" , 0) * 1000)
             for data , read_time in candidates) , key = lambda item: item[1]
        )
        duration_ms = (end - start) / 1e6
        if duration_ms > self.max_age_ms or age_ms > self.max_age_ms:
            return "capture/telemetry too slow"
        sdk_time = selected["sdk_timestamp_us"]
        if sdk_time == self.last_sdk_timestamp:
            return "duplicate SDK frame"
        if self.crop:
            left , top , width , height = self.crop
            if left + width > image.width or top + height > image.height:
                raise ValueError("Crop exceeds capture area")
            image = image.crop((left , top , left + width , top + height))
        image = image.resize(self.size , Image.Resampling.BILINEAR)
        self.writer.write(image , selected , wall_start + (end - start) // 2 , duration_ms , age_ms)
        self.last_sdk_timestamp = sdk_time
        return "recording"


def build_parser():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--output" , type = Path , default = DEFAULT_OUTPUT)
    parser.add_argument("--session" , default = None)
    parser.add_argument("--fps" , type = float , default = 10)
    parser.add_argument("--duration" , type = float , default = 0 , help = "Wall seconds; 0 = until Ctrl+C")
    parser.add_argument("--max-samples" , type = int , default = 0 , help = "0 = unlimited")
    parser.add_argument("--window-title" , default = window_name)
    parser.add_argument("--crop" , type = int , nargs = 4 , metavar = ("LEFT" , "TOP" , "WIDTH" , "HEIGHT"))
    parser.add_argument("--image-format" , choices = ("jpg" , "png") , default = "jpg")
    parser.add_argument("--max-age-ms" , type = float , default = 100)
    return parser


def main(argv = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if not math.isfinite(args.duration) or args.duration < 0 or args.max_samples < 0:
        parser.error("--duration and --max-samples must be nonnegative")
    if not math.isfinite(args.max_age_ms) or args.max_age_ms <= 0:
        parser.error("--max-age-ms must be finite and positive")
    if args.crop and (min(args.crop[:2]) < 0 or min(args.crop[2:]) <= 0):
        parser.error("Crop requires nonnegative LEFT/TOP and positive WIDTH/HEIGHT")
    if os.name != "nt":
        parser.error("Live collection currently requires Windows")

    from ctypes import wintypes
    from MyAutoPilot.screen_capture import ScreenCapture

    capture = ScreenCapture(monitor_index)
    capture.user32.GetForegroundWindow.argtypes = []
    capture.user32.GetForegroundWindow.restype = wintypes.HWND
    telemetry = SCSTelemetry()
    writer = None
    try:
        session = args.session or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        writer = SessionWriter(args.output , session , args.image_format , {
            "requested_fps": args.fps , "image_size": [640 , 352] ,
            "capture_fullscreen": capture_fullscreen , "monitor_index": monitor_index ,
            "window_title": args.window_title , "crop": args.crop , "max_age_ms": args.max_age_ms ,
        })

        def grab_rgb():
            window = capture.user32.FindWindowW(None , args.window_title)
            if not window or window != capture.user32.GetForegroundWindow():
                raise RuntimeError("Waiting for the game window in the foreground")
            bgr = capture.capture_frame() if capture_fullscreen else capture.capture_window(args.window_title)
            if window != capture.user32.GetForegroundWindow():
                raise RuntimeError("Game lost focus during capture")
            return Image.fromarray(bgr[: , : , ::-1].copy())

        collector = Collector(telemetry , grab_rgb , writer , crop = args.crop , max_age_ms = args.max_age_ms)
        print(f"Dataset: {writer.directory}\nKeep the game visible. Ctrl+C stops and saves." , flush = True)
        started = deadline = time.monotonic()
        previous_status = None
        while not args.max_samples or writer.count < args.max_samples:
            if args.duration and time.monotonic() - started >= args.duration:
                break
            status = collector.sample()
            if status != previous_status or (status == "recording" and writer.count % 100 == 0):
                print(f"{status}; saved {writer.count} samples" , flush = True)
            previous_status = status
            deadline = max(deadline + 1 / args.fps , time.monotonic())
            time.sleep(max(0 , deadline - time.monotonic()))
    except KeyboardInterrupt:
        print("\nCollection stopped.")
    finally:
        telemetry.close()
        capture.close()
        if writer:
            writer.close()
            print(f"Saved {writer.count} samples: {writer.directory / 'samples.csv'}")


if __name__ == "__main__":
    main()
