import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler , ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs , urlsplit


ui_dir = Path(__file__).resolve().parent
log_dir = ui_dir.parent / "MyAutoPilot" / "logs"
host = "127.0.0.1"
port = int(os.environ.get("UI_PORT" , "8765"))
stale_interval = 3.0
tail_size = 1024 * 1024
replay_cache = {}
replay_lock = threading.Lock()

static_files = {
    "/": ("index.html" , "text/html; charset=utf-8") ,
    "/index.html": ("index.html" , "text/html; charset=utf-8") ,
    "/style.css": ("style.css" , "text/css; charset=utf-8") ,
    "/app.js": ("app.js" , "text/javascript; charset=utf-8") ,
    "/favicon.svg": ("favicon.svg" , "image/svg+xml") ,
}


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def read_record(line):
    try:
        record = json.loads(line , parse_float = finite_float , parse_constant = lambda value: None)
    except (ValueError , UnicodeDecodeError):
        return None
    if not isinstance(record , dict) or not isinstance(record.get("data") , dict):
        return None
    return record


def list_logs():
    logs = []
    for path in log_dir.glob("*.jsonl"):
        if path.resolve().parent != log_dir.resolve() or not path.is_file():
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        logs.append({"name": path.name , "size": stat.st_size , "modified": stat.st_mtime})
    logs.sort(key = lambda log: (log["modified"] , log["name"]) , reverse = True)
    return logs


def get_log_path(name):
    if Path(name).name != name or not name.endswith(".jsonl"):
        raise ValueError("无效的日志名称")
    path = log_dir / name
    if path.resolve().parent != log_dir.resolve():
        raise ValueError("无效的日志名称")
    if not path.is_file():
        raise FileNotFoundError("日志不存在")
    return path


def latest_record(path):
    with path.open("rb") as file:
        file.seek(0 , 2)
        size = file.tell()
        start = max(0 , size - tail_size)
        file.seek(start)
        lines = file.read(size - start).splitlines(keepends = True)
    if start:
        lines = lines[1:]
    for line in reversed(lines):
        if not line.endswith(b"\n"):
            continue
        record = read_record(line)
        if record is not None:
            return record
    return None


def replay_record(path , index):
    with replay_lock:
        stat = path.stat()
        cache = replay_cache.get(path.name)
        if cache is None or stat.st_size < cache["size"] or (
            stat.st_size == cache["size"] and stat.st_mtime_ns != cache["modified"]
        ):
            cache = {"size": 0 , "modified": 0 , "position": 0 , "offsets": []}
        if stat.st_size != cache["size"] or stat.st_mtime_ns != cache["modified"]:
            with path.open("rb") as file:
                file.seek(cache["position"])
                while file.tell() < stat.st_size:
                    position = file.tell()
                    line = file.readline(stat.st_size - position)
                    if not line.endswith(b"\n"):
                        break
                    if read_record(line) is not None:
                        cache["offsets"].append(position)
                    cache["position"] = file.tell()
            cache["size"] = stat.st_size
            cache["modified"] = stat.st_mtime_ns
        replay_cache.pop(path.name , None)
        replay_cache[path.name] = cache
        while len(replay_cache) > 3:
            replay_cache.pop(next(iter(replay_cache)))
        count = len(cache["offsets"])
        if not count:
            return None , 0 , 0
        index = min(max(0 , index) , count - 1)
        with path.open("rb") as file:
            file.seek(cache["offsets"][index])
            record = read_record(file.readline())
        return record , index , count


def get_state(name , index = 0):
    state = {
        "mode": "live" if name == "latest" else "replay" ,
        "source": None ,
        "updated_at": None ,
        "age_s": None ,
        "live": False ,
        "record": None ,
        "index": None ,
        "count": None ,
        "error": None ,
    }
    if name == "latest":
        logs = list_logs()
        if not logs:
            state["error"] = "尚无运行日志"
            return state
        path = get_log_path(logs[0]["name"])
        state["record"] = latest_record(path)
    else:
        path = get_log_path(name)
        state["record"] , state["index"] , state["count"] = replay_record(path , index)
    stat = path.stat()
    state["source"] = path.name
    state["updated_at"] = stat.st_mtime
    state["age_s"] = max(0.0 , time.time() - stat.st_mtime)
    state["live"] = state["mode"] == "live" and state["age_s"] <= stale_interval and state["record"] is not None
    if state["record"] is None:
        state["error"] = "尚无完整日志帧"
    return state


def capture_result():
    if os.name != "nt":
        raise RuntimeError("Result 窗口画面仅支持 Windows")
    try:
        import cv2
        from MyAutoPilot.screen_capture import ScreenCapture
    except ImportError as error:
        raise RuntimeError("画面采集依赖不可用，请安装项目 requirements.txt 中的依赖") from error
    capturer = None
    try:
        capturer = ScreenCapture(0)
        frame = capturer.capture_window("Result")
        if frame.shape[1] > 1280:
            height = round(frame.shape[0] * 1280 / frame.shape[1])
            frame = cv2.resize(frame , (1280 , height))
        success , image = cv2.imencode(".jpg" , frame , [cv2.IMWRITE_JPEG_QUALITY , 82])
        if not success:
            raise RuntimeError("Result 画面编码失败")
        return image.tobytes()
    except RuntimeError as error:
        if "Window not found:" in str(error):
            raise RuntimeError("等待 Result 窗口，请先运行原有自动驾驶程序") from error
        if "Window is minimized:" in str(error):
            raise RuntimeError("Result 窗口已最小化，请恢复窗口") from error
        if "Window has no capture area:" in str(error):
            raise RuntimeError("Result 窗口暂时没有可用画面") from error
        raise
    finally:
        if capturer is not None:
            capturer.close()


class UIHandler(BaseHTTPRequestHandler):
    def send_bytes(self , content , content_type , status = 200):
        self.send_response(status)
        self.send_header("Content-Type" , content_type)
        self.send_header("Content-Length" , str(len(content)))
        self.send_header("Cache-Control" , "no-store")
        self.send_header("X-Content-Type-Options" , "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self , data , status = 200):
        content = json.dumps(data , ensure_ascii = False , allow_nan = False).encode("utf-8")
        self.send_bytes(content , "application/json; charset=utf-8" , status)

    def do_GET(self):
        request = urlsplit(self.path)
        query = parse_qs(request.query)
        try:
            if request.path == "/api/logs":
                logs = list_logs()
                self.send_json({"logs": logs , "latest": logs[0]["name"] if logs else None})
            elif request.path == "/api/state":
                name = query.get("log" , ["latest"])[0]
                try:
                    index = int(query.get("index" , ["0"])[0])
                except ValueError:
                    raise ValueError("无效的回放帧序号")
                self.send_json(get_state(name , index))
            elif request.path == "/api/frame":
                try:
                    image = capture_result()
                except Exception as error:
                    message = str(error) if isinstance(error , RuntimeError) else "Result 画面暂时不可用，请检查窗口是否可见"
                    self.send_json({"error": message} , 503)
                    return
                self.send_bytes(image , "image/jpeg")
            elif request.path in static_files:
                filename , content_type = static_files[request.path]
                self.send_bytes((ui_dir / filename).read_bytes() , content_type)
            else:
                self.send_json({"error": "页面或接口不存在"} , 404)
        except (BrokenPipeError , ConnectionResetError , ConnectionAbortedError):
            return
        except FileNotFoundError:
            self.send_json({"error": "文件不存在"} , 404)
        except ValueError as error:
            self.send_json({"error": str(error)} , 400)
        except OSError:
            self.send_json({"error": "文件暂时无法读取"} , 503)

    def log_message(self , format , *args):
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer((host , port) , UIHandler)
    print("26sv UI:" , f"http://{host}:{port}" , flush = True)
    print("log directory:" , log_dir , flush = True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
