const $ = (id) => document.getElementById(id);
const canvas = $("perception-canvas");
const context = canvas.getContext("2d");
const telemetryNames = {ok: "遥测正常", paused: "游戏暂停", missing: "未连接 SDK", stale: "遥测过期", inactive: "SDK 未激活", unsupported: "SDK 不兼容"};
const directionNames = {stationary: "静止", forward: "前进", backward: "后退", right: "向右", left: "向左", unknown: "未知"};
const classNames = {0: "行人", 1: "车辆", 2: "卡车"};
const brakeNames = {none: "无制动请求", ttc: "碰撞时间阈值", near: "近距离目标", emergency: "紧急制动", held: "保持制动"};
let source = "latest";
let state = null;
let logs = [];
let view = "geometry";
let paused = false;
let playing = false;
let nextPlayback = 0;
let loading = false;
let requestId = 0;
let cameraRequest = 0;
let cameraLoading = false;
let cameraUrl = null;
let cameraFresh = false;
let toastTimer;
let theme = "light";
try {
    if (localStorage.getItem("26sv-theme") === "dark") theme = "dark";
} catch {}

function setTheme(nextTheme) {
    theme = nextTheme;
    document.documentElement.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]').content = theme === "dark" ? "#0c1015" : "#ffffff";
    setText("theme-label", theme === "dark" ? "浅色" : "深色");
    $("theme-toggle").ariaLabel = theme === "dark" ? "切换到浅色主题" : "切换到深色主题";
    $("theme-toggle").title = $("theme-toggle").ariaLabel;
    drawGeometry();
}

function number(value, digits = 0) {
    return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function elapsed(value) {
    if (!Number.isFinite(value)) return "00:00.0";
    const minutes = Math.floor(value / 60).toString().padStart(2, "0");
    return `${minutes}:${(value % 60).toFixed(1).padStart(4, "0")}`;
}

function setText(id, text) {
    $(id).textContent = text;
}

function toast(message) {
    setText("toast", message);
    $("toast").hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $("toast").hidden = true, 3500);
}

async function getJson(url) {
    const response = await fetch(url, {cache: "no-store", signal: AbortSignal.timeout(10000)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    return result;
}

async function refreshLogs() {
    try {
        const result = await getJson("/api/logs");
        logs = result.logs;
        const selected = source;
        const options = [new Option("自动跟随最新日志", "latest")];
        for (const log of logs) {
            const name = log.name;
            const date = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/.exec(name);
            const label = date ? `${date[1]}.${date[2]}.${date[3]}  ${date[4]}:${date[5]}:${date[6]}` : name;
            options.push(new Option(`${label} · ${(log.size / 1048576).toFixed(1)} MB`, name));
        }
        $("log-select").replaceChildren(...options);
        if (selected !== "latest" && !logs.some((log) => log.name === selected)) {
            $("log-select").add(new Option(selected, selected));
        }
        $("log-select").value = selected;
    } catch (error) {
        if (!state) showConnectionError(error.message);
    }
}

function showConnectionError(message) {
    $("connection-status").className = "status-badge offline";
    $("connection-status").replaceChildren(document.createElement("i"), document.createTextNode("连接中断"));
    setText("data-state", "本地服务未连接");
    setText("last-update", "已保留最后一次读取结果");
    setText("playback-label", "服务连接中断");
    $("status-notice").querySelector("p").textContent = "请在 26sv 根目录运行 python -B -m UI.server，然后刷新页面。";
    if (!state?.record) showEmpty("本地服务未连接", message === "Failed to fetch" ? "启动 UI 服务后访问 http://127.0.0.1:8765。" : message);
}

async function loadState(index = 0) {
    const currentRequest = ++requestId;
    const requestedSource = source;
    loading = true;
    try {
        const result = await getJson(`/api/state?log=${encodeURIComponent(requestedSource)}&index=${index}`);
        if (currentRequest !== requestId) return;
        const previous = state?.record;
        state = result;
        render();
        if (source !== "latest" && playing) {
            const frameTime = result.record?.elapsed - previous?.elapsed;
            const delay = Number.isFinite(frameTime) && frameTime > 0 ? frameTime * 1000 : 80;
            nextPlayback = performance.now() + Math.max(16, delay / Number($("playback-speed").value));
            if (result.index >= result.count - 1) {
                playing = false;
                updatePlayButton();
            }
        }
    } catch (error) {
        if (currentRequest === requestId) {
            playing = false;
            updatePlayButton();
            showConnectionError(error.message);
        }
    } finally {
        if (currentRequest === requestId) loading = false;
    }
}

function updatePlayButton() {
    const active = source === "latest" ? !paused : playing;
    $("play-icon").setAttribute("href", active ? "#i-pause" : "#i-play");
    const label = source === "latest" ? (paused ? "继续显示" : "暂停显示") : (playing ? "暂停回放" : "播放回放");
    $("play-button").ariaLabel = label;
    $("play-button").title = label;
}

function render() {
    const record = state?.record;
    const data = record?.data || {};
    const telemetry = data.telemetry || {};
    const replay = source !== "latest";
    const status = replay ? "日志回放" : paused ? "显示已暂停" : state?.live ? "实时接入" : record ? "历史快照" : "等待数据";
    $("connection-status").className = `status-badge${!replay && !paused && state?.live ? " live" : ""}`;
    $("connection-status").replaceChildren(document.createElement("i"), document.createTextNode(status));
    setText("data-state", status);
    setText("last-update", state?.updated_at ? `日志更新于 ${new Date(state.updated_at * 1000).toLocaleString("zh-CN", {hour12: false})}` : "尚未发现运行日志");
    setText("source-detail", state?.source || "MyAutoPilot / logs");
    $("source-detail").title = state?.source || "";
    setText("footer-source", state?.source ? `${replay ? "REPLAY" : "LATEST"} / ${state.source}` : "本地只读数据 · 等待连接");
    setText("frame-label", record ? `FRAME ${String(record.frame_id).padStart(6, "0")}` : "FRAME —");
    setText("resolution-label", `${data.image_size?.[0] || 640} × ${data.image_size?.[1] || 352}`);
    setText("elapsed-label", elapsed(record?.elapsed));
    setText("playback-label", replay ? record ? `帧 ${(state.index || 0) + 1} / ${state.count || 0}` : "等待回放数据" : paused ? "显示暂停 · 原程序继续运行" : state?.live ? "跟随最新日志" : record ? "日志已停止更新 · 可选择历史回放" : "等待日志写入");
    $("timeline").disabled = !replay || !record;
    $("timeline").max = Math.max(0, (state?.count || 1) - 1);
    $("timeline").value = state?.index || 0;
    $("playback-speed").disabled = !replay;
    $("export-button").disabled = !record;
    updatePlayButton();

    const speed = telemetry.status === "ok" ? telemetry.speed_kmh : null;
    setText("speed-value", number(speed));
    setText("target-speed-value", number(data.target_speed));
    $("speed-arc").style.strokeDasharray = `${Number.isFinite(speed) ? Math.min(100, Math.max(0, speed / 120 * 100)) : 0} 100`;
    setText("telemetry-badge", telemetryNames[telemetry.status] || "等待遥测");
    $("telemetry-badge").className = `small-badge${telemetry.status === "ok" ? " good" : ""}`;
    setText("gear-value", telemetry.gear === 0 ? "N" : Number.isFinite(telemetry.gear) ? telemetry.gear < 0 ? `R${Math.abs(telemetry.gear)}` : `D${telemetry.gear}` : "—");
    $("rpm-value").replaceChildren(document.createTextNode(`${number(telemetry.engine_rpm)} `), unit("rpm"));
    $("fps-value").replaceChildren(document.createTextNode(`${number(data.fps, 1)} `), unit("fps"));
    setText("control-badge", data.control === "enable" ? "指令输出启用" : data.control ? "指令输出关闭" : "—");
    setText("steering-value", Number.isFinite(data.steering) ? `${data.steering > 0 ? "+" : ""}${data.steering.toFixed(3)}` : "—");
    $("steering-marker").style.left = `${50 + (Number.isFinite(data.steering) ? Math.max(-1, Math.min(1, data.steering)) * 50 : 0)}%`;
    $("steering-wheel").style.transform = `rotate(${Number.isFinite(data.steering) ? data.steering * 180 : 0}deg)`;
    for (const key of ["throttle", "brake"]) {
        setText(`${key}-value`, Number.isFinite(data[key]) ? `${Math.round(data[key] * 100)}%` : "—");
        $(`${key}-fill`).style.width = `${Number.isFinite(data[key]) ? Math.max(0, Math.min(1, data[key])) * 100 : 0}%`;
    }

    setText("priority-value", data.control_priority === "center" ? "车道居中" : ["lane change", "lane_change"].includes(data.control_priority) ? "lane change" : "—");
    for (const [id, key] of [["lane-left", "L"], ["lane-current", "current"], ["lane-right", "R"]]) {
        const available = data.lane_available?.[key];
        $(id).classList.toggle("available", available === true);
        $(id).querySelector("strong").textContent = available === true ? "可用" : available === false ? "不可用" : "未知";
    }
    for (const [id, key] of [["change-left", "L"], ["change-right", "R"]]) {
        const available = data.lane_change_available?.[key];
        setText(id, available === true ? "可用" : available === false ? "不可用" : "未知");
    }
    setText("path-count", Array.isArray(data.path_points) ? data.path_points.length : "—");
    setText("lane-status", {current: "当前帧有效", held: "保持历史车道", missing: "未识别"}[data.lane_status] || data.lane_status || "—");
    setText("brake-reason", brakeNames[data.brake_reason] || data.brake_reason || "—");
    setText("change-direction", {L: "向左", R: "向右"}[data.lane_change_direction] || (record ? "无" : "—"));
    setText("brake-target", data.brake_target_id != null ? `#${data.brake_target_id}` : "—");
    setText("ttc-value", Number.isFinite(data.ttc) ? `${data.ttc.toFixed(2)} s` : "—");
    renderTargets(data);
    let note = "读取本地运行日志，不向车辆发送控制指令。";
    if (replay) note = "正在回放已记录的运行数据。相机原图未保存在日志中。";
    else if (paused) note = "网页显示已暂停。原程序的运行与控制不受影响。";
    else if (record && !state.live) note = "日志已超过 3 秒未更新。当前为历史快照，不代表车辆此刻状态。";
    else if (telemetry.status && telemetry.status !== "ok") note = `${telemetryNames[telemetry.status] || telemetry.status}；车速显示为缺失值，感知日志仍单独更新。`;
    $("status-notice").querySelector("p").textContent = note;
    if (view === "geometry") {
        if (record) $("view-empty").hidden = true;
        else showEmpty("等待感知数据", "尚未发现有效日志。运行原有自动驾驶程序后，页面会自动读取结果。");
    }
    drawGeometry();
}

function unit(text) {
    const element = document.createElement("small");
    element.textContent = text;
    return element;
}

function trackedObjects(data) {
    return Object.entries(data.trackings || {}).filter(([, tracking]) => tracking?.status === "tracked");
}

function renderTargets(data) {
    const targets = trackedObjects(data);
    setText("target-count", targets.length);
    const rows = [];
    for (const [id, target] of targets) {
        const row = document.createElement("tr");
        for (const [index, value] of [
            `${classNames[target.label] || "目标"} #${id}`,
            Number.isFinite(target.score) ? `${(target.score * 100).toFixed(1)}%` : "—",
            directionNames[target.direction] || target.direction || "未知",
            "追踪中"
        ].entries()) {
            const cell = document.createElement("td");
            cell.textContent = value;
            if (index === 3) cell.className = "tracked-label";
            row.append(cell);
        }
        rows.push(row);
    }
    if (!rows.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 4;
        cell.className = "table-empty";
        cell.textContent = state?.record ? "当前帧无正在追踪的目标" : "等待目标数据";
        row.append(cell);
        rows.push(row);
    }
    $("target-list").replaceChildren(...rows);
}

function validPoints(points) {
    return Array.isArray(points) ? points.filter((point) => Array.isArray(point) && Number.isFinite(point[0]) && Number.isFinite(point[1])) : [];
}

function line(points, color, width, dashed = false) {
    if (points.length < 2) return;
    context.beginPath();
    context.moveTo(...points[0]);
    for (const point of points.slice(1)) context.lineTo(...point);
    context.strokeStyle = color;
    context.lineWidth = width;
    context.setLineDash(dashed ? [5, 4] : []);
    context.stroke();
    context.setLineDash([]);
}

function canvasLabel(text, x, y, color) {
    const scale = Math.max(1, canvas.width / 2 / (canvas.clientWidth || 640));
    context.font = `${9 * scale}px 'Segoe UI', 'Microsoft YaHei', sans-serif`;
    const width = context.measureText(text).width + 16 * scale;
    x = Math.max(4, Math.min(canvas.width / 2 - width - 4, x));
    y = Math.max(60, Math.min(canvas.height / 2 - 35, y));
    context.fillStyle = theme === "dark" ? "#101720eb" : "#ffffffeb";
    context.beginPath();
    context.roundRect(x, y - 13 * scale, width, 21 * scale, 6 * scale);
    context.fill();
    context.fillStyle = color;
    context.fillText(text, x + 8 * scale, y + scale);
}

function drawGeometry() {
    const data = state?.record?.data || {};
    const width = data.image_size?.[0] || 640;
    const height = data.image_size?.[1] || 352;
    const dark = theme === "dark";
    canvas.width = width * 2;
    canvas.height = height * 2;
    context.scale(2, 2);
    const gradient = context.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, dark ? "#111a24" : "#ffffff");
    gradient.addColorStop(.4, dark ? "#0e151e" : "#fdfdfd");
    gradient.addColorStop(1, dark ? "#0b1118" : "#f3f3f3");
    context.fillStyle = gradient;
    context.fillRect(0, 0, width, height);
    context.lineCap = "round";
    context.lineJoin = "round";

    if (dark) {
        context.strokeStyle = "#23303b";
        context.lineWidth = .5;
        context.beginPath();
        for (let x = 0; x <= width; x += 40) { context.moveTo(x, 0); context.lineTo(x, height); }
        for (let y = 0; y <= height; y += 40) { context.moveTo(0, y); context.lineTo(width, y); }
        context.stroke();
    }

    const cl = validPoints(data.CL_lane_points);
    const cr = validPoints(data.CR_lane_points);
    if ($("show-lanes").checked) {
        if (cl.length > 1 && cr.length > 1) {
            context.beginPath();
            context.moveTo(...cl[0]);
            for (const point of [...cl.slice(1), ...cr.slice().reverse()]) context.lineTo(...point);
            context.closePath();
            const road = context.createLinearGradient(0, height * .45, 0, height);
            road.addColorStop(0, dark ? "#b1e88a04" : "#ededed20");
            road.addColorStop(1, dark ? "#b1e88a18" : "#e4e4e4bb");
            context.fillStyle = road;
            context.fill();
        }
        for (const side of ["NL", "CL", "CR", "NR"]) {
            const points = validPoints(data[`${side}_lane_points`]);
            const neighbour = side.startsWith("N");
            const lane = context.createLinearGradient(0, height * .4, 0, height);
            lane.addColorStop(0, dark ? (neighbour ? "#6d8056" : "#a7c960") : (neighbour ? "#d4d4d4" : "#c4c4c4"));
            lane.addColorStop(1, dark ? (neighbour ? "#8ba763" : "#b1e88a") : (neighbour ? "#b7b7b7" : "#8b8b8b"));
            line(points, lane, neighbour ? 1.5 : 2.6, neighbour);
        }
    }
    if ($("show-path").checked) {
        const path = validPoints(data.path_points);
        line(path, dark ? "#61d8d10b" : "#3333330b", 18);
        line(path, dark ? "#61d8d120" : "#33333310", 10);
        const route = context.createLinearGradient(0, height * .4, 0, height);
        route.addColorStop(0, dark ? "#639da9" : "#aaaaaa");
        route.addColorStop(1, dark ? "#61d8d1" : "#525252");
        line(path, route, 2.6);
        if (path.length) {
            const end = path.reduce((a, b) => a[1] < b[1] ? a : b);
            canvasLabel("规划路径", end[0] - 26, end[1] - 26, dark ? "#61d8d1" : "#686868");
        }
    }
    if ($("show-objects").checked) {
        for (const [id, target] of trackedObjects(data)) {
            if (!Array.isArray(target.box) || target.box.length !== 4 || !target.box.every(Number.isFinite)) continue;
            const [x1, y1, x2, y2] = target.box;
            const boxWidth = x2 - x1;
            const boxHeight = y2 - y1;
            if (boxWidth <= 0 || boxHeight <= 0) continue;
            context.strokeStyle = dark ? "#e6af74" : "#a0a0a0";
            context.lineWidth = 1;
            context.strokeRect(x1, y1, boxWidth, boxHeight);
            canvasLabel(`${classNames[target.label] || "目标"} #${id}`, x1, y1 - 13, dark ? "#e6af74" : "#606060");
        }
    }
    if (state?.record && !cl.length && !cr.length && !trackedObjects(data).length && !validPoints(data.path_points).length) {
        context.fillStyle = dark ? "#a7b1bd" : "#888888";
        context.font = "11px 'Microsoft YaHei', sans-serif";
        context.textAlign = "center";
        context.fillText("当前帧没有可绘制的车道、路径或目标", width / 2, height / 2 + 25);
        context.textAlign = "left";
    }
}

function showEmpty(title, description) {
    $("view-empty").hidden = false;
    setText("empty-title", title);
    setText("empty-description", description);
}

function setView(nextView) {
    view = nextView;
    cameraRequest++;
    cameraFresh = false;
    $("geometry-tab").classList.toggle("selected", view === "geometry");
    $("camera-tab").classList.toggle("selected", view === "camera");
    $("geometry-tab").ariaPressed = String(view === "geometry");
    $("camera-tab").ariaPressed = String(view === "camera");
    canvas.hidden = view !== "geometry";
    $("camera-image").hidden = true;
    $("overlay-bar").hidden = view !== "geometry";
    $("viewport-tag").replaceChildren(document.createElement("i"), document.createTextNode(view === "geometry" ? "LOG GEOMETRY" : "RESULT WINDOW"));
    setText("view-caption", view === "geometry" ? "感知示意 · 原始图像坐标" : "窗口画面与日志分别采样 · 非逐帧同步");
    if (view === "geometry") {
        $("view-empty").hidden = Boolean(state?.record);
        if (!state?.record) showEmpty("等待感知数据", "运行原有自动驾驶程序后，这里会显示日志中的感知结果。");
        drawGeometry();
    } else {
        showEmpty("正在连接 Result 窗口", "请保持原程序的 Result 窗口可见且未被遮挡。");
        refreshCamera();
    }
}

async function refreshCamera() {
    if (view !== "camera" || cameraLoading || document.hidden) return;
    if (source !== "latest") {
        $("camera-image").hidden = true;
        showEmpty("历史日志不包含相机画面", "切换到“感知结构”查看此帧的车道、规划路径和检测框。");
        return;
    }
    if (paused) {
        if (!cameraFresh) showEmpty("画面显示已暂停", "点击下方播放按钮继续读取 Result 窗口。");
        return;
    }
    const currentRequest = ++cameraRequest;
    cameraLoading = true;
    try {
        const response = await fetch("/api/frame", {cache: "no-store", signal: AbortSignal.timeout(4000)});
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error);
        }
        const blob = await response.blob();
        if (currentRequest !== cameraRequest || view !== "camera" || source !== "latest" || paused) return;
        const oldUrl = cameraUrl;
        cameraUrl = URL.createObjectURL(blob);
        const image = $("camera-image");
        image.src = cameraUrl;
        await image.decode();
        if (oldUrl) URL.revokeObjectURL(oldUrl);
        if (currentRequest !== cameraRequest) return;
        image.hidden = false;
        cameraFresh = true;
        $("view-empty").hidden = true;
    } catch (error) {
        if (currentRequest === cameraRequest) {
            cameraFresh = false;
            $("camera-image").hidden = true;
            showEmpty("暂未获取到实时画面", `${error.message || "窗口读取失败"}。可切换到感知结构查看日志。`);
        }
    } finally {
        cameraLoading = false;
    }
}

async function selectSource(nextSource, index = 0) {
    source = nextSource;
    requestId++;
    cameraRequest++;
    paused = false;
    playing = false;
    state = null;
    loading = false;
    $("log-select").value = source;
    $("nav-monitor").classList.toggle("active", source === "latest");
    $("nav-replay").classList.toggle("active", source !== "latest");
    $("nav-monitor").ariaPressed = String(source === "latest");
    $("nav-replay").ariaPressed = String(source !== "latest");
    render();
    setView("geometry");
    $("export-button").disabled = true;
    setText("playback-label", source === "latest" ? "读取最新日志…" : "正在索引日志…");
    showEmpty(source === "latest" ? "读取最新日志" : "正在索引运行日志", "首次读取需要片刻，完成后显示此数据源的感知结果。");
    await loadState(index);
}

function togglePlayback() {
    if (source === "latest") {
        paused = !paused;
        cameraRequest++;
        requestId++;
        loading = false;
        if (paused && state) render();
        if (!paused) { loadState(); refreshCamera(); }
    } else {
        if (!state?.record) return;
        playing = !playing;
        if (!playing) {
            requestId++;
            loading = false;
        }
        if (playing && state.index >= state.count - 1) loadState(0);
        nextPlayback = performance.now();
    }
    updatePlayButton();
}

$("log-select").addEventListener("change", (event) => selectSource(event.target.value));
$("theme-toggle").addEventListener("click", () => {
    setTheme(theme === "dark" ? "light" : "dark");
    try { localStorage.setItem("26sv-theme", theme); } catch {}
});
window.addEventListener("resize", drawGeometry);
$("nav-monitor").addEventListener("click", () => selectSource("latest"));
$("nav-replay").addEventListener("click", () => {
    if (!logs.length) { toast("还没有可回放的日志"); return; }
    selectSource(logs[0].name);
});
$("geometry-tab").addEventListener("click", () => setView("geometry"));
$("camera-tab").addEventListener("click", () => setView("camera"));
for (const id of ["show-lanes", "show-path", "show-objects"]) $(id).addEventListener("change", drawGeometry);
$("play-button").addEventListener("click", togglePlayback);
$("timeline").addEventListener("input", (event) => {
    playing = false;
    updatePlayButton();
    loadState(Number(event.target.value));
});
$("fullscreen-button").addEventListener("click", async () => {
    try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await $("perception-panel").requestFullscreen();
    } catch { toast("当前浏览器不支持全屏显示"); }
});
$("help-button").addEventListener("click", () => $("help-dialog").showModal());
$("close-help").addEventListener("click", () => $("help-dialog").close());
$("got-it").addEventListener("click", () => $("help-dialog").close());
$("help-dialog").addEventListener("click", (event) => {
    if (event.target === $("help-dialog")) {
        const bounds = $("help-dialog").getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) $("help-dialog").close();
    }
});
$("export-button").addEventListener("click", () => {
    if (!state?.record) return;
    const blob = new Blob([JSON.stringify({source: state.source, mode: state.mode, record: state.record}, null, 2)], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `26sv-frame-${state.record.frame_id}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast("已导出当前帧数据");
});

let nextLive = 0;
let nextCamera = 0;
setInterval(() => {
    if (document.hidden) return;
    const now = performance.now();
    if (source === "latest" && !paused && !loading && now >= nextLive) {
        nextLive = now + 350;
        loadState();
    } else if (source !== "latest" && playing && !loading && now >= nextPlayback) {
        if (state && state.index < state.count - 1) loadState(state.index + 1);
    }
    if (view === "camera" && now >= nextCamera) {
        nextCamera = now + (cameraFresh ? 250 : 1500);
        refreshCamera();
    }
}, 40);
setInterval(refreshLogs, 10000);
setTheme(theme);
refreshLogs().then(async () => {
    const parameters = new URLSearchParams(location.search);
    const replay = parameters.get("replay");
    if (replay) {
        const name = replay === "latest" ? logs[0]?.name : replay;
        if (!name) { await loadState(); return; }
        const frame = Number(parameters.get("frame") || 0);
        await selectSource(name, Number.isInteger(frame) && frame >= 0 ? frame : 0);
        if (parameters.get("play") === "1" && state?.record) togglePlayback();
    } else {
        await loadState();
    }
});
