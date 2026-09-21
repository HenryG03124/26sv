# 2026-09-21 实验复查与验证

后续变更：按用户要求，主循环的分阶段计时及 `timings_ms` / `timings_frame_id` 日志字段已撤回，仅保留 FPS 计算和业务时间戳。下文关于分阶段日志的说明属于当时实现记录，不再代表当前 main；独立性能测试脚本不参与正常运行。

## 输入与对齐

- 录屏：`屏幕录制 2026-09-21 210708.mp4`，81.73 秒，30 FPS。
- 日志：`MyAutoPilot/logs/20260921_210518_798790.jsonl`，819 帧，119.56 秒。
- 用户确认：40～50 km/h，无人工接管，游戏转向设置未改。
- 通过录屏中终端的完整 steering 数值匹配日志：视频 10 秒对应日志 frame 155 / elapsed 23.674；视频 30 秒对应 frame 299 / elapsed 43.626；视频 50 秒对应 frame 428 / elapsed 63.671。视频时间约等于日志 elapsed 减 13.67 秒。
- 整份日志包含菜单阶段，不能把全部车道丢失都算成行驶失败。elapsed 15～100 秒内有 570 帧，其中 460 帧有中线，110 帧无中线。

## 性能

日志相邻 timestamp 的中位间隔为 141.29 ms（约 7.08 FPS）。旧日志没有分阶段计时，不能精确追溯游戏当时每一阶段的耗时。

只读实测（未运行游戏、未发送手柄输入）：

- 当前采集尺寸为 3840×2160。单独测现有 MSS 采集平均 61.59 ms。
- 原线程设置为 PyTorch 24 / OpenCV 32。完整只读采集链路平均 96.96 ms，其中 capture 69.57 ms，classifier 10.21 ms，detector predict 5.42 ms，draw/resize 10.97 ms。
- 调整为各 4 线程后，同类链路平均 81.30 ms，其中 capture 59.02 ms，classifier 9.21 ms，detector predict 4.71 ms，draw/resize 7.99 ms。
- 固定录屏帧、不含采集的对照：原线程 24.45 ms，4 线程 16.76 ms；refiner 约 0.75～1.47 ms，写日志约 0.18～0.24 ms。
- 上述完整链路测试不含 `imshow/waitKey`、终端输出、真实手柄调用。原 `steering_controller` 还有固定 10 ms sleep。桌面内容、缓存和 GPU 状态存在波动，不能把测量差值全部归因于线程设置。
- 录屏显示内存占用约 92%，可能加剧竞争，但没有当时的分页计数，不能断言发生了内存换页。

结论：已证实 4K 全屏 GDI 采集是主要耗时项之一，不能只归因于模型或 refiner。按用户后续要求，已撤回工作线程数调整和关闭逐帧终端输出的改动，测试脚本也不再修改线程数；上述对照数据作为历史测量保留。保留原采集实现、FPS 显示和真实主循环计时，优化方式另行讨论。

## 转向问题和改动

日志 elapsed 52.100～52.791 秒（视频约 38.43～39.12 秒）：

- 近端带状中位 x：313、299、329、319、304。
- 远端 x：306、307、303、304、301。
- 原 steering：+0.0276、-0.0258、+0.0332、+0.0658、-0.0088。

近端变化率被旧控制器放大为左右打舵；车辆速度未参与增益；偏差较大时远端预瞄消失，容易纠偏过头。丢线时原主循环还按“五帧”回正，7 FPS 和 30 FPS 的行为不同。

本次使用速度相关的图像空间预瞄，不将像素冒充米数，也不称为标定后的 pure pursuit：

- 45 km/h 时近端权重 0.35、远端权重 0.65，偏差较大时仍保留远端参考。
- 偏差增益在该速度下约 0.288（原来 0.6）。
- 变化率补偿上限约 0.0128（原来最高 0.07），并对目标偏差做 0.1 秒时间常数平滑。
- 转向目标上限约 0.108（原来 0.18），随速度变化；保留输出速率限制。
- 使用采集 timestamp 计算变化率；丢线也经过同一个按时间回正的控制器。
- SDK 缺失或失效时不假装读到 0 km/h，以 45 km/h 的保守默认参数计算，显示车速不可用。

同一日志 elapsed 40～80 秒的离线回放，固定假设 45 km/h：

| 指标 | 原记录 | 新回放 |
| --- | ---: | ---: |
| steering RMS | 0.0748 | 0.0402 |
| 绝对峰值 | 0.1800 | 0.1078 |
| 总变化量 sum(abs(diff)) | 8.3474 | 3.9628 |
| 超过 0.015 幅值的方向切换 | 40 | 9 |

这仅验证同一批视觉输入下输出更稳，不代表新的车辆轨迹：修改控制后，后续画面也会改变。仍需实车游戏闭环测试。中线缺失或边界选错也不能靠 steering 凭空恢复；本次未修改 refiner、classifier、road_center 或刹车。

## SDK 与日志

使用用户提供的 [truckermudgeon/scs-sdk-plugin](https://github.com/truckermudgeon/scs-sdk-plugin)，协议 revision 12，源码版本 `c8910d6e9a5ca0c2fa018942054263292cab19a4`。

共享内存：`Local\SCSTelemetry`。仅 `OpenFileMappingW` + `MapViewOfFile(FILE_MAP_READ)`，不会创建假 SDK 内存，不需要增加 Python 依赖。不支持的协议版本不猜测解析。

新增 `telemetry` 日志对象包括状态、SDK 时间戳、速度、挡位、RPM、输入和实际转向、油门/刹车、世界坐标、车身朝向、局部速度、横摆角速度及各车轮转角。

- SDK speed 为 m/s，乘 3.6 得 km/h，倒车为负。
- SDK userSteer/gameSteer 正值为左；另存 `*_steer_right` 转换为本项目右正方向。gameSteer 是归一化控制量，不是车轮角。
- SDK angular velocity 为转/秒，乘 2π 得 rad/s；`yaw_rate_ccw_rad_s` 保留逆时针为正。
- wheelSteering 为转，乘 360 得度。
- `telemetry.read_timestamp` 是当前命令发送之前的观测时间；`control_timestamp` 是本轮控制调用完成时间。不要把同一行 SDK 转向当作当前命令的即时响应。
- `status` 为 `missing`、`inactive`、`paused`、`stale`、`unsupported` 或 `ok`；非活动数据不作为实时车速使用。
- `steering_debug` 保存近远端 x、预瞄权重、滤波偏差、变化率项、原始转向目标等。
- FPS 是包含采集、推理、控制、日志、显示的循环帧率，按约 1 秒时间常数平滑，不是模型推理 FPS。
- `timings_ms` 是完整上一帧的耗时，`timings_frame_id` 明确指向它。这样可以包含日志写入和显示本身，第一帧为 null；退出的最后一帧没有后续记录承载其耗时。

本次没有活动游戏 SDK 映射，只读检查返回 `missing`。二进制解析、缺失/过期处理和主循环接线已测试，实时 SDK 数值及车辆闭环效果尚未验证。

## 验证命令

在项目根目录执行：

```powershell
python -B -m unittest discover -s MyAutoPilot/tests
python -B -m MyAutoPilot.tests.replay_steering MyAutoPilot/logs/20260921_210518_798790.jsonl --start 40 --end 80 --speed 45
python -B -m MyAutoPilot.tests.profile_pipeline --capture --frames 40
```

最后一个命令只采集和计算，不发送游戏输入、不打开窗口。测试包含模拟主循环、SDK 布局/单位/过期、显示文本、记录回放相关场景，以及一个未标定的简化自行车模型反馈检查；它不是 ETS2 驾驶效果保证。

下一轮先在相同设置下短测：确认右上角车速与游戏一致，再看日志 `timings_ms` 的实际瓶颈，和 `user_steer_right` / `game_steer_right` 对上一轮命令的响应。若 SDK 为 missing，应先确认插件被游戏加载，不要用猜测车速继续调参。
