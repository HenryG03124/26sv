# Pixel Classifier V3：ETS2 采集与转向学习

V3 保留 road/car、lane 两个分割头，并增加 steering **回归头**。网络只接收 RGB 图像，输出一个归一化转向值；speed、throttle、brake 保存在数据集中，当前不作为网络输入。

## 1. 安装遥测插件

采集器复用项目中的 `MyAutoPilot/scs_telemetry.py`，读取 Windows 的 `Local\SCSTelemetry` 共享内存。协议来自 [truckermudgeon/scs-sdk-plugin](https://github.com/truckermudgeon/scs-sdk-plugin)（RenCloud 项目的分支，基于 SCS 官方 Telemetry SDK）。

- 安装该项目 [Releases](https://github.com/truckermudgeon/scs-sdk-plugin/releases) 中 Windows x64 的插件 DLL 到 `Euro Truck Simulator 2/bin/win_x64/plugins/`，重启游戏并接受 SDK 插件提示。
- 读取器支持 **共享内存 revision 12**；遇到其他版本会报错，不能直接替换成不同布局的 telemetry 插件。
- 字段布局核对版本：[`c8910d6` 的 scs-telemetry-common.hpp](https://github.com/truckermudgeon/scs-sdk-plugin/blob/c8910d6e9a5ca0c2fa018942054263292cab19a4/scs-telemetry/inc/scs-telemetry-common.hpp)。[SCS 转向通道定义](https://github.com/truckermudgeon/scs-sdk-plugin/blob/c8910d6e9a5ca0c2fa018942054263292cab19a4/scs_sdk/include/common/scssdk_telemetry_truck_common_channels.h) 指定正值为逆时针，即向左。
- Python 依赖使用项目根目录的 `requirements.txt`，没有新增依赖。

以下命令均在项目根目录运行。训练入口沿用 v2 的顶层脚本结构，使用 `python -m` 启动。

## 2. 采集

```powershell
python NetTrainer/pixel_classifier_v3/tools/data_collector.py --fps 10
```

启动后切换到游戏，保持驾驶视角、游戏窗口在前台且画面无遮挡。按终端中的 `Ctrl+C` 停止。每次运行自动创建独立会话，不会覆盖旧数据；也可用 `--session drive_01` 命名。

和 `MyAutoPilot/main.py` 一样，在 `tools/data_collector.py` 顶部选择截图方式：

```python
capture_fullscreen = False #True: full monitor , False: window client area
monitor_index = 1
window_name = "Euro Truck Simulator 2"
```

`False` 截取游戏窗口客户区；`True` 截取 `monitor_index` 对应的整个显示器，1 表示第一个显示器。两种模式都要求游戏在前台，全屏采集时将游戏放在所选显示器上。`--crop` 相对于所选截图区域，`metadata.json` 会记录采集模式和显示器编号。

```text
NetTrainer/datasets/ets2_steering/20260924T123456_123456Z/
  metadata.json
  samples.csv
  images/00000000.jpg
  images/00000001.jpg
```

`samples.csv` 的前七列即所需数据，图片通过相对路径引用，整个会话可直接搬移：

| 列 | 定义 |
| --- | --- |
| timestamp | 截图期间中点的 UTC Unix 纳秒时间戳，整数；不是游戏时间 |
| image | 相对 samples.csv 的 RGB 图片路径，默认 JPEG quality=95、640×352 |
| userSteer | 玩家输入转向，[-1, 1]，正值向左 |
| gameSteer | 游戏应用平滑/转向限制后的实际转向输入，[-1, 1]，正值向左；不是车轮角度 |
| speed | SDK 有符号速度，m/s；乘 3.6 为 km/h |
| throttle | 玩家油门输入，[0, 1] |
| brake | 玩家制动输入，[0, 1] |
| sdkTimestamp | SDK 运行时间，微秒，用于识别重复帧 |
| captureDurationMs | 截图耗时，毫秒 |
| telemetryAgeMs | 所选遥测读取时刻距截图中点的距离，加上已观察到的遥测停滞时间，毫秒 |

截图前后各读取一次遥测，选择估计更接近截图中点的一次。SDK 与屏幕抓取没有原子帧同步，此方法属于近似配对；`telemetryAgeMs` 不包含游戏渲染管线延迟。暂停、缺失、过期、重复遥测、非法数值及失焦画面不会写入 CSV。默认拒绝截图耗时或遥测年龄大于 100 ms 的样本。

可选参数：

```powershell
# 录制 5 分钟，使用无损 PNG
python NetTrainer/pixel_classifier_v3/tools/data_collector.py --duration 300 --image-format png

# 裁剪所选截图区域：左、上、宽、高（原始像素），再缩放到 640×352
python NetTrainer/pixel_classifier_v3/tools/data_collector.py --crop 0 100 1920 900
```

还支持 `--output`、`--window-title`、`--max-samples`、`--max-age-ms`。统一各次采集的视角和裁剪方式；推理时提供相同视野的图像。若画面中可见方向盘、转向指示器等标签线索，可裁去这些区域，避免模型依赖这些线索。建议采集多段独立驾驶会话，覆盖左右转弯和直行。

## 3. 训练

`train.py` 直接以 v2 的训练脚本为底稿，保留顶部数据加载、loader、损失函数、模型和优化器、训练循环、最终验证与绘图的顺序。参数直接在代码里修改，不使用命令行参数层：

```powershell
python -m NetTrainer.pixel_classifier_v3.train
```

- `batch_size = 16`；训练轮数在 `for epoch in range(30)` 修改。
- 学习率在 `torch.optim.Adam(model.parameters() , lr = 0.001)` 修改。
- `steering_target = "userSteer"`，如需学习游戏平滑后的转向，改为 `"gameSteer"`。
- 数据路径在脚本开头的 Dataset 初始化语句中修改。

默认同时训练 road/car、lane 和 steering 三个任务，需要以下数据：

- road/car：`NetTrainer/datasets/processed_pc_ds/{train,val}_pairs.csv`
- lane：`NetTrainer/datasets/processed_pc_lane_ds_bdd100k/{train,val}_pairs.csv`
- steering：递归读取 `NetTrainer/datasets/ets2_steering/**/samples.csv`

road 和 lane 保留 v2 的 `train_task` / `evaluate_task`；steering 使用 `train_steering` / `evaluate_steering` 和 SmoothL1 损失。三个任务按 batch 进度交替更新同一个网络，各自使用对应数据，不需要为同一图片同时准备三种标签。

训练期间记录各分割头的 IoU、steering 的 MAE/RMSE；全部训练结束后像 v2 一样统一验证和绘图。steering 误差单位为归一化 SDK 转向值，不是角度。两种标签均保持正值向左；对接正值向右的控制接口时，需要在控制侧取负号。

steering 默认按会话划分训练集/验证集，种子为 42，约 20% 会话留作验证。只有一个会话时，最后 20% 帧作为验证集，在分界处丢弃至少 1 秒数据；短会话无法划分时需要继续采集。不同驾驶会话更适合检验泛化。

输出到 v3 目录：

- `pc_model_v3.pth`：模型参数、转向标签/符号、输入大小及任务信息，供现有 test.py 读取。
- `figure_v3.png`：训练 loss、分割 mIoU、steering MAE 曲线与最终验证值。
- `pc_model_v3.split.json`：训练/验证图片的绝对路径，用于重现留出集测试。搬移数据后需要重新生成。

每轮后保存模型；中断处理沿用 v2，保存当前权重并进行最终验证。之前助手添加的 `--tasks`、`--epochs`、`--init-checkpoint` 等训练参数已移除。

## 4. 测试

```powershell
python NetTrainer/pixel_classifier_v3/test.py --image NetTrainer/datasets/ets2_steering/drive_01/images/00000000.jpg
```

默认读取 `pc_model_v3.pth`，在 `predictions/` 输出 `*_prediction_v3.json`，包括 `steering`、`steering_target` 和 `steering_positive`。若 checkpoint 具有训练过的分割头，还会输出 `*_mask_v3.png`：背景黑、道路绿、车辆红、车道线黄。只训练 steering 的模型不会输出随机初始化的分割结果。

测试训练时留出的 steering 验证集：

```powershell
python NetTrainer/pixel_classifier_v3/test.py --csv NetTrainer/datasets/ets2_steering --split-file NetTrainer/pixel_classifier_v3/pc_model_v3.split.json
```

输出 `steering_predictions.csv`（真实值、预测值、绝对误差）和 MAE/RMSE。不提供 `--split-file` 时评估传入的全部数据；评估独立新会话时可直接传入其 CSV。标签自动从 checkpoint 读取，避免误把 userSteer 模型按 gameSteer 评分。

代码调用：

```python
from Models.pc_v3 import UNet

model = UNet()
# images: [B, 3, 352, 640], float32 RGB / 255
steering = model(images , task = "steering") #[B , 1], [-1, 1]
# model(images) 仍返回共享特征；model.head(images, features, task) 也保留。
```

## 5. 离线验证

```powershell
python -m unittest discover -s NetTrainer/pixel_classifier_v3/tests -v
python -m unittest MyAutoPilot.tests.test_scs_telemetry -v
```

测试使用临时数据，验证 CSV/RGB/标签、采集过滤、会话划分、三个输出头、steering 反向传播、多任务更新和 V2 权重兼容性。真实游戏画面与遥测延迟需要安装插件后实测；合成数据上的训练只能验证管线可运行，不能说明驾驶准确率。
