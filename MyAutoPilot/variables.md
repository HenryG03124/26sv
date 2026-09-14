# MyAutoPilot 变量说明

本文档按源码文件记录变量、参数和主要中间量。坐标均采用 OpenCV/NumPy 的图像坐标系：`x` 向右增大，`y` 向下增大；颜色元组使用 OpenCV 的 BGR 顺序。

## 通用数据约定

| 名称 | 含义 |
| --- | --- |
| `frame` | 原始或正在处理的一帧 BGR 图像，形状通常为 `[高度, 宽度, 3]`。 |
| `mask` | 像素分类结果，形状为 `[高度, 宽度]`；`0` 是背景，`1` 是道路，`2` 是车辆，v2 中的 `3` 是车道线。 |
| `height` / `width` | 图像或掩码的高度、宽度，单位为像素。 |
| `x` / `y` | 像素横、纵坐标；在检测器网格中则表示网格列、行。 |
| `center_points` | 道路中心点列表，每项是 `[x, y]`，通常按从近到远排列。 |
| `self` | 当前类实例；`self.xxx` 是保存在实例上的网络层、模型或采集资源。 |

## `main.py`（基础版入口）

### 全局配置

| 变量 | 当前值 | 用途 |
| --- | --- | --- |
| `pc_colors` | 3 个 BGR 颜色 | 把像素类别 `0/1/2` 映射为背景黑、道路绿、车辆红，用于叠加显示。 |
| `od_colors` | 2 个 BGR 颜色 | 目标检测框的类别颜色，索引与 `label` 对应。 |
| `class_names` | `("pedestrian", "car")` | 目标检测类别名称，索引与 `label` 对应。 |
| `alpha` | `0.3` | 分割颜色层的透明度；原图权重为 `1 - alpha`。 |
| `car_mask_ratio_threshold` | `0.6` | 检测框内车辆分割像素至少应占的比例，低于它的框会被过滤。 |
| `monitor_index` | `2` | `mss` 使用的显示器编号；`0` 通常表示所有屏幕组成的虚拟区域。 |
| `obj_detect` | `"enable"` | 是否执行并绘制目标检测；只有字符串等于 `"enable"` 时开启。 |

### `draw_boxes(...)`

| 变量 | 用途 |
| --- | --- |
| `detector` | 已加载权重的 `ObjectDetector`。 |
| `frame` | 接收检测框和标签绘制的图像。 |
| `mask` | 用来验证车辆检测框的语义分割掩码。 |
| `car_mask_ratio_threshold` | 本次过滤检测框使用的车辆像素占比阈值。 |
| `pos` | 置信度达标的检测网格位置。 |
| `boxes` | 过滤和 NMS 后的边界框集合。 |
| `scores` | 各保留框的置信分数。 |
| `labels` | 各保留框的类别编号。 |
| `box` / `score` / `label` | 当前循环正在绘制的单个框、分数和类别。 |
| `x1`, `y1`, `x2`, `y2` | 当前框的左上角和右下角像素坐标。 |

### `main()` 中的局部变量

| 变量 | 用途 |
| --- | --- |
| `capturer` | 屏幕采集器实例。 |
| `classifier` | 基础版像素分类器实例。 |
| `detector` | 目标检测器实例。 |
| `frame` | 最新抓取的屏幕帧。 |
| `mask` | `classifier` 输出的三分类掩码。 |
| `display_frame` | 缩放并叠加可视化结果后的显示帧。 |
| `active` | `mask != 0` 的布尔数组，标记需要着色的非背景像素。 |
| `color_mask` | 通过 `pc_colors[mask]` 生成的彩色分割图。 |
| `blended` | 原图与彩色分割图按 `alpha` 混合后的图像。 |
| `width` / `height` | 被采集显示器的原生宽、高，用于把结果窗口缩放到一半。 |

## `main_v2.py`（车道与转向版入口）

### 全局配置

| 变量 | 当前值 | 用途 |
| --- | --- | --- |
| `pc_colors` | 4 个 BGR 颜色 | 把类别 `0/1/2/3` 映射为背景、道路、车辆、车道线的显示颜色。 |
| `od_colors` | 2 个 BGR 颜色 | 行人与车辆检测框的颜色。 |
| `class_names` | `("pedestrian", "car")` | 检测类别名称。 |
| `alpha` | `0.3` | 分割掩码叠加透明度。 |
| `car_mask_ratio_threshold` | `0.6` | 检测框内车辆像素占比阈值。 |
| `lane_prob_threshold` | `0.7` | 像素被判定为车道线所需的最小概率。越高越保守。 |
| `center_offset_threshold` | `10 px` | 相邻道路中心点允许的最大横向跳变；超过后会进行平滑修正。 |
| `y_far` | `200 px` | 较远的道路中心采样行，用于计算航向角。 |
| `y_near` | `324 px` | 较近的道路中心采样行，用于计算横向偏移和航向角。 |
| `K_offset` | `1` | 横向偏移在转向公式中的增益。 |
| `K_hdg` | `1` | 航向误差在转向公式中的增益。 |
| `monitor_index` | `1` | 要采集的显示器编号。 |
| `obj_detect` | `"enable"` | 目标检测功能开关。 |
| `road_center` | `"enable"` | 道路中心检测与绘制开关。 |
| `road_center_algorithm` | `1` | 道路中心算法选择：`1` 为 `RoadCenterA`，`2` 为 `RoadCenterB`；其他值会导致检测器未定义。 |
| `steer` | `"enable"` | 转向值计算开关；还要求 `road_center` 同时开启。 |

`draw_boxes(...)` 中的变量与 `main.py` 同名变量含义相同；其中过滤使用 `mask.shape` 作为原始框坐标尺寸。

### `draw_road_center(...)`

| 变量 | 用途 |
| --- | --- |
| `frame` | 要绘制道路中心线和采样点的显示帧。 |
| `center_points` | 检测和过滤后的道路中心点。少于 3 个时不绘制。 |
| `y_far` / `y_near` | 需要特别标出的远、近采样行。 |
| `points` | 转为 OpenCV 折线格式 `[-1, 1, 2]` 的中心点数组。 |
| `xydict` | 从纵坐标 `y` 到横坐标 `x` 的字典，便于按指定采样行查点。 |
| `x` / `y` | 构建字典和绘制采样点时使用的中心点坐标。 |

### `main()` 中的局部变量

| 变量 | 用途 |
| --- | --- |
| `steering` | 当前平滑后的转向指令，范围由计算过程限制在约 `[-1, 1]`。 |
| `steering_prev` | 上一帧的转向指令，供低通平滑使用。 |
| `invalid_count` | 连续缺少 `y_far` 或 `y_near` 中心点的帧数。前 5 帧会逐步回正。 |
| `diff` | 最近一次有效转向值的五分之一，中心点短暂丢失时每帧从 `steering` 中减去。 |
| `capturer` | 屏幕采集器实例。 |
| `classifier` | v2 像素分类器实例。 |
| `detector` | 目标检测器实例。 |
| `road_center_detector` | 当前选中的道路中心算法类（A 或 B）。 |
| `frame` | 最新屏幕帧。 |
| `mask` | 包含背景、道路、车辆和车道线的四分类掩码。 |
| `display_frame` | 用于叠加和显示结果的帧。 |
| `active` | 非背景像素布尔掩码。 |
| `color_mask` | 按类别着色后的分割图。 |
| `blended` | 原图与分割颜色的混合结果。 |
| `center_points` | 当前帧检测并平滑后的道路中心点。 |
| `offset` | 近处道路中心相对画面中心的归一化横向偏移。 |
| `hdg` | 由远、近中心点计算出的航向误差，单位为弧度。 |
| `width` / `height` | 显示器原生分辨率，用于调整结果窗口大小。 |

## `object_detector.py`

### 全局变量

| 变量 | 当前值 | 用途 |
| --- | --- | --- |
| `device` | `"cuda"` 或 `"cpu"` | PyTorch 推理设备；检测到 CUDA 时使用 GPU。 |
| `ResNet18_encoder` | 神经网络模块 | 从输入图像提取特征的 ResNet-18 风格编码器。 |
| `detector` | 神经网络模块 | 把编码特征变为 7 通道检测输出：4 个框参数、1 个目标置信度、2 个类别分数。 |
| `grid_height` | `16` | 检测输出网格高度。 |
| `grid_width` | `28` | 检测输出网格宽度。 |
| `confidence_threshold` | `0.5` | 候选框最低综合置信度。 |
| `nms_threshold` | `0.35` | 非极大值抑制的 IoU 阈值。 |
| `car_mask_class_id` | `2` | 分割掩码中的车辆类别编号。 |

### `BasicBlock`

| 变量 | 用途 |
| --- | --- |
| `in_channels` / `out_channels` | 残差块输入、输出通道数。 |
| `stride` | 第一层卷积和必要时捷径分支的步长。 |
| `self.conv1` / `self.conv2` | 残差主分支的两层卷积。 |
| `self.bn1` / `self.bn2` | 两层卷积后的批归一化层。 |
| `self.relu` | 可原地执行的 ReLU 激活层。 |
| `self.shortcut` | 用于残差相加的捷径；尺寸变化时执行 `1x1` 卷积，否则为恒等映射。 |
| `x` | 残差块输入张量。 |
| `out` | 主分支逐层处理并与捷径相加后的中间/输出张量。 |

### `ObjectDetector` 与推理

| 变量 | 用途 |
| --- | --- |
| `self.model` | 由 `ResNet18_encoder` 和 `detector` 串联成的完整检测模型。 |
| `model_path` | 要加载的 `.pth` 权重文件路径。 |
| `model_input` | 缩放到 `448x256`、转为 RGB、归一化并增加批维度后的输入张量。 |
| `output` | 模型对单张图像的原始 7 通道输出。 |
| `box_output` | `output` 的前 4 个通道，表示框中心偏移及宽高。 |
| `obj` | 第 5 通道经 Sigmoid 后的目标存在概率。 |
| `probabilities` | 最后 2 个通道经 Softmax 后的类别概率。 |
| `class_scores` | 每个网格位置的最高类别概率。 |
| `labels` | 最高概率类别的编号。 |
| `scores` | `obj * class_scores` 得到的综合置信度。 |
| `selected` | `scores >= confidence_threshold` 的布尔网格。 |
| `positions` | 所有入选网格位置的坐标。 |

### 非极大值抑制与 IoU

| 变量 | 用途 |
| --- | --- |
| `boxes` | 待去重的边界框张量。 |
| `scores` / `labels` | 各框的置信度和类别。 |
| `threshold` | 当前 NMS 调用使用的 IoU 上限。 |
| `kept` | NMS 最终保留的框索引。 |
| `class_id` | 当前单独处理的类别编号。 |
| `indices` | 属于当前类别的框索引。 |
| `order` | 按置信度从高到低排列的索引。 |
| `current` | 当前确定保留的最高分框索引。 |
| `remaining` | 尚待与当前框比较的索引。 |
| `iou` | 当前框与其余框的交并比数组。 |
| `box` | IoU 计算中的单个基准框。 |
| `intersection_x1`, `intersection_y1` | 交集矩形左上角坐标。 |
| `intersection_x2`, `intersection_y2` | 交集矩形右下角坐标。 |
| `intersection` | 交集面积。 |
| `box_area` | 基准框面积。 |
| `boxes_area` | 其余各框面积。 |

### `box_filter(...)`

| 变量 | 用途 |
| --- | --- |
| `positions` | 置信度达标的网格位置。 |
| `box_output` | 网络输出的框参数。 |
| `scores` / `labels` | 全部网格位置的分数与类别。 |
| `original_width` / `original_height` | 边界框最终采用的图像坐标尺寸。 |
| `mask` | 用于车辆像素占比校验的二维分割掩码。 |
| `car_mask_ratio_threshold` | 框内车辆像素的最低比例。 |
| `boxes` | 解码后且通过掩码校验的框列表，随后会转为张量。 |
| `selected_scores` / `selected_labels` | 与 `boxes` 对应的分数和类别。 |
| `mask_height` / `mask_width` | 掩码尺寸。 |
| `position` | 当前候选网格位置。 |
| `x` / `y` | 当前网格列、行。 |
| `offset_x` / `offset_y` | 框中心在当前网格单元内的归一化偏移。 |
| `width_in_cells` / `height_in_cells` | 框宽高，以网格单元为单位。 |
| `center_x` / `center_y` | 映射到目标图像尺寸后的框中心。 |
| `box_width` / `box_height` | 映射后的框宽高。 |
| `first_x` / `first_y` | 裁剪到图像边界内的框左上角。 |
| `second_x` / `second_y` | 裁剪到图像边界内的框右下角。 |
| `mask_first_x` / `mask_first_y` | 映射到掩码坐标后的左上角。 |
| `mask_second_x` / `mask_second_y` | 映射到掩码坐标后的右下角。 |
| `box_mask` | 当前边界框覆盖的掩码切片。 |
| `kept` | NMS 返回的保留索引。 |

## `pixel_classifier.py`（基础 U-Net）

### 通用网络块变量

| 变量 | 用途 |
| --- | --- |
| `device` | 模型运行设备，优先 CUDA，否则 CPU。 |
| `in_channels` / `out_channels` | 当前网络块的输入、输出通道数。 |
| `stride` | 编码残差块第一层卷积的步长。 |
| `self.conv1` / `self.conv2` | `EncoderBlock` 主分支卷积层。 |
| `self.bn1` / `self.bn2` | 编码块批归一化层。 |
| `self.relu` | 编码块 ReLU 激活。 |
| `self.shortcut` | 编码块的残差捷径。 |
| `self.block` | `DecoderBlock` 中的两组“卷积 + 批归一化 + ReLU”。 |
| `x` | 网络或网络块的输入张量。 |
| `out` | `EncoderBlock` 的中间和最终输出。 |

### `UNet`

| 变量 | 用途 |
| --- | --- |
| `self.root` | 输入端的 `7x7` 卷积、归一化和激活。 |
| `self.pool` | 首次下采样的最大池化层。 |
| `self.encoder1` / `self.encoder2` / `self.encoder3` / `self.encoder4` | 四级编码器；逐级提取更抽象、分辨率更低的特征。 |
| `self.up1` / `self.up2` / `self.up3` / `self.up4` | 四级转置卷积上采样层。 |
| `self.decoder1` / `self.decoder2` / `self.decoder3` / `self.decoder4` | 融合跳跃连接特征后的四级解码器。 |
| `self.output` | 把最后的 32 通道特征上采样并变为 3 个类别分数。 |
| `root` | 根卷积输出，也是最后一级跳跃连接特征。 |
| `encoder1` / `encoder2` / `encoder3` / `encoder4` | 各编码阶段的特征张量。 |
| `decoder1` / `decoder2` / `decoder3` / `decoder4` | 各解码阶段的特征张量。 |
| `output` | 3 通道像素分类 logits。 |

### `PixelClassifier`

| 变量 | 用途 |
| --- | --- |
| `self.model` | 放到 `device` 上的 `UNet` 实例。 |
| `model_path` | U-Net 权重文件路径。 |
| `frame` | 待分割的 BGR 图像。 |
| `model_input` | 缩放至 `448x256`、转 RGB、归一化后的四维输入张量。 |
| `output` | 模型输出的像素类别 logits。 |
| `mask` | 对类别维取最大值后得到的 `uint8` 二维类别图。 |

## `pixel_classifier_v2.py`（道路/车道双头 U-Net）

基础编码器、解码器中的同名变量与 `pixel_classifier.py` 含义相同。

### 尺寸与模型结构

| 变量 | 当前值 | 用途 |
| --- | --- | --- |
| `device` | `"cuda"` 或 `"cpu"` | 模型运行设备。 |
| `INPUT_HEIGHT` | `352` | 模型输入高度。 |
| `INPUT_WIDTH` | `640` | 模型输入宽度。 |
| `INPUT_SIZE` | `(640, 352)` | 传给 OpenCV `resize` 的 `(宽, 高)`。 |
| `self.road_head` | 3 类输出头 | 预测背景、道路和车辆。 |
| `self.lane_up` | 上采样模块 | 把共享解码特征转换成 16 通道车道语义特征。 |
| `self.lane_detail` | 浅层细节模块 | 直接从原输入提取 8 通道局部细节。 |
| `self.lane_head` | 2 类输出头 | 融合语义与细节后预测“非车道线/车道线”。 |

### `forward(...)` 与 `head(...)`

| 变量 | 用途 |
| --- | --- |
| `x` | 预处理后的输入图像张量。 |
| `root`, `encoder1`, `encoder2`, `encoder3`, `encoder4` | 根特征和四级编码特征。 |
| `decoder1`, `decoder2`, `decoder3`, `decoder4` | 四级解码特征；`forward` 最终返回 `decoder1`。 |
| `task` | 输出头选择；`"road"` 使用道路头，其他值使用车道头。 |
| `lane_semantic` | 从共享解码特征上采样得到的车道语义特征。 |
| `lane_detail` | 从原图直接提取的细节特征。 |
| `lane_feature` | `lane_semantic` 与 `lane_detail` 沿通道维拼接的 24 通道特征。 |

### `PixelClassifierV2.predict(...)`

| 变量 | 用途 |
| --- | --- |
| `self.model` | v2 `UNet` 模型实例。 |
| `model_path` | v2 权重文件路径。 |
| `frame` | 待分割的 BGR 图像。 |
| `lane_prob_threshold` | 车道线概率阈值。 |
| `model_input` | 按 `INPUT_SIZE` 预处理后的模型输入。 |
| `output` | `UNet.forward` 返回的共享解码特征。 |
| `road_output` | 道路头的 3 类 logits。 |
| `lane_output` | 车道头的 2 类 logits。 |
| `mask` | 道路头取最大类别后得到的类别张量，最后转成 NumPy 数组。 |
| `lane_probability` | 每个像素属于车道线的概率。 |
| `lane_mask` | `lane_probability > lane_prob_threshold` 的布尔掩码。 |

## `road_center_A.py`

算法 A 从画面近处向远处逐行扫描。

| 变量 | 用途 |
| --- | --- |
| `mask` | v2 四分类掩码。 |
| `height` / `width` | 掩码尺寸。 |
| `center_points` | 已找到的道路中心点；初始点位于画面底部中心附近。 |
| `prev_center_x` | 上一个可靠中心点的横坐标，用于区分左右车道线。 |
| `left_x` / `right_x` | 最近识别到的左、右边界横坐标。 |
| `lane_width` | 最近一次双侧车道线可靠时估计出的车道宽度。 |
| `unreliable_count` | 连续使用单侧推算或无法可靠识别的次数；超过 3 次时暂停添加点。 |
| `bottom` | 是否还处于尚未同时看到左右车道线的底部区域。 |
| `y` | 当前扫描行，每次向上移动 4 像素。 |
| `road` | 当前行中类别为道路（`1`）的所有横坐标。 |
| `lane` | 当前行中类别为车道线（`3`）的所有横坐标。 |
| `left` / `right` | 位于 `prev_center_x` 左、右侧的车道线横坐标。 |
| `lane_center_x` | 当前行估计的道路中心横坐标。 |
| `center_offset_threshold` | 中心点过滤时允许的最大横向跳变。 |
| `prev` | 过滤器使用的前一个中心点。 |
| `i` | 当前过滤的中心点索引。 |

## `road_center_B.py`

算法 B 先找锚点，再分别向近处和远处扫描。

| 变量 | 当前值/用途 |
| --- | --- |
| `anchor_y` | `280 px`；期望的起始扫描行，实际会选择可用扫描行中最接近它的一行。 |
| `mask` | v2 四分类掩码。 |
| `y_values` | 某个扫描方向上依次处理的行坐标。 |
| `prev_center_x` | 上一个中心横坐标。 |
| `lane_width` | 已知车道宽度，用于缺失单侧边界时估算中心。 |
| `width` / `height` | 掩码宽、高。 |
| `center_points` | 当前方向或完整算法检测到的中心点列表。 |
| `unreliable_count` | 连续不可靠估算次数。 |
| `y` | 当前扫描行。 |
| `road` / `lane` | 当前行的道路、车道线像素横坐标。 |
| `left` / `right` | 中心点左、右侧的车道线横坐标。 |
| `left_x` / `right_x` | 选中的左右边界坐标。 |
| `lane_center_x` | 估算的当前道路中心横坐标。 |
| `scan_rows` | 从近到远、间隔 4 像素的全部候选扫描行。 |
| `anchor` | `scan_rows` 中最接近 `anchor_y` 的实际锚点行。 |
| `near_rows` / `far_rows` | 从锚点分别向画面近端、远端扩展的扫描行。 |
| `near_points` / `far_points` | 两个方向分别检测到的中心点。 |
| `center_offset_threshold` | 中心点过滤允许的最大横向跳变。 |
| `prev` | 过滤器中的前一个中心点。 |
| `i` | 当前过滤的中心点索引。 |

## `screen_capture.py`

| 变量 | 用途 |
| --- | --- |
| `monitor_index` | 交给 `mss` 的显示器编号。 |
| `self.screen` | `mss.mss()` 屏幕抓取会话。 |
| `self.capture_area` | 所选显示器的抓取区域字典，包含位置和尺寸。 |
| `screenshot` | `mss` 返回的 BGRA 原始截图对象。 |
| `frame` | 截图转为 NumPy 数组并由 BGRA 转成 BGR 后的图像。 |

`native_resolution()` 从 `self.capture_area` 返回宽、高；`show_frame(frame)` 中的 `frame` 是要预览的 BGR 图像。

## `steering.py`

| 变量 | 当前值/用途 |
| --- | --- |
| `prev_weight` | `0.8`；平滑时上一帧转向值的权重。 |
| `raw_weight` | `0.2`；平滑时当前原始转向值的权重。 |
| `width` | `640 px`；转向计算假定的掩码宽度。 |
| `center_points` | 道路中心点列表。 |
| `y_far` / `y_near` | 航向与偏移计算所需的远、近采样行。 |
| `existing_y` | 当前中心点中所有纵坐标组成的集合。 |
| `xydict` | 纵坐标到横坐标的映射。 |
| `offset` | 近处中心相对画面中心的归一化偏移：左侧为负，右侧为正。 |
| `hdg` | 远近中心连线的航向误差，单位为弧度。 |
| `K_offset` / `K_hdg` | 横向偏移和航向误差的控制增益。 |
| `steering_prev` | 上一帧已平滑的转向值。 |
| `steering_raw` | `K_offset * offset + K_hdg * hdg` 的结果，并裁剪到 `[-1, 1]`。 |
| `steering` | 按 `prev_weight` 和 `raw_weight` 平滑后的最终转向指令。 |

## 模型文件

| 文件 | 用途 |
| --- | --- |
| `pc_model_U-Net.pth` | 基础三分类像素模型权重，由 `main.py` 加载。 |
| `pc_model_v2.pth` | 道路/车道双头像素模型权重，由 `main_v2.py` 加载。 |
| `od_model_Lite.pth` | 轻量目标检测模型权重，由两个入口加载。 |
