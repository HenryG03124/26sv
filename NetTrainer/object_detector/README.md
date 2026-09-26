# BDD100K 检测输入

## Lite 三类检测

`train_Lite.py` 将 BDD 标签 `1 person、3 car、4 truck` 映射为模型类别 `0 pedestrian、1 car、2 truck`，其他标签不参与检测训练。类别损失权重为 `[1.0 , 0.30 , 1.0]`，训练 IoU 和验证 AP50、precision、recall 均包含 truck。

输入统一为 640×352（宽×高），与 PC V2 同步。`Models/od_Lite.py` 的网络结构不变，步长仍为 16，输出为 `(B,8,22,40)`：4 个框参数、1 个 objectness 和 3 个类别 logits。原有两类 `od_model_Lite.pth` 只有 7 个输出通道，不能直接加载到三类模型，需要重新训练。现有三类权重仍可加载，新分辨率的效果需要重新训练和验证。现有检测数据已经包含 truck，无需重新整理数据。

训练损失为 `class_loss + obj_loss + 5 × box_loss`：分类使用原有类别权重及 `0.05` 标签平滑；objectness 使用 Focal Loss（`alpha = 0.25`、`gamma = 2`），每张图按正样本网格数归一化后对 batch 求平均；框回归使用像素坐标下的 CIoU。空目标图片的归一化分母为 1。训练框不裁剪到图像边界，验证 IoU 仍按原来的裁剪方式计算。

实现参考 [YOLOv5 的 CIoU 与标签平滑](https://github.com/ultralytics/yolov5/blob/master/utils/loss.py)、[TorchVision RetinaNet 的 Focal Loss 归一化](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py) 和 [TorchVision CIoU](https://github.com/pytorch/vision/blob/main/torchvision/ops/ciou_loss.py)。使用项目 `requirements.txt` 中的 `torchvision` 算子；输出编码、目标分配、训练循环和数据加载保持原样。新旧 loss 数值不能直接比较，改进效果需要重训后比较相同验证集上的 mAP50、各类 AP50 和召回率。损失检查可运行 `python -m unittest discover -s NetTrainer/object_detector/tests -v`。

在项目根目录运行 `python -m NetTrainer.object_detector.train_Lite`。权重 `od_model_Lite.pth` 和曲线 `figure_Lite.png` 保存到运行时的当前目录。将训练好的权重放到 `NetTrainer/object_detector/od_model_Lite.pth` 用于单图测试，放到 `MyAutoPilot/weights/od_model_Lite.pth` 用于自动驾驶推理。

`test_Lite.py` 和 `MyAutoPilot` 已增加 truck 标签与橙色框；运动跟踪接收类别 2，并沿用现有制动逻辑。car 保留原有分割掩码过滤，truck 使用检测置信度和 NMS 筛选。

## 数据输入

输入与 PC V2 分辨率一致：RGB，`float32`，数值 `[0,1]`，单张图片形状 `(3,352,640)`。读取时使用 PIL 双线性缩放，不额外做 ImageNet 标准化。训练、单图测试和 `MyAutoPilot` 推理均使用 640×352 输入及 22×40 输出网格。

`train_pairs.csv` / `val_pairs.csv` 分别对应 69,863 / 10,000 张有检测标注的图片。CSV 的图片路径指向 `C:/Users/HenryGuo/Downloads/BDD100K_detection/bdd100k/images/100k`，移动原图后需要重新运行整理脚本。

下载的训练原图共 70,000 张，其中 137 张没有原始检测标注，保留在 Downloads，但不加入训练索引，也不作为空目标/背景样本使用。名单见 `train_unannotated_images.txt`。

每张图片的原始像素坐标标签存放在 `train/labels`、`val/labels`。读取器会将边界框同步缩放至 640×352：横坐标乘 `640/原宽`，纵坐标乘 `352/原高`。返回的 `boxes` 为 `(N,4)` 的 float32 `[x1,y1,x2,y2]`，单位是缩放后图片的像素，不是归一化坐标。

```python
from torch.utils.data import DataLoader
from MyImageIdentifier.object_detector.bdd_objdetect import BDDDetectionDataset, detection_collate_fn

train_dataset = BDDDetectionDataset("MyImageIdentifier/datasets/processed_objdetect_ds/train_pairs.csv")
val_dataset = BDDDetectionDataset("MyImageIdentifier/datasets/processed_objdetect_ds/val_pairs.csv")
train_loader = DataLoader(train_dataset , batch_size = 64 , shuffle = True , collate_fn = detection_collate_fn)
val_loader = DataLoader(val_dataset , batch_size = 64 , shuffle = False , collate_fn = detection_collate_fn)

images, targets = next(iter(train_loader))
# images: (B,3,352,640)，可直接送入你现有的 ResNet encoder。
# targets: 长度为 B 的 list，每张图片一个 dict；目标数量可以不同。
# targets[i]["boxes"]: (N,4), float32
# targets[i]["labels"]: (N,), int64
# 其他字段：area、iscrowd、image_id、orig_size、size。
```

类别编号固定：1 person、2 rider、3 car、4 truck、5 bus、6 train、7 motor、8 bike、9 traffic light、10 traffic sign。0 留给 detector 的背景类；若 detector 需要显式背景输出，类别总数为 11。空目标图片保留，返回 `(0,4)` 的 boxes 与 `(0,)` 的 labels。

本数据使用原始 2018 版检测标注，仅提取 `box2d`；道路、车道线多边形不作为检测框。超出原图边界的框会裁剪，裁剪后退化的框会剔除，数量记录在各 split 的 report 中。

现有 `train_ResNet.py`、分割数据读取器和 `processed_pc_ds` 均保持原样。你添加 detector 时，需要使用上面的检测 Dataset 和 collate 函数；原来的掩码损失循环仍是语义分割训练循环。

重新整理：在工作区根目录运行 `python -m MyImageIdentifier.object_detector.prepare_bdd_objdetect`。每次完成整理后，会生成对应的 CSV 和 report；未完成的 CSV 使用 `.partial` 后缀。
