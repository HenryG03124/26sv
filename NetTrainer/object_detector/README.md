# BDD100K 检测输入

输入与原 `BDDVehicleRoadDataset` 一致：RGB，`float32`，数值 `[0,1]`，单张图片形状 `(3,256,448)`。原始图片保留在 Downloads，读取时使用相同的 PIL 双线性缩放，不额外做 ImageNet 标准化。

`train_pairs.csv` / `val_pairs.csv` 分别对应 69,863 / 10,000 张有检测标注的图片。CSV 的图片路径指向 `C:/Users/HenryGuo/Downloads/BDD100K_detection/bdd100k/images/100k`，移动原图后需要重新运行整理脚本。

下载的训练原图共 70,000 张，其中 137 张没有原始检测标注，保留在 Downloads，但不加入训练索引，也不作为空目标/背景样本使用。名单见 `train_unannotated_images.txt`。

每张图片的原始像素坐标标签存放在 `train/labels`、`val/labels`。读取器会将边界框同步缩放至 448×256：横坐标乘 `448/原宽`，纵坐标乘 `256/原高`。返回的 `boxes` 为 `(N,4)` 的 float32 `[x1,y1,x2,y2]`，单位是缩放后图片的像素，不是归一化坐标。

```python
from torch.utils.data import DataLoader
from MyImageIdentifier.object_detector.bdd_objdetect import BDDDetectionDataset, detection_collate_fn

train_dataset = BDDDetectionDataset("MyImageIdentifier/datasets/processed_objdetect_ds/train_pairs.csv")
val_dataset = BDDDetectionDataset("MyImageIdentifier/datasets/processed_objdetect_ds/val_pairs.csv")
train_loader = DataLoader(train_dataset , batch_size = 64 , shuffle = True , collate_fn = detection_collate_fn)
val_loader = DataLoader(val_dataset , batch_size = 6 4, shuffle = False , collate_fn = detection_collate_fn)

images, targets = next(iter(train_loader))
# images: (B,3,256,448)，可直接送入你现有的 ResNet encoder。
# targets: 长度为 B 的 list，每张图片一个 dict；目标数量可以不同。
# targets[i]["boxes"]: (N,4), float32
# targets[i]["labels"]: (N,), int64
# 其他字段：area、iscrowd、image_id、orig_size、size。
```

类别编号固定：1 person、2 rider、3 car、4 truck、5 bus、6 train、7 motor、8 bike、9 traffic light、10 traffic sign。0 留给 detector 的背景类；若 detector 需要显式背景输出，类别总数为 11。空目标图片保留，返回 `(0,4)` 的 boxes 与 `(0,)` 的 labels。

本数据使用原始 2018 版检测标注，仅提取 `box2d`；道路、车道线多边形不作为检测框。超出原图边界的框会裁剪，裁剪后退化的框会剔除，数量记录在各 split 的 report 中。

现有 `trian_ResNet.py`、分割数据读取器和 `processed_pc_ds` 均保持原样。你添加 detector 时，需要使用上面的检测 Dataset 和 collate 函数；原来的掩码损失循环仍是语义分割训练循环。

重新整理：在工作区根目录运行 `python -m MyImageIdentifier.object_detector.prepare_bdd_objdetect`。每次完成整理后，会生成对应的 CSV 和 report；未完成的 CSV 使用 `.partial` 后缀。
