# 代码风格

用户明确要求：以后新增或修改的代码，沿用用户现有代码的语言风格，不套用助手自己的默认风格。

主要参考 `NetTrainer/pixel_classifier_v2/bdd100k/train.py` 和 `Models/pc_v2.py`，并先阅读待修改文件附近的代码。

风格首先指代码结构，不只是空格。扩展已有版本时，直接以用户原文件为底稿，保持导入顺序、顶层数据加载、参数直接赋值、函数签名、全局变量用法、训练/验证/保存/绘图顺序，只增添所需功能。不要自行加入 `argparse`、`main()` 包装、路径引导代码或通用辅助层，也不要为了保留助手先前自行添加的结构而继续偏离原文件。

- Python 参数、列表项及多个变量之间使用 ` , `；关键字参数、默认参数使用 ` = `。例如 `def train_task(loader , task , criterion , classes):`、`torch.softmax(outputs , dim = 1)[: , 1]`。
- 沿用 `import torch.nn as nn`、`import torch.utils.data as tud` 等现有导入方式。
- 使用直白的变量名和中间步骤，例如 `road_train_dataset`、`road_train_loader`、`lane_probability`、`total_loss`、`inter`、`union`、`IoU`、`mIoU`。
- 训练脚本保持明确的数据集/loader、损失函数、`train_task` / `evaluate_task`、epoch 循环、验证、保存和绘图。优先直接的条件分支与循环，不把这些逻辑改造成通用任务或指标框架。
- 注释使用现有的简短英文风格，必要时写张量形状，例如 `#[batch , ch , 352 , 640]`。输出沿用 `print("current device:" , device)` 这种形式。
- 不为简单逻辑新增不必要的类型标注、长 docstring、海象运算符、嵌套推导式或多层辅助函数。
- 能直接赋值的一行逻辑就直接写，不额外包装成函数。例如设备选择沿用 `device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"`，不要改成 `get_device()`。
- 功能正确性、已有行为和用户手动改动仍需保留；遵循风格不意味着复制已知错误，也不意味着顺便重排无关文件。
- 数据统一存入已有的 `NetTrainer/datasets`，不要在项目根目录另建 `datasets`。数据集名称使用 `pc_lane_ds_a2d2`、`processed_pc_lane_ds_a2d2` 这种顺序，与 bdd100k、culane 保持一致。
