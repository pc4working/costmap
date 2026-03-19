# Costmap Generator

将 ZED 左目点云转换为 2D `uint8` 代价图，默认仅保留机器狗前方区域。

## 特性

- 预计算 4x4 齐次矩阵完成左目坐标系到机器人基座坐标系的变换
- 顶部 `DEFAULT_CONFIG` 可直接改参数，且每项附带中文说明
- `numpy` 向量化栅格统计，无逐点 Python 循环
- 基于栅格高度图的坡度估计
- 启发式楼梯检测
- 不做机器狗膨胀半径扩张，直接把 3D 结构映射为 2D 图像
- 机器人到首个观测点之间的无点区域自动标为绿色
- 其他无点区域默认标为红色
- `generate()`、`visualize()`、`run()` 三个主接口

## 依赖

```bash
python3 -m pip install -r requirements.txt
```

运行时需要安装 ZED SDK 对应版本的 `pyzed`，没有相机时也可以直接对 `numpy` 点云调用 `generate(...)`。

可直接编辑 [costmap_generator.py](/home/pc/code/costmap/costmap_generator.py) 顶部的 `DEFAULT_CONFIG`：

```python
DEFAULT_CONFIG = {
    "camera": {
        "tx": 0.41,  # 相机中心相对机器狗基座的前向偏移，单位 m
        ...
    },
    "costmap": {
        "forward_min": 0.0,  # 仅保留机器狗前方区域，前向起点
        "forward_max": 6.0,  # 仅保留机器狗前方区域，前向终点
        ...
    },
}
```

## 快速使用

```python
from costmap_generator import CostmapGenerator

generator = CostmapGenerator()
costmap = generator.generate(frame="camera")
image = generator.visualize(costmap)
```

直接运行主循环：

```bash
python3 costmap_generator.py
```

不打开窗口、只跑一帧：

```bash
python3 costmap_generator.py --once --no-vis
```

## 测试

```bash
python3 -m pytest -q
```
