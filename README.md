# Costmap Generator

将 ZED 左目点云转换为 2D `uint8` 代价图，面向四足机器人前向导航。

## 特性

- 预计算 4x4 齐次矩阵完成左目坐标系到机器人基座坐标系的变换
- `numpy` 向量化栅格统计，无逐点 Python 循环
- 基于栅格高度图的坡度估计
- 启发式楼梯检测
- 障碍物膨胀与可通行区域平滑
- `generate()`、`visualize()`、`run()` 三个主接口

## 依赖

```bash
python3 -m pip install -r requirements.txt
```

运行时需要安装 ZED SDK 对应版本的 `pyzed`，没有相机时也可以直接对 `numpy` 点云调用 `generate(...)`。

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
