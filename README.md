# Costmap Generator

将 ZED 左目点云转换为 2D `uint8` 代价图，默认仅保留机器狗前方区域。

当前公开坐标约定：
- `x` 轴左为正
- `y` 轴前为正
- 图像显示也与此一致：左边对应 `x` 正方向，上方对应 `y` 正方向

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
        "x": 0.0,   # 左向偏移，左为正
        "y": 0.41,  # 前向偏移，前为正
        ...
    },
    "costmap": {
        "x_min": -3.0,  # 右边界
        "x_max": 3.0,   # 左边界
        "y_min": 0.0,   # 机器狗所在位置
        "y_max": 6.0,   # 前方最远距离
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
