# ZED 点云 → 代价图生成方案

## Context

机器狗前侧安装 ZED 立体相机（前方 41cm、上方 30cm、俯角 15°、左目偏中心 6cm），需要将实时点云转换为 2D 代价图供导航使用。核心挑战：正确的坐标变换、高效的栅格化、以及适配四足机器人运动能力的地形分类。

## 文件结构

单文件实现：`/home/pc/code/localplanner/costmap_generator.py`

## 实现步骤

### Step 1: 数据类定义

`CameraConfig` — 相机外参
- tx=0.41m, ty=0.0m, tz=0.30m, pitch=-15°, left_eye_offset=0.06m

`CostmapConfig` — 栅格参数与代价阈值（全部可配置）
- 分辨率 5cm，栅格 6m×6m（前向偏置），机器人半径 0.25m
- 障碍物高度阈值：默认 0.5m，作为可调参数 `obstacle_height`
- 代价值：未知区域=254, 障碍物=254, 平地=10, 楼梯=120, 坡度按角度线性插值 10~150
- 所有阈值均为构造参数，可在实例化时覆盖

### Step 2: 坐标变换（预计算 4×4 齐次矩阵）

变换链：左目坐标系 → 相机中心（平移 -0.06m Y）→ 旋转 -15° pitch → 平移到机器人基座
- 坐标系：`RIGHT_HANDED_Z_UP`（X 前，Y 左，Z 上）
- 一次矩阵乘法完成全部点的变换

### Step 3: ZED 相机初始化与点云获取

```python
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
init_params.coordinate_units = sl.UNIT.METER
camera.retrieve_measure(mat, sl.MEASURE.XYZRGBA)
```
- 过滤 NaN/inf 无效点，距离裁剪

### Step 4: 栅格化（核心性能热点）

全 numpy 向量化，零 Python 循环：
- `np.bincount` 统计每格点数、高度和、高度平方和
- `np.minimum.at` / `np.maximum.at` 求每格最小/最大高度
- 计算均值、方差 → 用于地形分类

### Step 5: 坡度估计

对高度图做中心差分梯度（类 Sobel），`arctan(gradient_mag)` 得到坡度角。
- 不做逐点法线估计（太慢），用栅格级梯度代替

### Step 6: 楼梯检测

启发式规则：
- 格内高度差在 0.10~0.75m（1~3 级台阶）
- 高度方差超阈值（非平面）
- 邻域一致性检查（楼梯跨多格）

### Step 7: 代价赋值

优先级从高到低：
1. 点数不足 → 未知(254)
2. 最大高度 ≥ 障碍阈值 → 不可通行(254)
3. 坡度 ≥ 45° → 不可通行(254)
4. 检测为楼梯 → 可通行但高代价(120)
5. 5°~45° 坡度 → 线性插值(10~150)
6. <5° → 平地(10)

### Step 8: 后处理

- 障碍物膨胀（按机器人半径，`scipy.ndimage.maximum_filter`）
- 轻度高斯平滑（仅可通行区域）

### Step 9: 可视化与主循环

- `generate()` → 返回 `np.ndarray (rows, cols)` uint8 代价图
- `visualize(costmap)` → OpenCV 彩色渲染（绿=低代价，红=高代价，深灰=不可通行），标记机器人位置
- `run(visualize=True)` → 连续循环，每帧生成 + 显示
- 输出格式：纯 numpy 数组，方便下游集成

## 依赖

```
pyzed (ZED SDK 5.1)
numpy
scipy
opencv-python  # 可视化（必需）
```

## 验证方式

1. 将相机对准已知距离的物体，打印变换前后坐标，验证外参正确性
2. 对准平地 → costmap 应全绿（低代价）
3. 对准墙壁/高障碍 → 应标记为不可通行
4. 对准楼梯 → 应标记为可通行但高代价
5. 对准斜坡 → 代价应随角度递增
