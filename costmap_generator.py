from __future__ import annotations

import argparse
import math
import time
import warnings
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Literal

import cv2
import numpy as np

def _can_use_scipy() -> bool:
    try:
        numpy_version = metadata.version("numpy")
        scipy_version = metadata.version("scipy")
    except metadata.PackageNotFoundError:
        return False

    numpy_major = int(numpy_version.split(".")[0])
    scipy_parts = scipy_version.split(".")
    scipy_major = int(scipy_parts[0])
    scipy_minor = int(scipy_parts[1]) if len(scipy_parts) > 1 else 0

    if numpy_major >= 2 and (scipy_major, scipy_minor) < (1, 11):
        return False
    return True


if _can_use_scipy():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
            from scipy.ndimage import maximum_filter as scipy_maximum_filter
            from scipy.ndimage import minimum_filter as scipy_minimum_filter
        except Exception:  # pragma: no cover - exercised by import-time environments
            scipy_gaussian_filter = None
            scipy_maximum_filter = None
            scipy_minimum_filter = None
else:  # pragma: no cover - depends on local Python environment
    scipy_gaussian_filter = None
    scipy_maximum_filter = None
    scipy_minimum_filter = None

try:
    import pyzed.sl as sl
except Exception:  # pragma: no cover - hardware dependency
    sl = None


@dataclass(frozen=True)
class CameraConfig:
    tx: float = 0.41
    ty: float = 0.0
    tz: float = 0.30
    pitch_deg: float = -15.0
    left_eye_offset: float = 0.06

    def transform_matrix(self) -> np.ndarray:
        pitch = math.radians(self.pitch_deg)
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)

        translate_eye_to_center = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -self.left_eye_offset],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        rotate_pitch = np.array(
            [
                [cos_pitch, 0.0, sin_pitch, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-sin_pitch, 0.0, cos_pitch, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        translate_to_base = np.array(
            [
                [1.0, 0.0, 0.0, self.tx],
                [0.0, 1.0, 0.0, self.ty],
                [0.0, 0.0, 1.0, self.tz],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return translate_to_base @ rotate_pitch @ translate_eye_to_center


@dataclass(frozen=True)
class CostmapConfig:
    resolution: float = 0.05
    x_min: float = -1.0
    x_max: float = 5.0
    y_min: float = -3.0
    y_max: float = 3.0
    robot_radius: float = 0.25

    obstacle_height: float = 0.5
    flat_slope_deg: float = 5.0
    lethal_slope_deg: float = 45.0

    unknown_cost: int = 254
    obstacle_cost: int = 254
    flat_cost: int = 10
    stair_cost: int = 120
    slope_cost_min: int = 10
    slope_cost_max: int = 150

    min_points_per_cell: int = 3
    min_range: float = 0.2
    max_range: float = 6.0
    max_height_clip: float = 3.0

    stair_height_min: float = 0.10
    stair_height_max: float = 0.75
    stair_variance_threshold: float = 0.0025
    stair_neighbor_count: int = 3

    height_smoothing_sigma: float = 1.0
    traversable_smoothing_sigma: float = 0.8

    @property
    def rows(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def cols(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def inflation_radius_cells(self) -> int:
        return int(math.ceil(self.robot_radius / self.resolution))


class CostmapGenerator:
    def __init__(
        self,
        camera_config: CameraConfig | None = None,
        costmap_config: CostmapConfig | None = None,
    ) -> None:
        self.camera_config = camera_config or CameraConfig()
        self.costmap_config = costmap_config or CostmapConfig()
        self.transform = self.camera_config.transform_matrix()
        self.last_debug: dict[str, Any] = {}

        self._camera = None
        self._runtime_params = None
        self._point_cloud_mat = None

    def initialize_camera(self) -> None:
        if self._camera is not None:
            return
        if sl is None:
            raise RuntimeError("pyzed 未安装，无法直接读取 ZED 点云。")

        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_system = self._resolve_coordinate_system()
        init_params.coordinate_units = sl.UNIT.METER
        if hasattr(init_params, "depth_minimum_distance"):
            init_params.depth_minimum_distance = self.costmap_config.min_range
        if hasattr(init_params, "depth_maximum_distance"):
            init_params.depth_maximum_distance = self.costmap_config.max_range

        camera = sl.Camera()
        status = camera.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED 相机初始化失败: {status!r}")

        self._camera = camera
        self._runtime_params = sl.RuntimeParameters()
        self._point_cloud_mat = sl.Mat()

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
        self._camera = None
        self._runtime_params = None
        self._point_cloud_mat = None

    def __del__(self) -> None:  # pragma: no cover - destructor safety
        try:
            self.close()
        except Exception:
            pass

    def _resolve_coordinate_system(self) -> Any:
        candidates = (
            "RIGHT_HANDED_Z_UP_X_FWD",
            "RIGHT_HANDED_Z_UP_X_FORWARD",
            "RIGHT_HANDED_Z_UP",
        )
        for name in candidates:
            if hasattr(sl.COORDINATE_SYSTEM, name):
                return getattr(sl.COORDINATE_SYSTEM, name)
        raise AttributeError("当前 pyzed 不支持所需的 Z_UP 坐标系枚举。")

    def capture_point_cloud(self) -> np.ndarray:
        self.initialize_camera()
        status = self._camera.grab(self._runtime_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED 抓帧失败: {status!r}")

        self._camera.retrieve_measure(self._point_cloud_mat, sl.MEASURE.XYZRGBA)
        point_cloud = self._point_cloud_mat.get_data()
        return np.asarray(point_cloud[..., :3], dtype=np.float32)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        xyz = self._coerce_xyz(points)
        if xyz.size == 0:
            return xyz

        homogeneous = np.concatenate(
            [xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        transformed = homogeneous @ self.transform.T
        return transformed[:, :3].astype(np.float32, copy=False)

    def metric_to_grid(self, x: float, y: float) -> tuple[int, int] | None:
        cfg = self.costmap_config
        if not (cfg.x_min <= x < cfg.x_max and cfg.y_min <= y < cfg.y_max):
            return None
        x_bin = int((x - cfg.x_min) / cfg.resolution)
        y_bin = int((y - cfg.y_min) / cfg.resolution)
        row = cfg.rows - 1 - x_bin
        col = y_bin
        return row, col

    def generate(
        self,
        point_cloud: np.ndarray | None = None,
        frame: Literal["camera", "left_eye", "base"] = "camera",
        return_debug: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        if point_cloud is None:
            point_cloud = self.capture_point_cloud()
            frame = "camera"

        xyz = self._coerce_xyz(point_cloud)
        if frame in {"camera", "left_eye"}:
            xyz = self.transform_points(xyz)
        elif frame != "base":
            raise ValueError(f"不支持的点云坐标系: {frame}")

        costmap = self._generate_from_base_frame(xyz)
        if return_debug:
            return costmap, self.last_debug
        return costmap

    def _generate_from_base_frame(self, points: np.ndarray) -> np.ndarray:
        cfg = self.costmap_config
        rows, cols = cfg.shape
        total_cells = rows * cols

        points = self._filter_points(points)
        if points.size == 0:
            empty_costmap = np.full(cfg.shape, cfg.unknown_cost, dtype=np.uint8)
            self.last_debug = {"point_count": 0}
            return empty_costmap

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        x_bin = ((x - cfg.x_min) / cfg.resolution).astype(np.int32)
        y_bin = ((y - cfg.y_min) / cfg.resolution).astype(np.int32)
        row = rows - 1 - x_bin
        col = y_bin
        flat_index = row * cols + col

        counts = np.bincount(flat_index, minlength=total_cells).astype(np.int32)
        sum_z = np.bincount(flat_index, weights=z, minlength=total_cells).astype(np.float32)
        sum_z2 = np.bincount(flat_index, weights=z * z, minlength=total_cells).astype(np.float32)

        min_z = np.full(total_cells, np.inf, dtype=np.float32)
        max_z = np.full(total_cells, -np.inf, dtype=np.float32)
        np.minimum.at(min_z, flat_index, z)
        np.maximum.at(max_z, flat_index, z)

        mean_z = np.divide(
            sum_z,
            counts,
            out=np.full(total_cells, np.nan, dtype=np.float32),
            where=counts > 0,
        )
        variance_z = np.divide(
            sum_z2,
            counts,
            out=np.zeros(total_cells, dtype=np.float32),
            where=counts > 0,
        ) - np.square(np.nan_to_num(mean_z, nan=0.0))
        variance_z = np.clip(variance_z, 0.0, None)

        counts_grid = counts.reshape(cfg.shape)
        observed_mask = counts_grid > 0
        enough_points = counts_grid >= cfg.min_points_per_cell

        min_z_grid = min_z.reshape(cfg.shape)
        max_z_grid = max_z.reshape(cfg.shape)
        mean_z_grid = mean_z.reshape(cfg.shape)
        variance_grid = variance_z.reshape(cfg.shape)
        height_span = np.where(observed_mask, max_z_grid - min_z_grid, 0.0)

        smoothed_height = self._masked_gaussian(
            np.nan_to_num(mean_z_grid, nan=0.0),
            enough_points,
            cfg.height_smoothing_sigma,
        )
        slope_deg, slope_valid = self._estimate_slope(smoothed_height, enough_points)

        local_ground = self._local_min_height(min_z_grid, observed_mask)
        height_above_local_ground = np.where(
            observed_mask,
            max_z_grid - local_ground,
            0.0,
        )

        stair_mask = self._detect_stairs(height_span, variance_grid, enough_points)
        lethal_slope_mask = enough_points & slope_valid & (slope_deg >= cfg.lethal_slope_deg)
        obstacle_mask = enough_points & (
            (height_above_local_ground >= cfg.obstacle_height) | lethal_slope_mask
        )
        inflated_obstacles = self._inflate_obstacles(obstacle_mask)

        slope_mask = (
            enough_points
            & ~inflated_obstacles
            & ~stair_mask
            & slope_valid
            & (slope_deg >= cfg.flat_slope_deg)
            & (slope_deg < cfg.lethal_slope_deg)
        )
        flat_mask = enough_points & ~inflated_obstacles & ~stair_mask & ~slope_mask

        slope_cost = self._slope_cost(slope_deg)
        costmap = np.full(cfg.shape, cfg.unknown_cost, dtype=np.uint8)
        costmap[flat_mask] = np.uint8(cfg.flat_cost)
        costmap[slope_mask] = slope_cost[slope_mask]
        costmap[stair_mask & ~inflated_obstacles] = np.uint8(cfg.stair_cost)
        costmap[inflated_obstacles] = np.uint8(cfg.obstacle_cost)

        costmap = self._smooth_traversable_costs(costmap, flat_mask | slope_mask)

        self.last_debug = {
            "point_count": int(points.shape[0]),
            "counts": counts_grid,
            "mean_height": mean_z_grid,
            "min_height": min_z_grid,
            "max_height": max_z_grid,
            "height_span": height_span,
            "variance": variance_grid,
            "local_ground": local_ground,
            "height_above_local_ground": height_above_local_ground,
            "slope_deg": slope_deg,
            "slope_valid": slope_valid,
            "stair_mask": stair_mask,
            "obstacle_mask": obstacle_mask,
            "inflated_obstacle_mask": inflated_obstacles,
        }
        return costmap

    def visualize(self, costmap: np.ndarray, show: bool = False, window_name: str = "Costmap") -> np.ndarray:
        cfg = self.costmap_config
        costmap = np.asarray(costmap, dtype=np.uint8)

        traversable = costmap < cfg.obstacle_cost
        normalized = np.clip(
            (costmap.astype(np.float32) - cfg.flat_cost)
            / max(1.0, float(cfg.slope_cost_max - cfg.flat_cost)),
            0.0,
            1.0,
        )

        green = np.array([60, 180, 60], dtype=np.float32)
        red = np.array([50, 50, 220], dtype=np.float32)
        image = np.zeros((*cfg.shape, 3), dtype=np.uint8)
        image[...] = np.array([45, 45, 45], dtype=np.uint8)

        traversable_color = (
            green[None, None, :] * (1.0 - normalized[..., None])
            + red[None, None, :] * normalized[..., None]
        )
        image[traversable] = traversable_color[traversable].astype(np.uint8)

        robot_cell = self.metric_to_grid(0.0, 0.0)
        if robot_cell is not None:
            cv2.circle(
                image,
                (robot_cell[1], robot_cell[0]),
                max(2, int(round(self.costmap_config.robot_radius / self.costmap_config.resolution))),
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        image = cv2.resize(
            image,
            (cfg.cols * 4, cfg.rows * 4),
            interpolation=cv2.INTER_NEAREST,
        )
        if show:
            cv2.imshow(window_name, image)
        return image

    def run(self, visualize_output: bool = True, sleep_sec: float = 0.0) -> None:
        try:
            while True:
                costmap = self.generate(frame="camera")
                if visualize_output:
                    self.visualize(costmap, show=True)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                if sleep_sec > 0.0:
                    time.sleep(sleep_sec)
        finally:
            self.close()
            if visualize_output:
                cv2.destroyAllWindows()

    def _coerce_xyz(self, points: np.ndarray) -> np.ndarray:
        array = np.asarray(points, dtype=np.float32)
        if array.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if array.ndim == 3:
            array = array.reshape(-1, array.shape[-1])
        elif array.ndim != 2:
            raise ValueError("点云数组必须是 Nx3/Nx4 或 HxWx3/HxWx4")
        if array.shape[1] < 3:
            raise ValueError("点云数组至少需要 3 个通道")
        return array[:, :3].astype(np.float32, copy=False)

    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        cfg = self.costmap_config
        finite_mask = np.isfinite(points).all(axis=1)
        if not finite_mask.any():
            return np.empty((0, 3), dtype=np.float32)

        filtered = points[finite_mask]
        distance = np.linalg.norm(filtered, axis=1)
        in_range = (
            (distance >= cfg.min_range)
            & (distance <= cfg.max_range)
            & (filtered[:, 2] >= -cfg.max_height_clip)
            & (filtered[:, 2] <= cfg.max_height_clip)
            & (filtered[:, 0] >= cfg.x_min)
            & (filtered[:, 0] < cfg.x_max)
            & (filtered[:, 1] >= cfg.y_min)
            & (filtered[:, 1] < cfg.y_max)
        )
        return filtered[in_range]

    def _estimate_slope(self, height_map: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        resolution = self.costmap_config.resolution

        forward = np.roll(height_map, -1, axis=0)
        backward = np.roll(height_map, 1, axis=0)
        right = np.roll(height_map, -1, axis=1)
        left = np.roll(height_map, 1, axis=1)

        valid_x = valid_mask & np.roll(valid_mask, -1, axis=0) & np.roll(valid_mask, 1, axis=0)
        valid_y = valid_mask & np.roll(valid_mask, -1, axis=1) & np.roll(valid_mask, 1, axis=1)

        dzdx = np.zeros_like(height_map, dtype=np.float32)
        dzdy = np.zeros_like(height_map, dtype=np.float32)
        dzdx[valid_x] = (forward - backward)[valid_x] / (2.0 * resolution)
        dzdy[valid_y] = (right - left)[valid_y] / (2.0 * resolution)

        gradient_mag = np.hypot(dzdx, dzdy)
        slope = np.degrees(np.arctan(gradient_mag)).astype(np.float32)
        return slope, valid_x & valid_y

    def _detect_stairs(
        self,
        height_span: np.ndarray,
        variance_grid: np.ndarray,
        enough_points: np.ndarray,
    ) -> np.ndarray:
        cfg = self.costmap_config
        stair_candidate = (
            enough_points
            & (height_span >= cfg.stair_height_min)
            & (height_span <= cfg.stair_height_max)
            & (variance_grid >= cfg.stair_variance_threshold)
        )
        kernel = np.ones((3, 3), dtype=np.float32)
        neighbor_count = cv2.filter2D(
            stair_candidate.astype(np.float32),
            ddepth=-1,
            kernel=kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        return stair_candidate & (neighbor_count >= cfg.stair_neighbor_count)

    def _local_min_height(self, min_height: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        large_value = np.float32(1e6)
        padded = np.where(observed_mask, min_height, large_value).astype(np.float32)

        if scipy_minimum_filter is not None:
            local_ground = scipy_minimum_filter(padded, size=3, mode="nearest")
        else:
            kernel = np.ones((3, 3), dtype=np.uint8)
            local_ground = cv2.erode(padded, kernel, iterations=1)

        return np.where(observed_mask, local_ground, np.nan).astype(np.float32)

    def _inflate_obstacles(self, obstacle_mask: np.ndarray) -> np.ndarray:
        radius = self.costmap_config.inflation_radius_cells
        if radius <= 0:
            return obstacle_mask

        if scipy_maximum_filter is not None:
            inflated = scipy_maximum_filter(
                obstacle_mask.astype(np.uint8),
                size=radius * 2 + 1,
                mode="constant",
            )
            return inflated > 0

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
        return cv2.dilate(obstacle_mask.astype(np.uint8), kernel, iterations=1) > 0

    def _masked_gaussian(self, values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
        values = values.astype(np.float32, copy=False)
        if sigma <= 0.0 or not mask.any():
            return values.copy()

        weight = mask.astype(np.float32)
        weighted_values = values * weight

        if scipy_gaussian_filter is not None:
            blurred_values = scipy_gaussian_filter(weighted_values, sigma=sigma, mode="nearest")
            blurred_weight = scipy_gaussian_filter(weight, sigma=sigma, mode="nearest")
        else:
            blurred_values = cv2.GaussianBlur(
                weighted_values,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REPLICATE,
            )
            blurred_weight = cv2.GaussianBlur(
                weight,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REPLICATE,
            )

        normalized = np.divide(
            blurred_values,
            blurred_weight,
            out=values.copy(),
            where=blurred_weight > 1e-6,
        )
        return normalized.astype(np.float32, copy=False)

    def _smooth_traversable_costs(self, costmap: np.ndarray, smoothable_mask: np.ndarray) -> np.ndarray:
        sigma = self.costmap_config.traversable_smoothing_sigma
        if sigma <= 0.0 or not smoothable_mask.any():
            return costmap

        smoothed = self._masked_gaussian(costmap.astype(np.float32), smoothable_mask, sigma=sigma)
        result = costmap.copy()
        result[smoothable_mask] = np.clip(np.rint(smoothed[smoothable_mask]), 0, 253).astype(np.uint8)
        return result

    def _slope_cost(self, slope_deg: np.ndarray) -> np.ndarray:
        cfg = self.costmap_config
        normalized = (slope_deg - cfg.flat_slope_deg) / max(
            1e-6,
            cfg.lethal_slope_deg - cfg.flat_slope_deg,
        )
        normalized = np.clip(normalized, 0.0, 1.0)
        cost = cfg.slope_cost_min + normalized * (cfg.slope_cost_max - cfg.slope_cost_min)
        return np.clip(np.rint(cost), 0, 253).astype(np.uint8)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ZED 点云到 2D 代价图生成器")
    parser.add_argument("--once", action="store_true", help="只处理一帧")
    parser.add_argument("--no-vis", action="store_true", help="不打开 OpenCV 可视化窗口")
    parser.add_argument("--sleep", type=float, default=0.0, help="循环模式下的帧间隔秒数")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    generator = CostmapGenerator()

    if args.once:
        costmap = generator.generate(frame="camera")
        if args.no_vis:
            print(costmap)
        else:
            generator.visualize(costmap, show=True)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        generator.close()
        return

    generator.run(visualize_output=not args.no_vis, sleep_sec=args.sleep)


if __name__ == "__main__":
    main()
