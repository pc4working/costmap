import math

import numpy as np

from costmap_generator import CameraConfig
from costmap_generator import CostmapConfig
from costmap_generator import CostmapGenerator


def make_generator(**overrides) -> CostmapGenerator:
    config = CostmapConfig(
        x_min=-1.0,
        x_max=1.0,
        y_min=0.0,
        y_max=3.0,
        min_points_per_cell=2,
        height_smoothing_sigma=0.0,
        traversable_smoothing_sigma=0.0,
        **overrides,
    )
    return CostmapGenerator(costmap_config=config)


def sample_surface(x_values, y_values, z_func):
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    zz = z_func(xx, yy)
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)


def test_camera_transform_matches_mount_geometry():
    camera_config = CameraConfig()
    generator = CostmapGenerator(camera_config=camera_config)

    transformed_origin = generator.transform_points(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(
        transformed_origin[0],
        np.array([-0.06, 0.41, 0.30], dtype=np.float32),
        atol=1e-6,
    )


def test_flat_ground_stays_low_cost():
    generator = make_generator()
    x_values = np.arange(-0.8, 0.8, 0.02)
    y_values = np.arange(0.4, 2.6, 0.02)
    ground = sample_surface(x_values, y_values, lambda xx, yy: np.zeros_like(xx))

    costmap = generator.generate(ground, frame="base")
    cell = generator.metric_to_grid(0.0, 1.5)

    assert cell is not None
    assert costmap[cell] == generator.costmap_config.flat_cost


def test_visible_free_space_is_marked_green():
    generator = make_generator(min_points_per_cell=1)
    points = np.array(
        [
            [-0.05, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.05, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    costmap = generator.generate(points, frame="base")
    free_cell = generator.metric_to_grid(0.0, 0.3)
    unknown_cell = generator.metric_to_grid(0.6, 0.3)

    assert free_cell is not None
    assert unknown_cell is not None
    assert costmap[free_cell] == generator.costmap_config.flat_cost
    assert costmap[unknown_cell] == generator.costmap_config.unknown_cost


def test_wall_becomes_obstacle():
    generator = make_generator()
    x_values = np.arange(-0.8, 0.8, 0.02)
    y_values = np.arange(0.4, 2.6, 0.02)
    ground = sample_surface(x_values, y_values, lambda xx, yy: np.zeros_like(xx))

    wall_x = np.arange(-0.12, 0.12, 0.02)
    wall_z = np.arange(0.0, 0.9, 0.02)
    xx, zz = np.meshgrid(wall_x, wall_z, indexing="ij")
    yy = np.full_like(xx, 1.5)
    wall = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)

    costmap = generator.generate(np.concatenate([ground, wall], axis=0), frame="base")
    cell = generator.metric_to_grid(0.0, 1.5)

    assert cell is not None
    assert costmap[cell] == generator.costmap_config.obstacle_cost


def test_stairs_are_high_cost_but_traversable():
    generator = make_generator(
        stair_height_max=0.4,
        obstacle_height=0.5,
        stair_variance_threshold=0.001,
    )
    x_values = np.arange(-0.6, 0.6, 0.02)
    y_values = np.arange(0.6, 2.4, 0.02)
    ground = sample_surface(x_values, y_values, lambda xx, yy: np.zeros_like(xx))

    stair_points = []
    step_ranges = [
        (1.0, 1.3, 0.0),
        (1.3, 1.6, 0.12),
        (1.6, 1.9, 0.24),
    ]
    for start_y, end_y, height in step_ranges:
        xs = np.arange(-0.3, 0.3, 0.02)
        ys = np.arange(start_y, end_y, 0.02)
        stair_points.append(sample_surface(xs, ys, lambda xx, yy, h=height: np.full_like(xx, h)))

    riser_specs = [
        (1.3, 0.0, 0.12),
        (1.6, 0.12, 0.24),
    ]
    for y_pos, z_start, z_end in riser_specs:
        xs = np.arange(-0.3, 0.3, 0.02)
        zs = np.arange(z_start, z_end, 0.02)
        xx, zz = np.meshgrid(xs, zs, indexing="ij")
        yy = np.full_like(xx, y_pos)
        stair_points.append(np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32))

    stairs = np.concatenate(stair_points, axis=0)
    costmap = generator.generate(np.concatenate([ground, stairs], axis=0), frame="base")
    cell = generator.metric_to_grid(0.0, 1.45)

    assert cell is not None
    assert costmap[cell] == generator.costmap_config.stair_cost


def test_slope_cost_increases_with_angle():
    generator = make_generator()
    x_values = np.arange(-0.6, 0.6, 0.02)
    y_values = np.arange(0.6, 2.6, 0.02)
    slope = sample_surface(
        x_values,
        y_values,
        lambda xx, yy: 0.2 * (yy - 0.6),
    )

    costmap = generator.generate(slope, frame="base")
    cell = generator.metric_to_grid(0.0, 1.8)

    assert cell is not None
    assert generator.costmap_config.flat_cost < costmap[cell] < generator.costmap_config.obstacle_cost
    expected_angle = math.degrees(math.atan(0.2))
    assert expected_angle > generator.costmap_config.flat_slope_deg
