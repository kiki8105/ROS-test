"""A* path planning + obstacle-aware Pure Pursuit in one ROS 2 node.

Pipeline:
    /map + /amcl_pose + code waypoints or /goal_pose
    -> A* -> /plan -> Pure Pursuit
    /scan -> TRACK / TURN / BYPASS safety controller -> /cmd_vel

The original pure_pursuit_obstacle.py remains unchanged for comparison.
"""

from collections import deque
from dataclasses import dataclass
import heapq
import itertools
import math

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rcl_interfaces.msg import (
    FloatingPointRange,
    IntegerRange,
    ParameterDescriptor,
    SetParametersResult,
)
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import LaserScan


GridCell = tuple[int, int]
WorldPoint = tuple[float, float]

# turtlebot3_waffle_pi.urdf 기준 base_footprint 좌표 [m].
WAFFLE_PI_LIDAR_X = -0.064
WAFFLE_PI_FRONT_X = 0.080

# ============================================================================
# 사용자 코드 WP 입력 구역
# ----------------------------------------------------------------------------
# 순서대로 (x, y)를 입력하면 각 WP마다 A* 경로를 새로 계산하고,
# Pure Pursuit + LiDAR 회피로 이동합니다. 빈 목록이면 /goal_pose 단일 목표만
# 사용합니다. 좌표를 수정한 뒤 패키지를 다시 빌드하세요.
#
# 예시:
# CODE_MISSION_WAYPOINTS = [
#     (1.0, 0.5),
#     (2.0, 1.0),
#     (0.0, 0.5),
# ]
CODE_MISSION_WAYPOINTS = [
]
# ============================================================================


def validate_mission_waypoints(points):
    """Return finite (x, y) mission points or raise a clear configuration error."""
    validated = []
    for index, point in enumerate(points):
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError(
                f'CODE_MISSION_WAYPOINTS[{index}]는 (x, y) 형식이어야 합니다.')
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'CODE_MISSION_WAYPOINTS[{index}]의 x, y는 숫자여야 합니다.') from error
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(
                f'CODE_MISSION_WAYPOINTS[{index}]의 x, y는 유한한 값이어야 합니다.')
        validated.append((x, y))
    return validated


@dataclass(frozen=True)
class GridSpec:
    """Geometry used to convert between map cells and world coordinates."""

    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    frame_id: str = 'map'


def normalize_angle(angle):
    """Return an angle in the [-pi, pi) interval."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value, limit):
    """Clamp a signed value to +/- limit."""
    limit = abs(float(limit))
    return max(-limit, min(limit, float(value)))


def rate_limit(current, target, limit, dt):
    """Limit command change by limit per second."""
    maximum_change = max(0.0, float(limit)) * max(0.0, float(dt))
    delta = max(-maximum_change, min(maximum_change, target - current))
    return current + delta


def quaternion_yaw(orientation):
    """Convert a finite quaternion-like object to planar yaw."""
    values = (
        float(orientation.x),
        float(orientation.y),
        float(orientation.z),
        float(orientation.w),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('쿼터니언에 유한하지 않은 값이 있습니다.')
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-9:
        raise ValueError('쿼터니언 크기가 0입니다.')
    x, y, z, w = (value / norm for value in values)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def stamp_nanoseconds(stamp):
    """Return a positive ROS message timestamp, or None when it is unset."""
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else None


def propagate_odom_pose(map_anchor, odom_anchor, odom_current):
    """Apply an odom-frame SE(2) delta to an AMCL map-frame anchor."""
    map_x, map_y, map_yaw = map_anchor
    odom_x, odom_y, odom_yaw = odom_anchor
    current_x, current_y, current_yaw = odom_current
    delta_x = current_x - odom_x
    delta_y = current_y - odom_y
    frame_yaw = normalize_angle(map_yaw - odom_yaw)
    cosine = math.cos(frame_yaw)
    sine = math.sin(frame_yaw)
    return (
        map_x + cosine * delta_x - sine * delta_y,
        map_y + sine * delta_x + cosine * delta_y,
        normalize_angle(
            map_yaw + normalize_angle(current_yaw - odom_yaw)),
    )


def scan_points(scan, lidar_x=WAFFLE_PI_LIDAR_X):
    """Convert valid finite LaserScan returns to base-frame x/y points."""
    if (
            scan is None
            or not scan.ranges
            or not math.isfinite(float(scan.angle_min))
            or not math.isfinite(float(scan.angle_increment))
            or abs(float(scan.angle_increment)) < 1e-12
            or not math.isfinite(float(scan.range_min))
            or not math.isfinite(float(scan.range_max))
            or float(scan.range_min) < 0.0
            or float(scan.range_max) <= float(scan.range_min)):
        return [], 0

    minimum = float(scan.range_min)
    maximum = float(scan.range_max)
    epsilon = max(1e-6, maximum * 1e-6)
    points = []
    usable = 0
    for index, raw in enumerate(scan.ranges):
        try:
            distance = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(distance) or distance < minimum - epsilon:
            continue
        if math.isinf(distance):
            if distance > 0.0:
                usable += 1
            continue
        if distance > maximum + epsilon:
            continue
        usable += 1
        angle = float(scan.angle_min) + index * float(scan.angle_increment)
        points.append((
            float(lidar_x) + distance * math.cos(angle),
            distance * math.sin(angle),
        ))
    return points, usable


def scan_sector_distance(scan, center, half_width, percentile=0.10):
    """Return (near-distance percentile, usable-count) for an angle sector."""
    if (
            scan is None
            or not scan.ranges
            or not math.isfinite(float(scan.angle_min))
            or not math.isfinite(float(scan.angle_increment))
            or abs(float(scan.angle_increment)) < 1e-12
            or not math.isfinite(float(scan.range_min))
            or not math.isfinite(float(scan.range_max))
            or float(scan.range_min) < 0.0
            or float(scan.range_max) <= float(scan.range_min)):
        return math.nan, 0

    minimum = float(scan.range_min)
    maximum = float(scan.range_max)
    epsilon = max(1e-6, maximum * 1e-6)
    values = []
    for index, raw in enumerate(scan.ranges):
        angle = float(scan.angle_min) + index * float(scan.angle_increment)
        if abs(normalize_angle(angle - center)) > half_width:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(value) or value < minimum - epsilon:
            continue
        if math.isinf(value):
            if value < 0.0:
                continue
            value = maximum
        elif value > maximum + epsilon:
            continue
        values.append(value)

    if not values:
        return math.nan, 0
    values.sort()
    percentile = max(0.0, min(1.0, float(percentile)))
    index = int((len(values) - 1) * percentile)
    return values[index], len(values)


def nth_smallest(values, count):
    """Return the count-th smallest value or infinity when hits are sparse."""
    ordered = sorted(values)
    return ordered[count - 1] if len(ordered) >= count else math.inf


def front_body_clearance(
        scan, half_width=0.18, min_hits=3,
        lidar_x=WAFFLE_PI_LIDAR_X,
        body_front_x=WAFFLE_PI_FRONT_X):
    """Return free distance from the body front inside its forward corridor."""
    points, _ = scan_points(scan, lidar_x)
    gaps = [
        max(0.0, x - body_front_x)
        for x, y in points
        if x > lidar_x and abs(y) < half_width
    ]
    return nth_smallest(gaps, max(1, int(min_hits)))


def minimum_radial_clearance(scan, lidar_x=WAFFLE_PI_LIDAR_X):
    """Return the nearest obstacle radius around the robot rotation centre."""
    points, _ = scan_points(scan, lidar_x)
    if not points:
        return math.inf
    return min(math.hypot(x, y) for x, y in points)


def world_to_grid(x, y, spec):
    """Convert a world point to (grid_x, grid_y), or None when out of map."""
    x = float(x)
    y = float(y)
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    delta_x = x - spec.origin_x
    delta_y = y - spec.origin_y
    cosine = math.cos(spec.origin_yaw)
    sine = math.sin(spec.origin_yaw)
    local_x = cosine * delta_x + sine * delta_y
    local_y = -sine * delta_x + cosine * delta_y
    grid_x = math.floor(local_x / spec.resolution)
    grid_y = math.floor(local_y / spec.resolution)
    if 0 <= grid_x < spec.width and 0 <= grid_y < spec.height:
        return int(grid_x), int(grid_y)
    return None


def grid_to_world(grid_x, grid_y, spec):
    """Return the world coordinate at the centre of a grid cell."""
    local_x = (float(grid_x) + 0.5) * spec.resolution
    local_y = (float(grid_y) + 0.5) * spec.resolution
    cosine = math.cos(spec.origin_yaw)
    sine = math.sin(spec.origin_yaw)
    return (
        spec.origin_x + cosine * local_x - sine * local_y,
        spec.origin_y + sine * local_x + cosine * local_y,
    )


def inflate_obstacles(blocked, width, height, resolution, radius):
    """Inflate blocked cells by a circular robot clearance radius."""
    if len(blocked) != width * height:
        raise ValueError('blocked 배열 크기가 지도 크기와 다릅니다.')
    result = [bool(value) for value in blocked]
    radius = max(0.0, float(radius))
    if radius <= 0.0:
        return result
    cell_radius = int(math.ceil(radius / resolution))
    radius_squared = (radius / resolution) ** 2 + 1e-9
    offsets = [
        (offset_x, offset_y)
        for offset_y in range(-cell_radius, cell_radius + 1)
        for offset_x in range(-cell_radius, cell_radius + 1)
        if offset_x * offset_x + offset_y * offset_y <= radius_squared
    ]
    occupied = [
        (index % width, index // width)
        for index, value in enumerate(blocked) if value
    ]
    for grid_x, grid_y in occupied:
        for offset_x, offset_y in offsets:
            target_x = grid_x + offset_x
            target_y = grid_y + offset_y
            if 0 <= target_x < width and 0 <= target_y < height:
                result[target_y * width + target_x] = True
    return result


def occupancy_grid_to_blocked(
        message, occupied_threshold=50, unknown_is_blocked=True,
        inflation_radius=0.18):
    """Validate OccupancyGrid and return inflated cells plus map geometry."""
    width = int(message.info.width)
    height = int(message.info.height)
    resolution = float(message.info.resolution)
    if width <= 0 or height <= 0:
        raise ValueError('지도 width와 height는 0보다 커야 합니다.')
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError('지도 resolution이 유효하지 않습니다.')
    if len(message.data) != width * height:
        raise ValueError('OccupancyGrid data 크기가 width*height와 다릅니다.')
    origin = message.info.origin
    origin_x = float(origin.position.x)
    origin_y = float(origin.position.y)
    if not math.isfinite(origin_x) or not math.isfinite(origin_y):
        raise ValueError('지도 origin 위치가 유효하지 않습니다.')
    spec = GridSpec(
        width=width,
        height=height,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_yaw=quaternion_yaw(origin.orientation),
        frame_id=message.header.frame_id or 'map',
    )
    threshold = int(occupied_threshold)
    raw_blocked = [
        (int(value) < 0 and bool(unknown_is_blocked))
        or int(value) >= threshold
        for value in message.data
    ]
    return (
        inflate_obstacles(
            raw_blocked, width, height, resolution, inflation_radius),
        spec,
    )


def nearest_free_cell(blocked, spec, cell, search_radius):
    """Return closest free cell within the exact metric search radius."""
    if cell is None:
        return None
    start_x, start_y = cell
    if not blocked[start_y * spec.width + start_x]:
        return cell
    radius_cells = max(0.0, float(search_radius)) / spec.resolution
    maximum = int(math.ceil(radius_cells))
    offsets = []
    for offset_y in range(-maximum, maximum + 1):
        for offset_x in range(-maximum, maximum + 1):
            distance_squared = offset_x * offset_x + offset_y * offset_y
            if distance_squared <= radius_cells * radius_cells + 1e-12:
                offsets.append((distance_squared, offset_x, offset_y))
    offsets.sort()
    for _, offset_x, offset_y in offsets:
        grid_x = start_x + offset_x
        grid_y = start_y + offset_y
        if (
                0 <= grid_x < spec.width
                and 0 <= grid_y < spec.height
                and not blocked[grid_y * spec.width + grid_x]):
            return grid_x, grid_y
    return None


def _octile_heuristic(first, second):
    delta_x = abs(first[0] - second[0])
    delta_y = abs(first[1] - second[1])
    diagonal = min(delta_x, delta_y)
    straight = max(delta_x, delta_y) - diagonal
    return math.sqrt(2.0) * diagonal + straight


def astar_path(
        blocked, width, height, start, goal,
        allow_diagonal=True, max_expansions=250000):
    """Find an optimal grid path with A* and no diagonal corner cutting."""
    if len(blocked) != width * height:
        raise ValueError('blocked 배열 크기가 지도 크기와 다릅니다.')
    for cell in (start, goal):
        if cell is None:
            return None
        grid_x, grid_y = cell
        if not (0 <= grid_x < width and 0 <= grid_y < height):
            return None
        if blocked[grid_y * width + grid_x]:
            return None
    if start == goal:
        return [start]

    cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    moves = cardinal + diagonal if allow_diagonal else cardinal
    total = width * height
    start_index = start[1] * width + start[0]
    goal_index = goal[1] * width + goal[0]
    costs = [math.inf] * total
    previous = [-1] * total
    closed = bytearray(total)
    costs[start_index] = 0.0
    counter = itertools.count()
    queue = [(_octile_heuristic(start, goal), next(counter), start_index)]
    expansions = 0

    while queue and expansions < int(max_expansions):
        _, _, current_index = heapq.heappop(queue)
        if closed[current_index]:
            continue
        closed[current_index] = 1
        expansions += 1
        if current_index == goal_index:
            indices = []
            cursor = current_index
            while cursor >= 0:
                indices.append(cursor)
                cursor = previous[cursor]
            indices.reverse()
            return [(index % width, index // width) for index in indices]

        current_x = current_index % width
        current_y = current_index // width
        for delta_x, delta_y in moves:
            next_x = current_x + delta_x
            next_y = current_y + delta_y
            if not (0 <= next_x < width and 0 <= next_y < height):
                continue
            next_index = next_y * width + next_x
            if blocked[next_index] or closed[next_index]:
                continue
            is_diagonal = delta_x != 0 and delta_y != 0
            if is_diagonal:
                horizontal = current_y * width + next_x
                vertical = next_y * width + current_x
                if blocked[horizontal] or blocked[vertical]:
                    continue
            candidate = costs[current_index] + (
                math.sqrt(2.0) if is_diagonal else 1.0)
            if candidate + 1e-12 >= costs[next_index]:
                continue
            costs[next_index] = candidate
            previous[next_index] = current_index
            heuristic = _octile_heuristic((next_x, next_y), goal)
            heapq.heappush(
                queue,
                (candidate + heuristic, next(counter), next_index),
            )
    return None


def supercover_cells(start, end):
    """Return every grid cell touched by a segment between cell centres."""
    start_x, start_y = start
    end_x, end_y = end
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    count_x = abs(delta_x)
    count_y = abs(delta_y)
    sign_x = 1 if delta_x > 0 else -1 if delta_x < 0 else 0
    sign_y = 1 if delta_y > 0 else -1 if delta_y < 0 else 0
    x, y = start_x, start_y
    index_x = 0
    index_y = 0
    cells = [(x, y)]
    while index_x < count_x or index_y < count_y:
        decision = (1 + 2 * index_x) * count_y - (
            1 + 2 * index_y) * count_x
        if decision == 0:
            cells.append((x + sign_x, y))
            cells.append((x, y + sign_y))
            x += sign_x
            y += sign_y
            index_x += 1
            index_y += 1
        elif decision < 0:
            x += sign_x
            index_x += 1
        else:
            y += sign_y
            index_y += 1
        cells.append((x, y))
    return cells


def line_is_free(blocked, width, height, start, end):
    """Return true when a segment's supercover contains only free cells."""
    for grid_x, grid_y in supercover_cells(start, end):
        if not (0 <= grid_x < width and 0 <= grid_y < height):
            return False
        if blocked[grid_y * width + grid_x]:
            return False
    return True


def simplify_grid_path(path, blocked, width, height):
    """Remove redundant cells without cutting through inflated obstacles."""
    if not path:
        return []
    if len(path) <= 2:
        return list(path)
    simplified = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        candidate = len(path) - 1
        while candidate > anchor + 1:
            if line_is_free(
                    blocked, width, height,
                    path[anchor], path[candidate]):
                break
            candidate -= 1
        simplified.append(path[candidate])
        anchor = candidate
    return simplified


def select_lookahead_point(
        x, y, segment_start, path, waypoint_index, lookahead_distance,
        corner_angle_limit=math.radians(35.0)):
    """Select a point lookahead_distance ahead along a waypoint polyline."""
    if waypoint_index >= len(path):
        return float(x), float(y)
    start_x, start_y = segment_start
    goal_x, goal_y = path[waypoint_index]
    vector_x = goal_x - start_x
    vector_y = goal_y - start_y
    length_squared = vector_x * vector_x + vector_y * vector_y
    if length_squared > 1e-12:
        raw_projection = (
            (x - start_x) * vector_x + (y - start_y) * vector_y
        ) / length_squared
        if raw_projection > 1.0:
            return goal_x, goal_y
        projection = max(0.0, raw_projection)
        cursor_x = start_x + projection * vector_x
        cursor_y = start_y + projection * vector_y
    else:
        cursor_x, cursor_y = goal_x, goal_y
    remaining = max(0.0, float(lookahead_distance))
    index = waypoint_index
    while index < len(path):
        goal_x, goal_y = path[index]
        delta_x = goal_x - cursor_x
        delta_y = goal_y - cursor_y
        segment_length = math.hypot(delta_x, delta_y)
        if segment_length > 1e-12 and remaining <= segment_length:
            ratio = remaining / segment_length
            return (
                cursor_x + ratio * delta_x,
                cursor_y + ratio * delta_y,
            )
        if segment_length > 1e-12 and index + 1 < len(path):
            next_x, next_y = path[index + 1]
            next_heading = math.atan2(next_y - goal_y, next_x - goal_x)
            segment_heading = math.atan2(delta_y, delta_x)
            corner_angle = abs(normalize_angle(
                next_heading - segment_heading))
            if corner_angle >= float(corner_angle_limit):
                # 큰 코너에서는 현재 꼭짓점에 먼저 도착한 뒤 다음 선분을
                # 바라본다. Pure Pursuit가 장애물 안쪽을 자르는 것을 막는다.
                return goal_x, goal_y
        remaining = max(0.0, remaining - segment_length)
        cursor_x, cursor_y = goal_x, goal_y
        index += 1
    return path[-1]


def point_to_segment_distance(point, start, end):
    """Return distance from point to a finite line segment."""
    vector_x = end[0] - start[0]
    vector_y = end[1] - start[1]
    length_squared = vector_x * vector_x + vector_y * vector_y
    if length_squared <= 1e-12:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * vector_x
        + (point[1] - start[1]) * vector_y
    ) / length_squared
    projection = max(0.0, min(1.0, projection))
    closest = (
        start[0] + projection * vector_x,
        start[1] + projection * vector_y,
    )
    return math.hypot(point[0] - closest[0], point[1] - closest[1])


def distance_to_path(point, path, waypoint_index=0, segment_start=None):
    """Return distance to the remaining portion of a world path."""
    if not path or waypoint_index >= len(path):
        return math.inf
    points = list(path[waypoint_index:])
    if segment_start is not None:
        points.insert(0, segment_start)
    if len(points) == 1:
        return math.hypot(point[0] - points[0][0], point[1] - points[0][1])
    return min(
        point_to_segment_distance(point, points[index - 1], points[index])
        for index in range(1, len(points))
    )


def pure_pursuit_arc_is_free(
        blocked, spec, x, y, yaw, target_distance, alpha):
    """Check the predicted robot-centre arc against the inflated map."""
    if blocked is None or spec is None:
        return False
    target_distance = float(target_distance)
    alpha = float(alpha)
    if target_distance <= 1e-9:
        cell = world_to_grid(x, y, spec)
        return cell is not None and not blocked[cell[1] * spec.width + cell[0]]

    curvature = 2.0 * math.sin(alpha) / target_distance
    if abs(curvature) < 1e-9:
        arc_length = target_distance
    else:
        arc_length = abs((2.0 * alpha) / curvature)
    sample_step = max(0.01, spec.resolution * 0.40)
    sample_count = max(1, int(math.ceil(arc_length / sample_step)))
    previous_cell = world_to_grid(x, y, spec)
    if previous_cell is None:
        return False
    for index in range(1, sample_count + 1):
        distance = arc_length * index / sample_count
        if abs(curvature) < 1e-9:
            local_x = distance
            local_y = 0.0
        else:
            angle = curvature * distance
            local_x = math.sin(angle) / curvature
            local_y = (1.0 - math.cos(angle)) / curvature
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        world_x = x + cosine * local_x - sine * local_y
        world_y = y + sine * local_x + cosine * local_y
        cell = world_to_grid(world_x, world_y, spec)
        if cell is None:
            return False
        if not line_is_free(
                blocked, spec.width, spec.height, previous_cell, cell):
            return False
        previous_cell = cell
    return True


def floating_descriptor(description, minimum, maximum, step=0.01):
    """Build a floating-point ROS parameter descriptor."""
    return ParameterDescriptor(
        description=description,
        floating_point_range=[FloatingPointRange(
            from_value=float(minimum),
            to_value=float(maximum),
            step=float(step),
        )],
    )


def integer_descriptor(description, minimum, maximum, step=1):
    """Build an integer ROS parameter descriptor."""
    return ParameterDescriptor(
        description=description,
        integer_range=[IntegerRange(
            from_value=int(minimum),
            to_value=int(maximum),
            step=int(step),
        )],
    )


class PurePursuitObstacleV2(Node):
    """Plan globally with A* and track safely with LiDAR-aware Pure Pursuit."""

    def __init__(self, **kwargs):
        super().__init__('pure_pursuit_obstacle_v2', **kwargs)

        self._declare_parameters()

        self.latest_map = None
        self.blocked = None
        self.grid_spec = None
        self.map_signature = None
        self.costmap_dirty = False
        self.planner_state = 'WAIT_MAP'

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.is_localized = False
        self.last_pose_time = None
        self.last_amcl_time = None
        self.latest_odom_pose = None
        self.map_anchor_pose = None
        self.odom_anchor_pose = None
        self.last_odom_time = None
        self.latest_odom_stamp_ns = None
        self.odom_history = deque(maxlen=200)

        self.requested_goal = None
        self.requested_goal_yaw = 0.0
        self.goal_source = None
        self.active_goal = None
        self.path = []
        self.path_cells = []
        self.current_waypoint_index = 0
        self.segment_start = None
        self.completed = False
        self.plan_count = 0
        self.last_plan_time = None
        self.replan_requested = False

        # Mission WP index is separate from current_waypoint_index, which is
        # used for the simplified points inside one A* leg.
        self.mission_waypoints = validate_mission_waypoints(
            CODE_MISSION_WAYPOINTS)
        self.mission_waypoint_index = -1
        self.mission_active = False
        self.mission_completed = False
        self.mission_failed = False

        self.latest_scan = None
        self.last_scan_time = None
        self.left_dist = math.inf
        self.front_dist = math.inf
        self.right_dist = math.inf

        self.avoidance_state = 'TRACK'
        self.avoidance_active = False
        self.turn_direction = 0.0
        self.obstacle_clear_since = None
        self.avoidance_started = None
        self.avoidance_hit_distance = math.inf
        self.turn_start_yaw = 0.0
        self.replan_after_avoidance = False

        self.heading_align_active = False
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.last_drive_enabled = None
        self.last_control_time = self.get_clock().now()

        self.add_on_set_parameters_callback(self.validate_parameters)
        validation = self.validate_parameters([])
        if not validation.successful:
            raise ValueError(validation.reason)

        durable_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher_ = self.create_publisher(
            Twist, str(self.get_parameter('cmd_vel_topic').value), 10)
        self.path_publisher = self.create_publisher(
            Path, str(self.get_parameter('path_topic').value), durable_qos)
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter('map_topic').value),
            self.map_callback,
            durable_qos,
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            str(self.get_parameter('pose_topic').value),
            self.pose_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            str(self.get_parameter('odom_topic').value),
            self.odom_callback,
            qos_profile_sensor_data,
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            str(self.get_parameter('goal_topic').value),
            self.goal_callback,
            10,
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            str(self.get_parameter('scan_topic').value),
            self.scan_callback,
            qos_profile_sensor_data,
        )

        period = float(self.get_parameter('control_period').value)
        self.timer = self.create_timer(period, self.control_loop)
        self.get_logger().info(
            'Pure Pursuit Obstacle V2 준비: '
            '/map → A* → /plan → LiDAR 회피 → /cmd_vel')
        if self.mission_waypoints:
            self.start_code_mission()
            self.get_logger().info(
                f'코드 WP 임무 {len(self.mission_waypoints)}개를 로드했습니다. '
                '/plan을 확인한 뒤 drive_enabled=true로 변경하세요.')
        else:
            self.get_logger().info(
                'CODE_MISSION_WAYPOINTS가 비어 있습니다. Initial Pose와 '
                '/goal_pose를 지정한 뒤 drive_enabled=true로 변경하세요.')

    def _declare_parameters(self):
        """Declare planning, tracking and local-avoidance parameters."""
        self.declare_parameter(
            'drive_enabled', False,
            ParameterDescriptor(
                description='true일 때만 자동주행합니다. false면 즉시 정지합니다.'))
        self.declare_parameter(
            'scan_timeout', 0.5,
            floating_descriptor('LiDAR 미수신 정지 시간 [s].', 0.1, 2.0, 0.1))
        self.declare_parameter(
            'pose_timeout', 2.0,
            floating_descriptor(
                'AMCL과 odom 위치가 모두 미갱신일 때 정지 시간 [s].',
                0.1, 10.0, 0.1))
        self.declare_parameter(
            'linear_velocity', 0.08,
            floating_descriptor('정상 경로 추종 속도 [m/s].', 0.0, 0.22, 0.01))
        self.declare_parameter(
            'avoidance_angular_velocity', 0.23,
            floating_descriptor('장애물 앞 선회 속도 [rad/s].', 0.05, 1.0, 0.01))
        self.declare_parameter(
            'avoidance_linear_velocity', 0.05,
            floating_descriptor('장애물 옆 우회 속도 [m/s].', 0.01, 0.15, 0.01))
        self.declare_parameter(
            'obs_threshold', 0.40,
            floating_descriptor('차체 앞에서 회피를 시작할 거리 [m].', 0.15, 1.5, 0.01))
        self.declare_parameter(
            'lookahead_distance', 0.40,
            floating_descriptor('경로 위 전방 주시 거리 [m].', 0.10, 2.0, 0.01))
        self.declare_parameter(
            'max_angular_velocity', 0.60,
            floating_descriptor('모든 회전 명령의 절대 상한 [rad/s].', 0.10, 1.5, 0.01))
        self.declare_parameter(
            'goal_tolerance', 0.20,
            floating_descriptor('최종 목표 도착 허용 반경 [m].', 0.05, 0.50, 0.01))
        self.declare_parameter(
            'path_point_tolerance', 0.12,
            floating_descriptor('중간 경로점 통과 허용 반경 [m].', 0.05, 0.50, 0.01))
        self.declare_parameter(
            'linear_acceleration_limit', 0.15,
            floating_descriptor('선형 가속·감속 한계 [m/s^2].', 0.01, 1.0, 0.01))
        self.declare_parameter(
            'angular_acceleration_limit', 0.80,
            floating_descriptor('각가속·감속 한계 [rad/s^2].', 0.05, 3.0, 0.05))
        self.declare_parameter(
            'avoidance_clear_margin', 0.10,
            floating_descriptor('회피 해제 거리 추가 여유 [m].', 0.02, 0.50, 0.01))
        self.declare_parameter(
            'avoidance_clear_hold_time', 0.80,
            floating_descriptor('전방이 연속으로 열려 있어야 하는 시간 [s].', 0.10, 2.0, 0.10))
        self.declare_parameter(
            'turn_direction_deadband', 0.05,
            floating_descriptor('좌우 거리 차이가 이 값 이하면 동률로 봅니다 [m].', 0.0, 0.50, 0.01))
        self.declare_parameter(
            'front_corridor_half_width', 0.18,
            floating_descriptor('차체 전방 검사 통로의 반폭 [m].', 0.14, 0.50, 0.01))
        self.declare_parameter(
            'scan_min_hits', 3,
            integer_descriptor('유효 스캔과 장애물에 필요한 최소 광선 수.', 1, 20))
        self.declare_parameter(
            'spin_clearance_radius', 0.27,
            floating_descriptor('제자리 선회에 필요한 차체 중심 반경 [m].', 0.24, 0.60, 0.01))
        self.declare_parameter(
            'minimum_turn_angle', 0.35,
            floating_descriptor('우회 전 반드시 선회할 최소 각도 [rad].', 0.10, 1.57, 0.01))
        self.declare_parameter(
            'side_clearance', 0.35,
            floating_descriptor('우회 중 장애물 측면 목표 간격 [m].', 0.15, 1.0, 0.01))
        self.declare_parameter(
            'wall_follow_gain', 0.80,
            floating_descriptor('우회 중 측면 거리 보정 게인.', 0.10, 5.0, 0.10))
        self.declare_parameter(
            'avoidance_min_duration', 1.0,
            floating_descriptor('경로 재진입 전 최소 회피 시간 [s].', 0.20, 10.0, 0.10))
        self.declare_parameter(
            'avoidance_max_duration', 15.0,
            floating_descriptor('회피 제한 시간 [s]. 초과하면 안전 정지합니다.', 2.0, 120.0, 1.0))
        self.declare_parameter(
            'avoidance_progress_distance', 0.08,
            floating_descriptor('재진입 전 목표에 가까워질 최소 거리 [m].', 0.0, 1.0, 0.01))
        self.declare_parameter(
            'rejoin_heading_tolerance', 0.60,
            floating_descriptor('경로 재진입 방향 오차 상한 [rad].', 0.10, 1.50, 0.01))
        self.declare_parameter(
            'heading_rotate_enter', 1.05,
            floating_descriptor('제자리 목표 정렬을 시작할 방향 오차 [rad].', 0.30, 2.80, 0.01))
        self.declare_parameter(
            'heading_rotate_exit', 0.70,
            floating_descriptor('제자리 목표 정렬을 끝낼 방향 오차 [rad].', 0.10, 2.50, 0.01))
        self.declare_parameter(
            'heading_gain', 1.0,
            floating_descriptor('목표 방향 정렬 비례 게인.', 0.10, 4.0, 0.10))

        self.declare_parameter(
            'occupied_threshold', 50,
            integer_descriptor('이 값 이상인 지도 셀을 장애물로 봅니다.', 1, 100))
        self.declare_parameter(
            'unknown_is_blocked', True,
            ParameterDescriptor(description='미탐색(-1) 셀을 장애물로 봅니다.'))
        self.declare_parameter(
            'inflation_radius', 0.20,
            floating_descriptor('A* 장애물 팽창 반경 [m].', 0.14, 0.60, 0.01))
        self.declare_parameter(
            'nearest_free_radius', 0.30,
            floating_descriptor('막힌 시작/목표 주변 자유 셀 검색 반경 [m].', 0.0, 1.0, 0.01))
        self.declare_parameter(
            'allow_diagonal', True,
            ParameterDescriptor(description='A* 8방향 이동 허용 여부.'))
        self.declare_parameter(
            'max_expansions', 250000,
            integer_descriptor('A* 한 번의 최대 확장 셀 수.', 1000, 2000000, 1000))
        self.declare_parameter(
            'replan_distance', 0.45,
            floating_descriptor('TRACK 중 경로 이탈 재계획 거리 [m].', 0.10, 2.0, 0.01))
        self.declare_parameter(
            'replan_cooldown', 1.0,
            floating_descriptor('연속 재계획 최소 간격 [s].', 0.1, 10.0, 0.1))
        self.declare_parameter(
            'odom_anchor_max_age', 0.20,
            floating_descriptor(
                'AMCL 시각과 odom 기준 표본 최대 차이 [s].',
                0.01, 2.0, 0.01))
        self.declare_parameter(
            'odom_reanchor_timeout', 1.0,
            floating_descriptor(
                'odom 중단 후 기준점을 다시 잡는 시간 [s].',
                0.1, 10.0, 0.1))
        self.declare_parameter(
            'odom_jump_distance', 0.50,
            floating_descriptor(
                '한 odom 표본의 위치 점프 판정값 [m].',
                0.05, 5.0, 0.05))
        self.declare_parameter(
            'odom_jump_angle', 1.57,
            floating_descriptor(
                '한 odom 표본의 방향 점프 판정값 [rad].',
                0.1, 3.14, 0.01))
        self.declare_parameter(
            'control_period', 0.10,
            ParameterDescriptor(
                description='제어 주기 [s]. 변경하려면 재시작합니다.',
                read_only=True,
                floating_point_range=[FloatingPointRange(
                    from_value=0.02, to_value=1.0, step=0.01)]))

        for name, default, description in (
                ('map_topic', 'map', 'OccupancyGrid 지도 토픽.'),
                ('pose_topic', 'amcl_pose', 'AMCL 위치 토픽.'),
                ('odom_topic', 'odom', 'AMCL 사이를 보간할 Odometry 토픽.'),
                ('goal_topic', 'goal_pose', 'PoseStamped 목표 토픽.'),
                ('scan_topic', 'scan', 'LaserScan 토픽.'),
                ('cmd_vel_topic', 'cmd_vel', 'Twist 명령 토픽.'),
                ('path_topic', 'plan', '계획된 nav_msgs/Path 토픽.')):
            self.declare_parameter(
                name, default,
                ParameterDescriptor(description=description, read_only=True))

    def validate_parameters(self, parameters):
        """Reject unsafe live parameter combinations."""
        ranges = {
            'scan_timeout': (0.1, 2.0),
            'pose_timeout': (0.1, 10.0),
            'linear_velocity': (0.0, 0.22),
            'avoidance_angular_velocity': (0.05, 1.0),
            'avoidance_linear_velocity': (0.01, 0.15),
            'obs_threshold': (0.15, 1.5),
            'lookahead_distance': (0.10, 2.0),
            'max_angular_velocity': (0.10, 1.5),
            'goal_tolerance': (0.05, 0.50),
            'path_point_tolerance': (0.05, 0.50),
            'linear_acceleration_limit': (0.01, 1.0),
            'angular_acceleration_limit': (0.05, 3.0),
            'avoidance_clear_margin': (0.02, 0.50),
            'avoidance_clear_hold_time': (0.10, 2.0),
            'turn_direction_deadband': (0.0, 0.50),
            'front_corridor_half_width': (0.14, 0.50),
            'spin_clearance_radius': (0.24, 0.60),
            'minimum_turn_angle': (0.10, 1.57),
            'side_clearance': (0.15, 1.0),
            'wall_follow_gain': (0.10, 5.0),
            'avoidance_min_duration': (0.20, 10.0),
            'avoidance_max_duration': (2.0, 120.0),
            'avoidance_progress_distance': (0.0, 1.0),
            'rejoin_heading_tolerance': (0.10, 1.50),
            'heading_rotate_enter': (0.30, 2.80),
            'heading_rotate_exit': (0.10, 2.50),
            'heading_gain': (0.10, 4.0),
            'inflation_radius': (0.14, 0.60),
            'nearest_free_radius': (0.0, 1.0),
            'replan_distance': (0.10, 2.0),
            'replan_cooldown': (0.1, 10.0),
            'odom_anchor_max_age': (0.01, 2.0),
            'odom_reanchor_timeout': (0.1, 10.0),
            'odom_jump_distance': (0.05, 5.0),
            'odom_jump_angle': (0.1, 3.14),
            'control_period': (0.02, 1.0),
        }
        values = {name: self.get_parameter(name).value for name in ranges}
        values['scan_min_hits'] = self.get_parameter('scan_min_hits').value
        values['occupied_threshold'] = self.get_parameter(
            'occupied_threshold').value
        values['max_expansions'] = self.get_parameter('max_expansions').value
        for parameter in parameters:
            if parameter.name in values:
                values[parameter.name] = parameter.value

        for name, (minimum, maximum) in ranges.items():
            try:
                value = float(values[name])
            except (TypeError, ValueError):
                return SetParametersResult(
                    successful=False, reason=f'{name}에는 숫자가 필요합니다.')
            if not math.isfinite(value) or not minimum <= value <= maximum:
                return SetParametersResult(
                    successful=False,
                    reason=f'{name} 허용 범위는 {minimum}~{maximum}입니다.')
        for name, minimum, maximum in (
                ('scan_min_hits', 1, 20),
                ('occupied_threshold', 1, 100),
                ('max_expansions', 1000, 2000000)):
            try:
                value = int(values[name])
            except (TypeError, ValueError):
                return SetParametersResult(
                    successful=False, reason=f'{name}에는 정수가 필요합니다.')
            if not minimum <= value <= maximum:
                return SetParametersResult(
                    successful=False,
                    reason=f'{name} 허용 범위는 {minimum}~{maximum}입니다.')
        if float(values['avoidance_angular_velocity']) > float(
                values['max_angular_velocity']):
            return SetParametersResult(
                successful=False,
                reason='avoidance_angular_velocity는 max_angular_velocity 이하여야 합니다.')
        if float(values['heading_rotate_exit']) >= float(
                values['heading_rotate_enter']):
            return SetParametersResult(
                successful=False,
                reason='heading_rotate_exit는 heading_rotate_enter보다 작아야 합니다.')
        if float(values['avoidance_min_duration']) >= float(
                values['avoidance_max_duration']):
            return SetParametersResult(
                successful=False,
                reason='avoidance_min_duration은 avoidance_max_duration보다 작아야 합니다.')

        planning_names = {
            'occupied_threshold', 'unknown_is_blocked', 'inflation_radius',
            'nearest_free_radius', 'allow_diagonal', 'max_expansions',
        }
        if parameters and any(
                parameter.name in planning_names for parameter in parameters):
            self.costmap_dirty = True
            self.replan_requested = True
        return SetParametersResult(successful=True)

    def map_callback(self, message):
        """Keep the raw map and rebuild the inflated planning grid on change."""
        origin = message.info.origin.orientation
        signature = (
            int(message.info.width),
            int(message.info.height),
            float(message.info.resolution),
            float(message.info.origin.position.x),
            float(message.info.origin.position.y),
            float(origin.x), float(origin.y), float(origin.z), float(origin.w),
            message.header.frame_id,
            hash(tuple(message.data)),
        )
        if signature == self.map_signature:
            return
        self.latest_map = message
        self.map_signature = signature
        self.costmap_dirty = True
        if self.avoidance_state in ('TURN', 'BYPASS'):
            self.replan_requested = self.requested_goal is not None
            self.get_logger().info(
                '회피 중 지도 갱신 수신: TRACK 복귀 후 재계획합니다.')
            return
        if self.rebuild_costmap():
            self.replan_requested = self.requested_goal is not None

    def rebuild_costmap(self):
        """Recompute the inflated grid from the stored raw OccupancyGrid."""
        if self.latest_map is None:
            self.planner_state = 'WAIT_MAP'
            return False
        try:
            blocked, spec = occupancy_grid_to_blocked(
                self.latest_map,
                occupied_threshold=int(
                    self.get_parameter('occupied_threshold').value),
                unknown_is_blocked=bool(
                    self.get_parameter('unknown_is_blocked').value),
                inflation_radius=float(
                    self.get_parameter('inflation_radius').value),
            )
        except (TypeError, ValueError) as error:
            self.clear_path(f'지도 거부: {error}', state='WAIT_MAP')
            return False
        self.blocked = blocked
        self.grid_spec = spec
        self.costmap_dirty = False
        if self.mission_failed:
            self.planner_state = 'MISSION_FAILED'
        elif self.completed:
            self.planner_state = (
                'MISSION_COMPLETED' if self.mission_completed else 'COMPLETED')
        else:
            self.planner_state = (
                'WAIT_GOAL' if self.requested_goal is None else 'PLANNING')
        self.get_logger().info(
            f'지도 수신: {spec.width}x{spec.height}, '
            f'{spec.resolution:.3f}m/cell, frame={spec.frame_id}')
        return True

    def pose_callback(self, message):
        """Use AMCL as the absolute map anchor and reset odom drift."""
        expected_frame = self.grid_spec.frame_id if self.grid_spec else 'map'
        if (
                message.header.frame_id
                and message.header.frame_id != expected_frame):
            self.get_logger().error(
                f'AMCL frame={message.header.frame_id} 거부: '
                f'{expected_frame} 좌표가 필요합니다.',
                throttle_duration_sec=2.0)
            return
        x = float(message.pose.pose.position.x)
        y = float(message.pose.pose.position.y)
        try:
            yaw = quaternion_yaw(message.pose.pose.orientation)
        except ValueError as error:
            self.get_logger().error(
                f'AMCL pose 거부: {error}', throttle_duration_sec=2.0)
            return
        if not math.isfinite(x) or not math.isfinite(y):
            self.get_logger().error(
                'AMCL 위치가 유한하지 않아 무시합니다.',
                throttle_duration_sec=2.0)
            return
        first_pose = not self.is_localized
        self.current_x = x
        self.current_y = y
        self.current_yaw = yaw
        self.is_localized = True
        now = self.get_clock().now()
        self.map_anchor_pose = (x, y, yaw)
        self.odom_anchor_pose = self.odom_anchor_for_amcl(
            stamp_nanoseconds(message.header.stamp), now)
        if self.odom_anchor_pose is not None and self.latest_odom_pose is not None:
            self.current_x, self.current_y, self.current_yaw = (
                propagate_odom_pose(
                    self.map_anchor_pose,
                    self.odom_anchor_pose,
                    self.latest_odom_pose,
                )
            )
        self.last_amcl_time = now
        self.last_pose_time = now
        if first_pose:
            self.get_logger().info(
                f'AMCL 위치 수신: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}')
            if self.requested_goal is not None:
                self.replan_requested = True

    def odom_anchor_for_amcl(self, amcl_stamp_ns, now):
        """Find the odom sample closest to an AMCL measurement timestamp."""
        maximum_age = float(
            self.get_parameter('odom_anchor_max_age').value)
        if amcl_stamp_ns is not None and self.odom_history:
            stamp_ns, pose = min(
                self.odom_history,
                key=lambda sample: abs(sample[0] - amcl_stamp_ns),
            )
            if abs(stamp_ns - amcl_stamp_ns) <= maximum_age * 1e9:
                return pose
        if amcl_stamp_ns is not None:
            return None
        if self.latest_odom_pose is None or self.last_odom_time is None:
            return None
        receipt_age = (now - self.last_odom_time).nanoseconds / 1e9
        if 0.0 <= receipt_age <= maximum_age:
            return self.latest_odom_pose
        return None

    def odom_callback(self, message):
        """Continuously propagate the map pose with odometry deltas."""
        position = message.pose.pose.position
        try:
            yaw = quaternion_yaw(message.pose.pose.orientation)
        except ValueError as error:
            self.get_logger().warn(
                f'유효하지 않은 /odom pose 무시: {error}',
                throttle_duration_sec=2.0)
            return
        odom_pose = (float(position.x), float(position.y), yaw)
        if not all(math.isfinite(value) for value in odom_pose):
            self.get_logger().warn(
                '유효하지 않은 /odom pose를 무시합니다.',
                throttle_duration_sec=2.0)
            return

        now = self.get_clock().now()
        stamp_ns = stamp_nanoseconds(message.header.stamp)
        discontinuity_reasons = []
        if self.last_odom_time is not None:
            receipt_age = (now - self.last_odom_time).nanoseconds / 1e9
            if receipt_age < 0.0:
                discontinuity_reasons.append('시뮬레이션 시간 역행')
            elif receipt_age > float(
                    self.get_parameter('odom_reanchor_timeout').value):
                discontinuity_reasons.append(
                    f'odom {receipt_age:.1f}s 수신 중단')
        if (
                stamp_ns is not None
                and self.latest_odom_stamp_ns is not None
                and stamp_ns < self.latest_odom_stamp_ns):
            discontinuity_reasons.append('odom 타임스탬프 역행')
        if self.latest_odom_pose is not None:
            delta_distance = math.hypot(
                odom_pose[0] - self.latest_odom_pose[0],
                odom_pose[1] - self.latest_odom_pose[1],
            )
            delta_angle = abs(normalize_angle(
                odom_pose[2] - self.latest_odom_pose[2]))
            if delta_distance > float(
                    self.get_parameter('odom_jump_distance').value):
                discontinuity_reasons.append(
                    f'odom 위치 {delta_distance:.2f}m 점프')
            if delta_angle > float(
                    self.get_parameter('odom_jump_angle').value):
                discontinuity_reasons.append(
                    f'odom 방향 {delta_angle:.2f}rad 점프')

        if discontinuity_reasons:
            self.odom_history.clear()
        if stamp_ns is not None:
            self.odom_history.append((stamp_ns, odom_pose))
        self.latest_odom_pose = odom_pose
        self.latest_odom_stamp_ns = stamp_ns
        self.last_odom_time = now

        if not self.is_localized or self.map_anchor_pose is None:
            return
        if discontinuity_reasons:
            self.map_anchor_pose = (
                self.current_x, self.current_y, self.current_yaw)
            self.odom_anchor_pose = odom_pose
            self.last_pose_time = now
            self.get_logger().warn(
                'odom 기준점 재설정: ' + ', '.join(discontinuity_reasons),
                throttle_duration_sec=1.0)
            return
        if self.odom_anchor_pose is None:
            self.odom_anchor_pose = odom_pose
            self.map_anchor_pose = (
                self.current_x, self.current_y, self.current_yaw)
        else:
            self.current_x, self.current_y, self.current_yaw = (
                propagate_odom_pose(
                    self.map_anchor_pose,
                    self.odom_anchor_pose,
                    odom_pose,
                )
            )
        self.last_pose_time = now

    def _reset_route_for_new_goal(self):
        """Stop and discard state that belongs to the previous A* leg."""
        self.publish_stop()
        self.path = []
        self.path_cells = []
        self.current_waypoint_index = 0
        self.segment_start = None
        self.active_goal = None
        self.completed = False
        self.heading_align_active = False
        self.reset_avoidance_state()
        self.publish_path()

    def _activate_goal(self, x, y, yaw, source):
        """Install one validated goal and request an A* plan."""
        self._reset_route_for_new_goal()
        self.requested_goal = (float(x), float(y))
        self.requested_goal_yaw = float(yaw)
        self.goal_source = source
        self.planner_state = 'PLANNING'
        self.replan_requested = True

    def mission_goal_yaw(self, index):
        """Use the next mission leg direction for Path visualization."""
        current_x, current_y = self.mission_waypoints[index]
        if index + 1 < len(self.mission_waypoints):
            next_x, next_y = self.mission_waypoints[index + 1]
            return math.atan2(next_y - current_y, next_x - current_x)
        if index > 0:
            previous_x, previous_y = self.mission_waypoints[index - 1]
            return math.atan2(
                current_y - previous_y, current_x - previous_x)
        return 0.0

    def _activate_mission_waypoint(self, index):
        """Select one code-defined mission waypoint as the current goal."""
        if not 0 <= index < len(self.mission_waypoints):
            return False
        self.mission_waypoint_index = index
        x, y = self.mission_waypoints[index]
        self._activate_goal(
            x, y, self.mission_goal_yaw(index), source='MISSION')
        self.get_logger().info(
            f'코드 임무 WP {index + 1}/{len(self.mission_waypoints)} 설정: '
            f'({x:.3f}, {y:.3f})')
        return True

    def start_code_mission(self):
        """Start or restart the waypoint list declared at the top of this file."""
        if not self.mission_waypoints:
            self.get_logger().warn('실행할 코드 WP가 없습니다.')
            return False
        self.mission_active = True
        self.mission_completed = False
        self.mission_failed = False
        return self._activate_mission_waypoint(0)

    def cancel_code_mission(self, reason):
        """Permanently cancel the code mission when an external goal arrives."""
        if self.mission_active or self.mission_waypoint_index >= 0:
            self.get_logger().info(f'코드 WP 임무 취소: {reason}')
        self.mission_active = False
        self.mission_completed = False
        self.mission_failed = False
        self.mission_waypoint_index = -1

    def _advance_code_mission(self):
        """Stop at the reached WP and queue the next code-defined goal."""
        reached = self.mission_waypoint_index
        self.get_logger().info(
            f'코드 임무 WP {reached + 1}/{len(self.mission_waypoints)} 도착')
        next_index = reached + 1
        if next_index < len(self.mission_waypoints):
            return self._activate_mission_waypoint(next_index)

        self.mission_active = False
        self.mission_completed = True
        self.mission_failed = False
        self.completed = True
        self.requested_goal = None
        self.replan_requested = False
        self.goal_source = None
        self.planner_state = 'MISSION_COMPLETED'
        self.reset_avoidance_state()
        self.get_logger().info(
            f'코드 WP 임무 완료: {len(self.mission_waypoints)}/'
            f'{len(self.mission_waypoints)}, 정지합니다.')
        return True

    def goal_callback(self, message):
        """Cancel the code mission and request an external A* goal."""
        self.cancel_code_mission('외부 /goal_pose 수신')
        map_frame = self.grid_spec.frame_id if self.grid_spec else 'map'
        if message.header.frame_id and message.header.frame_id != map_frame:
            self.requested_goal = None
            self.requested_goal_yaw = 0.0
            self.goal_source = None
            self.clear_path(
                f'목표 frame={message.header.frame_id} 거부: '
                f'{map_frame} 좌표가 필요합니다.',
                state='NO_PATH')
            return
        x = float(message.pose.position.x)
        y = float(message.pose.position.y)
        if not math.isfinite(x) or not math.isfinite(y):
            self.requested_goal = None
            self.requested_goal_yaw = 0.0
            self.goal_source = None
            self.clear_path('목표 좌표가 유한하지 않아 거부합니다.', state='NO_PATH')
            return
        try:
            yaw = quaternion_yaw(message.pose.orientation)
        except ValueError:
            yaw = 0.0

        self._activate_goal(x, y, yaw, source='EXTERNAL')
        self.get_logger().info(f'새 목표 수신: x={x:.3f}, y={y:.3f}')

    def scan_callback(self, message):
        """Store the latest scan; validity is checked before motion."""
        self.latest_scan = message
        self.last_scan_time = self.get_clock().now()

    def scan_metrics(self, alpha):
        """Return angle-aware LiDAR metrics, or None for an invalid scan."""
        minimum_hits = int(self.get_parameter('scan_min_hits').value)
        _, usable = scan_points(self.latest_scan)
        if usable < max(12, minimum_hits * 4):
            return None
        sectors = {
            'front_left': (math.radians(45.0), math.radians(25.0), 0.15),
            'front_right': (math.radians(-45.0), math.radians(25.0), 0.15),
            'left': (math.pi / 2.0, math.radians(30.0), 0.15),
            'right': (-math.pi / 2.0, math.radians(30.0), 0.15),
            'goal': (alpha, math.radians(12.0), 0.05),
        }
        distances = {}
        for name, (center, width, percentile) in sectors.items():
            distance, count = scan_sector_distance(
                self.latest_scan, center, width, percentile)
            if name != 'goal' and count < minimum_hits:
                return None
            distances[name] = distance
            distances[f'{name}_count'] = count
        distances['front'] = front_body_clearance(
            self.latest_scan,
            float(self.get_parameter('front_corridor_half_width').value),
            minimum_hits,
        )
        distances['radial'] = minimum_radial_clearance(self.latest_scan)
        return distances

    def publish_stop(self, message=None):
        """Publish an immediate zero Twist and reset command history."""
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.publisher_.publish(Twist())
        if message:
            self.get_logger().info(message)

    def publish_motion(
            self, desired_linear, desired_angular, dt,
            immediate_linear_stop=False):
        """Apply acceleration and angular limits before publishing."""
        if immediate_linear_stop and desired_linear <= 0.0:
            self.current_linear = 0.0
        else:
            self.current_linear = rate_limit(
                self.current_linear,
                desired_linear,
                float(self.get_parameter('linear_acceleration_limit').value),
                dt,
            )
        self.current_angular = rate_limit(
            self.current_angular,
            desired_angular,
            float(self.get_parameter('angular_acceleration_limit').value),
            dt,
        )
        self.current_angular = clamp(
            self.current_angular,
            float(self.get_parameter('max_angular_velocity').value),
        )
        command = Twist()
        command.linear.x = self.current_linear
        command.angular.z = self.current_angular
        self.publisher_.publish(command)

    def publish_path(self):
        """Publish current route for RViz and the existing control GUI."""
        message = Path()
        message.header.frame_id = (
            self.grid_spec.frame_id if self.grid_spec else 'map')
        message.header.stamp = self.get_clock().now().to_msg()
        for index, (x, y) in enumerate(self.path):
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            if index + 1 < len(self.path):
                next_x, next_y = self.path[index + 1]
                yaw = math.atan2(next_y - y, next_x - x)
            else:
                yaw = self.requested_goal_yaw
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            message.poses.append(pose)
        self.path_publisher.publish(message)

    def clear_path(self, reason, state='NO_PATH'):
        """Discard stale planning output, reset avoidance, and stop."""
        mission_failure = self.mission_active and state == 'NO_PATH'
        self.path = []
        self.path_cells = []
        self.current_waypoint_index = 0
        self.segment_start = None
        self.active_goal = None
        self.completed = False
        self.heading_align_active = False
        self.replan_requested = False
        self.reset_avoidance_state()
        if mission_failure:
            self.mission_active = False
            self.mission_completed = False
            self.mission_failed = True
            self.requested_goal = None
            self.requested_goal_yaw = 0.0
            self.goal_source = None
            self.planner_state = 'MISSION_FAILED'
        else:
            if not self.mission_active:
                self.goal_source = None
            self.planner_state = state
        self.publish_path()
        self.publish_stop()
        self.get_logger().error(reason)
        if mission_failure:
            self.get_logger().error(
                f'코드 WP 임무 중단: '
                f'{self.mission_waypoint_index + 1}/'
                f'{len(self.mission_waypoints)} 경로 생성 실패')

    def plan_path(self, reason):
        """Plan from the current AMCL pose to the requested goal."""
        if self.costmap_dirty and not self.rebuild_costmap():
            return False
        if self.blocked is None or self.grid_spec is None:
            self.planner_state = 'WAIT_MAP'
            self.publish_stop()
            return False
        if not self.is_localized:
            self.planner_state = 'WAIT_POSE'
            self.publish_stop()
            return False
        if self.requested_goal is None:
            self.planner_state = 'WAIT_GOAL'
            self.publish_stop()
            return False

        self.planner_state = 'PLANNING'
        self.publish_stop()
        spec = self.grid_spec
        start_cell = world_to_grid(self.current_x, self.current_y, spec)
        goal_cell = world_to_grid(
            self.requested_goal[0], self.requested_goal[1], spec)
        if start_cell is None:
            self.clear_path('현재 위치가 지도 범위 밖입니다.')
            return False
        if goal_cell is None:
            self.clear_path('목표 위치가 지도 범위 밖입니다.')
            return False

        search_radius = float(
            self.get_parameter('nearest_free_radius').value)
        if self.blocked[start_cell[1] * spec.width + start_cell[0]]:
            self.clear_path(
                '현재 로봇 위치가 팽창 장애물 안에 있어 출발 경로를 만들지 않습니다.')
            return False
        free_start = start_cell
        free_goal = nearest_free_cell(
            self.blocked, spec, goal_cell, search_radius)
        if free_start is None:
            self.clear_path('현재 위치 주변에 자유 셀이 없습니다.')
            return False
        if free_goal is None:
            self.clear_path('목표 주변에 자유 셀이 없습니다.')
            return False

        cells = astar_path(
            self.blocked,
            spec.width,
            spec.height,
            free_start,
            free_goal,
            allow_diagonal=bool(
                self.get_parameter('allow_diagonal').value),
            max_expansions=int(
                self.get_parameter('max_expansions').value),
        )
        if not cells:
            self.clear_path('A*가 목표까지 연결된 경로를 찾지 못했습니다.')
            return False
        simplified = simplify_grid_path(
            cells, self.blocked, spec.width, spec.height)
        world_path = [
            grid_to_world(grid_x, grid_y, spec)
            for grid_x, grid_y in simplified
        ]
        if free_start == start_cell:
            world_path[0] = (self.current_x, self.current_y)
        if free_goal == goal_cell:
            world_path[-1] = self.requested_goal

        self.path_cells = simplified
        self.path = world_path
        self.current_waypoint_index = 1 if len(world_path) > 1 else 0
        self.segment_start = world_path[0]
        self.active_goal = world_path[-1]
        self.completed = False
        self.heading_align_active = False
        self.reset_avoidance_state()
        self.last_plan_time = self.get_clock().now()
        self.plan_count += 1
        self.replan_requested = False
        self.replan_after_avoidance = False
        self.planner_state = 'READY'
        self.publish_path()
        self.get_logger().info(
            f'A* 경로 생성({reason}): 원본 {len(cells)}셀 → '
            f'추종점 {len(world_path)}개, 계획 #{self.plan_count}')
        if free_goal != goal_cell:
            adjusted_x, adjusted_y = world_path[-1]
            self.get_logger().warn(
                f'막힌 목표를 가까운 자유 위치 '
                f'({adjusted_x:.2f}, {adjusted_y:.2f})로 조정했습니다.')
        return True

    def reset_avoidance_state(self, state='TRACK'):
        """Clear state that belongs to a previous obstacle."""
        self.avoidance_state = state
        self.avoidance_active = state in ('TURN', 'BYPASS')
        self.turn_direction = 0.0
        self.obstacle_clear_since = None
        self.avoidance_started = None
        self.avoidance_hit_distance = math.inf
        self.turn_start_yaw = self.current_yaw

    def set_avoidance_state(self, state, reason):
        """Change local avoidance state and log the transition."""
        if state != self.avoidance_state:
            self.avoidance_state = state
            self.avoidance_active = state in ('TURN', 'BYPASS')
            self.get_logger().info(f'회피 상태 -> {state}: {reason}')

    def advance_waypoints(self):
        """Advance reached or safely passed A* tracking points."""
        if not self.path:
            return False
        tolerance = float(
            self.get_parameter('path_point_tolerance').value)
        while self.current_waypoint_index < len(self.path) - 1:
            goal = self.path[self.current_waypoint_index]
            distance = math.hypot(
                goal[0] - self.current_x, goal[1] - self.current_y)
            vector_x = goal[0] - self.segment_start[0]
            vector_y = goal[1] - self.segment_start[1]
            length_squared = vector_x * vector_x + vector_y * vector_y
            passed = False
            if length_squared > 1e-12:
                progress = (
                    (self.current_x - self.segment_start[0]) * vector_x
                    + (self.current_y - self.segment_start[1]) * vector_y
                ) / length_squared
                passed = progress >= 1.0
            if distance > tolerance and not passed:
                break
            self.segment_start = goal
            self.current_waypoint_index += 1

        if self.active_goal is None:
            return False
        goal_distance = math.hypot(
            self.active_goal[0] - self.current_x,
            self.active_goal[1] - self.current_y,
        )
        if goal_distance <= float(self.get_parameter('goal_tolerance').value):
            if self.goal_source == 'MISSION' and self.mission_active:
                return self._advance_code_mission()
            if not self.completed:
                self.completed = True
                self.planner_state = 'COMPLETED'
                self.requested_goal = None
                self.replan_requested = False
                self.goal_source = None
                self.reset_avoidance_state()
                self.get_logger().info('최종 목표 도착: 정지합니다.')
            return True
        return False

    def path_geometry(self):
        """Return final-goal distance and Pure Pursuit look-ahead error."""
        goal_distance = math.hypot(
            self.active_goal[0] - self.current_x,
            self.active_goal[1] - self.current_y,
        )
        target_x, target_y = select_lookahead_point(
            self.current_x,
            self.current_y,
            self.segment_start,
            self.path,
            self.current_waypoint_index,
            float(self.get_parameter('lookahead_distance').value),
        )
        delta_x = target_x - self.current_x
        delta_y = target_y - self.current_y
        target_distance = math.hypot(delta_x, delta_y)
        alpha = normalize_angle(
            math.atan2(delta_y, delta_x) - self.current_yaw)
        return goal_distance, target_distance, alpha

    def path_command(self, goal_distance, target_distance, alpha):
        """Calculate a smoothed Pure Pursuit command."""
        linear_velocity = float(self.get_parameter('linear_velocity').value)
        max_angular = float(
            self.get_parameter('max_angular_velocity').value)
        enter_angle = float(
            self.get_parameter('heading_rotate_enter').value)
        exit_angle = float(
            self.get_parameter('heading_rotate_exit').value)
        if self.heading_align_active:
            if abs(alpha) <= exit_angle:
                self.heading_align_active = False
        elif abs(alpha) >= enter_angle:
            self.heading_align_active = True
        if linear_velocity <= 1e-9:
            return 0.0, 0.0, True
        if self.heading_align_active:
            return (
                0.0,
                clamp(
                    float(self.get_parameter('heading_gain').value) * alpha,
                    max_angular),
                True,
            )
        tolerance = float(self.get_parameter('goal_tolerance').value)
        acceleration = float(
            self.get_parameter('linear_acceleration_limit').value)
        stopping_speed = math.sqrt(
            2.0 * acceleration * max(0.0, goal_distance - tolerance))
        desired_linear = min(linear_velocity, stopping_speed) * max(
            0.0, math.cos(alpha))
        curvature = (
            0.0 if target_distance <= 1e-6
            else 2.0 * math.sin(alpha) / target_distance
        )
        return (
            desired_linear,
            clamp(desired_linear * curvature, max_angular),
            False,
        )

    def avoidance_elapsed(self, now):
        if self.avoidance_started is None:
            return 0.0
        return max(
            0.0,
            (now - self.avoidance_started).nanoseconds / 1e9,
        )

    def start_avoidance(self, metrics, goal_distance, alpha, now):
        """Latch one safe turn direction when entering avoidance."""
        if metrics['radial'] <= float(
                self.get_parameter('spin_clearance_radius').value):
            self.reset_avoidance_state('BLOCKED')
            self.get_logger().error(
                f'제자리 선회 공간 부족 ({metrics["radial"]:.2f}m): 정지합니다.')
            return False
        left_score = min(metrics['front_left'], metrics['left'])
        right_score = min(metrics['front_right'], metrics['right'])
        difference = left_score - right_score
        deadband = float(
            self.get_parameter('turn_direction_deadband').value)
        if difference > deadband:
            self.turn_direction = 1.0
        elif difference < -deadband:
            self.turn_direction = -1.0
        else:
            self.turn_direction = 1.0 if alpha >= 0.0 else -1.0
        self.avoidance_started = now
        self.avoidance_hit_distance = goal_distance
        self.turn_start_yaw = self.current_yaw
        self.obstacle_clear_since = None
        self.replan_after_avoidance = True
        direction = 'LEFT' if self.turn_direction > 0.0 else 'RIGHT'
        self.set_avoidance_state(
            'TURN', f'전방 {metrics["front"]:.2f}m, {direction} 방향 고정')
        return True

    def turn_command(self, metrics, now, dt):
        """Turn in the latched direction until the corridor is clear."""
        if metrics['radial'] <= float(
                self.get_parameter('spin_clearance_radius').value):
            self.set_avoidance_state('BLOCKED', '선회 반경 내 장애물')
            self.publish_stop()
            return
        if self.avoidance_elapsed(now) > float(
                self.get_parameter('avoidance_max_duration').value):
            self.set_avoidance_state('BLOCKED', '회피 제한 시간 초과')
            self.publish_stop()
            return
        clear_distance = (
            float(self.get_parameter('obs_threshold').value)
            + float(self.get_parameter('avoidance_clear_margin').value)
        )
        turned = abs(normalize_angle(
            self.current_yaw - self.turn_start_yaw))
        if (
                metrics['front'] >= clear_distance
                and turned >= float(
                    self.get_parameter('minimum_turn_angle').value)):
            if self.obstacle_clear_since is None:
                self.obstacle_clear_since = self.last_scan_time
            clear_duration = (
                self.last_scan_time - self.obstacle_clear_since
            ).nanoseconds / 1e9
            if clear_duration >= float(
                    self.get_parameter('avoidance_clear_hold_time').value):
                self.obstacle_clear_since = None
                self.set_avoidance_state('BYPASS', '전방 통로 연속 확인')
                self.bypass_command(metrics, now, dt)
                return
        else:
            self.obstacle_clear_since = None
        self.publish_motion(
            0.0,
            self.turn_direction * float(
                self.get_parameter('avoidance_angular_velocity').value),
            dt,
            immediate_linear_stop=True,
        )

    def bypass_command(self, metrics, now, dt):
        """Move past an obstacle while maintaining side clearance."""
        if self.avoidance_elapsed(now) > float(
                self.get_parameter('avoidance_max_duration').value):
            self.set_avoidance_state('BLOCKED', '우회 제한 시간 초과')
            self.publish_stop()
            return
        if metrics['front'] <= float(
                self.get_parameter('obs_threshold').value):
            self.turn_start_yaw = self.current_yaw
            self.obstacle_clear_since = None
            self.set_avoidance_state('TURN', '우회 중 전방 재차단')
            self.turn_command(metrics, now, dt)
            return
        side_distance = (
            metrics['right'] if self.turn_direction > 0.0
            else metrics['left'])
        target = float(self.get_parameter('side_clearance').value)
        bounded_side = min(side_distance, 2.0 * target)
        angular = (
            -self.turn_direction
            * float(self.get_parameter('wall_follow_gain').value)
            * (bounded_side - target)
        )
        self.publish_motion(
            float(self.get_parameter('avoidance_linear_velocity').value),
            clamp(
                angular,
                float(self.get_parameter('max_angular_velocity').value)),
            dt,
        )

    def can_rejoin(self, metrics, goal_distance, alpha, now):
        """Return true after measurable progress and a clear goal ray."""
        if self.avoidance_elapsed(now) < float(
                self.get_parameter('avoidance_min_duration').value):
            return False
        if self.avoidance_hit_distance - goal_distance < float(
                self.get_parameter('avoidance_progress_distance').value):
            return False
        if abs(alpha) > float(
                self.get_parameter('rejoin_heading_tolerance').value):
            return False
        clear_distance = (
            float(self.get_parameter('obs_threshold').value)
            + float(self.get_parameter('avoidance_clear_margin').value)
        )
        return (
            metrics['front'] > clear_distance
            and metrics['goal_count'] >= int(
                self.get_parameter('scan_min_hits').value)
            and metrics['goal'] > clear_distance
        )

    def _maybe_replan_for_deviation(self, now):
        """Replan only in TRACK and report whether a replan was attempted."""
        if self.avoidance_state != 'TRACK' or not self.path:
            return False
        cooldown = float(self.get_parameter('replan_cooldown').value)
        since_plan = (
            math.inf if self.last_plan_time is None
            else (now - self.last_plan_time).nanoseconds / 1e9
        )
        deviation = distance_to_path(
            (self.current_x, self.current_y),
            self.path,
            self.current_waypoint_index,
            self.segment_start,
        )
        if (
                deviation > float(
                    self.get_parameter('replan_distance').value)
                and since_plan >= cooldown):
            # plan_path() can fail after publishing a zero command.  This
            # control tick must still end here; otherwise the caller could
            # overwrite that stop with a command computed from the old path.
            self.plan_path(f'경로 이탈 {deviation:.2f}m')
            return True
        return False

    def control_loop(self):
        """Plan when necessary, then publish one safe velocity command."""
        now = self.get_clock().now()
        period = float(self.get_parameter('control_period').value)
        dt = (now - self.last_control_time).nanoseconds / 1e9
        if not math.isfinite(dt) or dt <= 0.0 or dt > 1.0:
            dt = period
        self.last_control_time = now

        avoiding = self.avoidance_state in ('TURN', 'BYPASS')
        if self.costmap_dirty and not avoiding:
            self.rebuild_costmap()
        if (
                self.replan_requested
                and self.requested_goal is not None
                and not avoiding):
            self.plan_path('새 목표/지도/파라미터')

        enabled = bool(self.get_parameter('drive_enabled').value)
        if enabled != self.last_drive_enabled:
            if enabled:
                self.get_logger().info('자동주행 시작: drive_enabled=true')
            else:
                self.reset_avoidance_state()
                self.heading_align_active = False
                self.publish_stop('자동주행 잠금: drive_enabled=false')
            self.last_drive_enabled = enabled
        if not enabled:
            return

        # A completed manual goal or code mission stays latched at zero until
        # a new valid /goal_pose arrives or the node is restarted.
        if self.completed:
            self.publish_stop()
            return

        if not self.is_localized or self.last_pose_time is None:
            self.planner_state = 'WAIT_POSE'
            self.publish_stop()
            self.get_logger().warn(
                'AMCL 위치를 기다리는 중입니다.', throttle_duration_sec=2.0)
            return
        pose_age = (now - self.last_pose_time).nanoseconds / 1e9
        if (
                not math.isfinite(pose_age)
                or pose_age < 0.0
                or pose_age > float(self.get_parameter('pose_timeout').value)):
            self.reset_avoidance_state('WAIT_POSE')
            self.publish_stop()
            self.get_logger().error(
                f'AMCL 위치가 {pose_age:.1f}s 동안 갱신되지 않아 정지합니다.',
                throttle_duration_sec=2.0)
            return
        if not self.path or self.active_goal is None:
            self.publish_stop()
            self.get_logger().warn(
                f'유효한 A* 경로를 기다리는 중입니다 ({self.planner_state}).',
                throttle_duration_sec=2.0)
            return
        if self.advance_waypoints():
            self.publish_stop()
            return

        goal_distance, target_distance, alpha = self.path_geometry()

        if self.latest_scan is None or self.last_scan_time is None:
            self.reset_avoidance_state('WAIT_SCAN')
            self.publish_stop()
            self.get_logger().warn(
                'LiDAR 데이터를 기다리는 중입니다.', throttle_duration_sec=2.0)
            return
        scan_age = (now - self.last_scan_time).nanoseconds / 1e9
        if (
                not math.isfinite(scan_age)
                or scan_age < 0.0
                or scan_age > float(self.get_parameter('scan_timeout').value)):
            self.reset_avoidance_state('WAIT_SCAN')
            self.publish_stop()
            self.get_logger().error(
                f'LiDAR가 {scan_age:.1f}s 동안 갱신되지 않아 정지합니다.',
                throttle_duration_sec=2.0)
            return
        metrics = self.scan_metrics(alpha)
        if metrics is None:
            self.reset_avoidance_state('WAIT_SCAN')
            self.publish_stop()
            self.get_logger().error(
                '유효한 LiDAR 표본이 부족해 정지합니다.',
                throttle_duration_sec=2.0)
            return
        self.front_dist = metrics['front']
        self.left_dist = metrics['left']
        self.right_dist = metrics['right']
        if self.avoidance_state in ('WAIT_SCAN', 'WAIT_POSE'):
            self.reset_avoidance_state()
            self.get_logger().info('센서 입력 회복: TRACK 재개')

        if self.avoidance_state == 'BLOCKED':
            self.publish_stop()
            return
        if self.avoidance_state == 'TURN':
            self.turn_command(metrics, now, dt)
            return
        if self.avoidance_state == 'BYPASS':
            if self.can_rejoin(metrics, goal_distance, alpha, now):
                self.reset_avoidance_state()
                if (
                        self.replan_after_avoidance
                        or self.replan_requested
                        or self.costmap_dirty):
                    self.replan_after_avoidance = False
                    self.plan_path('장애물 우회 후 현재 위치')
                    return
                self.get_logger().info('우회 진전 확인: TRACK 복귀')
            else:
                self.bypass_command(metrics, now, dt)
                return

        if self._maybe_replan_for_deviation(now):
            return
        if metrics['front'] <= float(
                self.get_parameter('obs_threshold').value):
            if self.start_avoidance(metrics, goal_distance, alpha, now):
                self.turn_command(metrics, now, dt)
            else:
                self.publish_stop()
            return

        desired_linear, desired_angular, rotate_only = self.path_command(
            goal_distance, target_distance, alpha)
        if (
                not rotate_only
                and not pure_pursuit_arc_is_free(
                    self.blocked,
                    self.grid_spec,
                    self.current_x,
                    self.current_y,
                    self.current_yaw,
                    target_distance,
                    alpha,
                )):
            # 경로 선분은 자유 공간이어도 Pure Pursuit 원호가 코너 안쪽을
            # 자를 수 있다. 먼저 목표 방향으로 정렬해 원호를 곧게 만든다.
            desired_linear = 0.0
            desired_angular = clamp(
                float(self.get_parameter('heading_gain').value) * alpha,
                float(self.get_parameter('max_angular_velocity').value),
            )
            rotate_only = True
            self.get_logger().warn(
                '예측 주행 원호가 팽창 장애물과 겹쳐 제자리 정렬합니다.',
                throttle_duration_sec=1.0)
        if (
                rotate_only
                and metrics['radial'] <= float(
                    self.get_parameter('spin_clearance_radius').value)):
            self.set_avoidance_state('BLOCKED', '제자리 정렬 선회 공간 부족')
            self.publish_stop()
            return
        self.publish_motion(
            desired_linear,
            desired_angular,
            dt,
            immediate_linear_stop=rotate_only,
        )
        self.get_logger().info(
            f'A* WP {self.current_waypoint_index + 1}/{len(self.path)} '
            f'goal={goal_distance:.2f}m alpha={alpha:.2f}rad '
            f'v={self.current_linear:.2f}m/s '
            f'w={self.current_angular:.2f}rad/s '
            f'state={self.avoidance_state}',
            throttle_duration_sec=1.0,
        )


def main(args=None):
    """Run the integrated planner/controller until interrupted."""
    rclpy.init(args=args)
    node = PurePursuitObstacleV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_stop()
        except (KeyboardInterrupt, RuntimeError):
            pass
        try:
            node.destroy_node()
        except (KeyboardInterrupt, RuntimeError):
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RuntimeError):
            pass


if __name__ == '__main__':
    main()
