import math
from dataclasses import dataclass

from action_msgs.msg import GoalStatus
from nav2_msgs.action import Spin
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


@dataclass
class RobotPose:
    x: float
    y: float
    yaw: float


def quaternion_to_yaw(rotation) -> float:
    siny_cosp = 2.0 * (rotation.w * rotation.z + rotation.x * rotation.y)
    cosy_cosp = 1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class MovementObj:
    def __init__(self, node):
        self.node = node

        self.node.declare_parameter('global_frame', 'map')
        self.node.declare_parameter('robot_frame', 'base_link')
        self.node.declare_parameter('spin_action_name', 'spin')
        self.node.declare_parameter('spin_goal_interval', 1.0)
        self.node.declare_parameter('spin_time_allowance', 30.0)
        self.node.declare_parameter('person_yaw_deadband', 0.08)
        self.node.declare_parameter('invert_person_yaw', False)

        self.global_frame = self.node.get_parameter('global_frame').value
        self.robot_frame = self.node.get_parameter('robot_frame').value
        self.spin_action_name = self.node.get_parameter('spin_action_name').value
        self.spin_goal_interval = self.node.get_parameter('spin_goal_interval').value
        self.spin_time_allowance = self.node.get_parameter('spin_time_allowance').value
        self.person_yaw_deadband = self.node.get_parameter('person_yaw_deadband').value
        self.invert_person_yaw = self.node.get_parameter('invert_person_yaw').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.spin_client = ActionClient(self.node, Spin, self.spin_action_name)
        self.pending_spin_timer = self.node.create_timer(0.2, self._drain_pending_spin)

        self.start_pose = None
        self.last_spin_time = None
        self.last_spin_yaw = None
        self.spin_active = False
        self.pending_track_yaw = None
        self.reset_requested = False
        self.saw_person_last_update = False

    def update_with_person_position(self, person_position):
        current_pose = self._get_current_pose()
        if current_pose is None:
            return

        self._capture_start_pose(current_pose)

        person_yaw = math.atan2(person_position[0], person_position[2])
        if self.invert_person_yaw:
            person_yaw *= -1.0

        self.reset_requested = False

        if abs(person_yaw) < self.person_yaw_deadband:
            self.saw_person_last_update = True
            self.pending_track_yaw = None
            return

        self._request_track_spin(normalize_angle(-person_yaw))
        self.saw_person_last_update = True

    def update_with_empty_frame(self):
        current_pose = self._get_current_pose()
        if current_pose is None:
            return

        self._capture_start_pose(current_pose)

        if not self.saw_person_last_update:
            return

        self.pending_track_yaw = None
        self.reset_requested = True
        self._try_send_reset_spin(current_pose)

    def _capture_start_pose(self, current_pose):
        if self.start_pose is None:
            self.start_pose = current_pose
            self.node.get_logger().info(
                f"Saved startup pose in {self.global_frame}: "
                f"x={current_pose.x:.2f}, y={current_pose.y:.2f}, yaw={current_pose.yaw:.2f}"
            )

    def _get_current_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
        except TransformException as exc:
            self.node.get_logger().warn(
                f"Cannot read robot pose from {self.global_frame}->{self.robot_frame}: {exc}"
            )
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return RobotPose(
            x=translation.x,
            y=translation.y,
            yaw=quaternion_to_yaw(rotation),
        )

    def _request_track_spin(self, target_yaw):
        if self.spin_active:
            self.pending_track_yaw = target_yaw
            return

        if not self._send_spin_goal(target_yaw, 'track'):
            self.pending_track_yaw = target_yaw

    def _try_send_reset_spin(self, current_pose=None):
        if self.spin_active or self.start_pose is None:
            return

        if current_pose is None:
            current_pose = self._get_current_pose()
            if current_pose is None:
                return

        return_yaw = normalize_angle(self.start_pose.yaw - current_pose.yaw)
        if abs(return_yaw) < self.person_yaw_deadband:
            self.reset_requested = False
            self.saw_person_last_update = False
            return

        self._send_spin_goal(return_yaw, 'reset')

    def _send_spin_goal(self, target_yaw, purpose, force=False):
        now = self.node.get_clock().now()
        if not force and self.last_spin_time is not None:
            elapsed = (now - self.last_spin_time).nanoseconds / 1e9
            if elapsed < self.spin_goal_interval:
                return False

        if not self.spin_client.server_is_ready():
            self.node.get_logger().warn(f"Nav2 spin action server '{self.spin_action_name}' is not ready")
            return False

        goal_msg = Spin.Goal()
        goal_msg.target_yaw = float(target_yaw)
        goal_msg.time_allowance = Duration(seconds=self.spin_time_allowance).to_msg()

        self.spin_active = True
        send_future = self.spin_client.send_goal_async(goal_msg)
        send_future.add_done_callback(
            lambda future: self._spin_goal_response_callback(future, purpose)
        )
        self.last_spin_time = now
        self.last_spin_yaw = target_yaw
        return True

    def _spin_goal_response_callback(self, future, purpose):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.node.get_logger().warn(f"Failed to send Nav2 spin goal: {exc}")
            self.spin_active = False
            self._drain_pending_spin()
            return

        if not goal_handle.accepted:
            self.node.get_logger().warn("Nav2 spin goal was rejected")
            self.spin_active = False
            self._drain_pending_spin()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result: self._spin_result_callback(result, purpose)
        )

    def _spin_result_callback(self, future, purpose):
        try:
            result = future.result()
            status = result.status
        except Exception as exc:
            self.node.get_logger().warn(f"Failed to get Nav2 spin result: {exc}")
            status = GoalStatus.STATUS_UNKNOWN

        self.spin_active = False

        if status != GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().warn(f"Nav2 spin ended with status {status}")

        if purpose == 'reset' and status == GoalStatus.STATUS_SUCCEEDED:
            self._finish_reset_if_at_start_yaw()

        self._drain_pending_spin()

    def _finish_reset_if_at_start_yaw(self):
        if self.start_pose is None:
            return

        current_pose = self._get_current_pose()
        if current_pose is None:
            return

        return_yaw = normalize_angle(self.start_pose.yaw - current_pose.yaw)
        if abs(return_yaw) < self.person_yaw_deadband:
            self.reset_requested = False
            self.saw_person_last_update = False

    def _drain_pending_spin(self):
        if self.spin_active:
            return

        if self.reset_requested:
            self._try_send_reset_spin()
            return

        if self.pending_track_yaw is not None:
            target_yaw = self.pending_track_yaw
            self.pending_track_yaw = None
            self._request_track_spin(target_yaw)