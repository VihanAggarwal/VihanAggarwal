"""Fixed facts about the BEHAVIOR-1K 2026 challenge evaluator (v3.9.3-post1/post2 and the 2026/eval branch).

Every value here was read from the evaluator source (OmniGibson/omnigibson/eval/*) or the challenge
metadata; the source location is noted next to each block so it can be re-checked if the evaluator moves.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from importlib import resources

# --------------------------------------------------------------------------------------------------
# Observation keys (eval/r1pro.yaml: robot name "robot_r1"; flattened with "::" by flatten_obs_dict)
# --------------------------------------------------------------------------------------------------
ROBOT_NAME = "robot_r1"
PROPRIO_KEY = f"{ROBOT_NAME}::proprio"
CAM_REL_POSES_KEY = f"{ROBOT_NAME}::cam_rel_poses"
TASK_ID_KEY = "task_id"
ACTION_CHUNK_REQUEST_KEY = "__action_chunk_size__"  # network_utils.ACTION_CHUNK_REQUEST_KEY
RESET_KEY = "reset"

CAMERA_LINKS = {
    "head": "zed_link",
    "left_wrist": "left_realsense_link",
    "right_wrist": "right_realsense_link",
}


def rgb_key(role: str) -> str:
    return f"{ROBOT_NAME}::{ROBOT_NAME}:{CAMERA_LINKS[role]}:Camera:0::rgb"


def depth_key(role: str) -> str:
    return f"{ROBOT_NAME}::{ROBOT_NAME}:{CAMERA_LINKS[role]}:Camera:0::depth_linear"


HEAD_RGB_KEY = rgb_key("head")
LEFT_RGB_KEY = rgb_key("left_wrist")
RIGHT_RGB_KEY = rgb_key("right_wrist")

# Render sizes of the provided wrappers (eval/utils/eval_utils.py HEAD_RESOLUTION / WRIST_RESOLUTION).
FULL_RES = {"head": (720, 720), "left_wrist": (480, 480), "right_wrist": (480, 480)}
DEFAULT_WRAPPER_RES = {"head": (224, 224), "left_wrist": (224, 224), "right_wrist": (224, 224)}

# --------------------------------------------------------------------------------------------------
# 61-D proprioception (eval/utils/eval_utils.py PROPRIOCEPTION_INDICES["R1Pro"]). Identical to the
# 2026 dataset's observation.state. base_qvel is robot-local [vx, vy, wz] since the 07/27 dataset fix.
# --------------------------------------------------------------------------------------------------
PROPRIO_DIM = 61
PROPRIO_INDICES_2026 = {
    "base_qvel": slice(0, 3),
    "arm_left_qpos": slice(3, 10),
    "arm_left_qvel": slice(10, 17),
    "eef_left_pos": slice(17, 20),
    "eef_left_quat": slice(20, 24),
    "gripper_left_qpos": slice(24, 26),
    "gripper_left_qvel": slice(26, 28),
    "arm_right_qpos": slice(28, 35),
    "arm_right_qvel": slice(35, 42),
    "eef_right_pos": slice(42, 45),
    "eef_right_quat": slice(45, 49),
    "gripper_right_qpos": slice(49, 51),
    "gripper_right_qvel": slice(51, 53),
    "trunk_qpos": slice(53, 57),
    "trunk_qvel": slice(57, 61),
}

# --------------------------------------------------------------------------------------------------
# 23-D R1Pro action (default eval/r1pro.yaml, action_normalize: false)
#   base   0:3   HolonomicBaseJointController, normalized [-1,1] -> (+-0.75 m/s, +-0.75 m/s, +-1 rad/s), robot frame
#   torso  3:7   JointController, absolute joint position (rad)
#   L arm  7:14  JointController, absolute joint position (rad)
#   L grip 14    MultiFingerGripperController smooth, [-1 closed, +1 open]
#   R arm  15:22 JointController, absolute joint position (rad)
#   R grip 22    as left
# --------------------------------------------------------------------------------------------------
ACTION_DIM = 23
ACTION_SLICES = {
    "base": slice(0, 3),
    "torso": slice(3, 7),
    "left_arm": slice(7, 14),
    "left_gripper": slice(14, 15),
    "right_arm": slice(15, 22),
    "right_gripper": slice(22, 23),
}
LEFT_GRIPPER_ACTION_IDX = 14
RIGHT_GRIPPER_ACTION_IDX = 22
BASE_ACTION_LIMIT = 1.0  # controller input limits for base are [-1, 1]
GRIPPER_MAX_WIDTH = 0.1  # sum of the two finger joint positions when fully open (m)

# --------------------------------------------------------------------------------------------------
# Instances (eval/utils/eval_utils.py TEST_INSTANCE_IDS, evaluator.resolve_instance_ids)
# --------------------------------------------------------------------------------------------------
TEST_INSTANCE_IDS = tuple(range(301, 341))
REPORTED_INSTANCE_IDS = tuple(range(301, 311))  # public_test indices 0-9: the leaderboard set
HELDOUT_PUBLIC_INSTANCE_IDS = tuple(range(311, 321))  # public_test indices 10-19: allowed pre-submission test set
HIDDEN_INSTANCE_IDS = tuple(range(321, 341))  # hidden_test indices 0-19 (organizers only)
EVAL_TIMEOUT_MULTIPLIER = 1.5
NUM_TASKS = 100
ACTION_HZ = 30


def public_index_to_instance_id(index: int) -> int:
    if not 0 <= index < 20:
        raise ValueError(f"public_test instance index must be in [0, 20), got {index}")
    return 301 + index


# --------------------------------------------------------------------------------------------------
# Tasks (B100_task_misc.csv order == LeRobot chunk index == obs["task_id"])
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskInfo:
    task_id: int
    name: str
    scene: str
    rooms: tuple[str, ...]
    instruction: str
    instruction_comet2025: str | None
    human_mean_len: float
    max_steps: int
    human_distance_traveled: float
    human_left_eef_displacement: float
    human_right_eef_displacement: float
    new_in_2026: bool


@functools.lru_cache(maxsize=1)
def tasks() -> tuple[TaskInfo, ...]:
    raw = json.loads(resources.files("b1k26.data").joinpath("tasks.json").read_text())
    out = tuple(TaskInfo(**{**t, "rooms": tuple(t["rooms"])}) for t in raw["tasks"])
    assert [t.task_id for t in out] == list(range(NUM_TASKS))
    return out


@functools.lru_cache(maxsize=1)
def task_by_name() -> dict[str, TaskInfo]:
    return {t.name: t for t in tasks()}


def task(task_id_or_name: int | str) -> TaskInfo:
    if isinstance(task_id_or_name, str):
        return task_by_name()[task_id_or_name]
    return tasks()[int(task_id_or_name)]
