"""Constants for the released CALVIN policy."""
from enum import Enum
class NormalizationType(str, Enum):
    NORMAL = "normal"
    BOUNDS = "bounds"
    BOUNDS_Q99 = "bounds_q99"
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2
ROBOT_PLATFORM = "CALVIN"
NUM_ACTIONS_CHUNK = 10
ACTION_DIM = 7
PROPRIO_DIM = 7
ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99
