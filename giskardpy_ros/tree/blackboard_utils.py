from __future__ import annotations
import traceback
import typing
from functools import wraps
from typing import TypeVar, Callable, TYPE_CHECKING, Optional

from py_trees.common import Status, Access

from giskardpy.data_types.exceptions import DontPrintStackTrace

if TYPE_CHECKING:
    from giskardpy_ros.configs.behavior_tree_config import BehaviorTreeConfig
    from giskardpy_ros.configs.giskard import Giskard
    from giskardpy_ros.ros2.ros_msg_visualization import ROSMsgVisualization
    from giskardpy_ros.tree.behaviors.action_server import ActionServerHandler
    from giskardpy_ros.tree.branches.giskard_bt import GiskardBT


class GiskardBlackboard:
    giskard: Giskard
    tree_config: BehaviorTreeConfig
    runtime: float
    move_action_server: ActionServerHandler
    world_action_server: ActionServerHandler
    ros_visualizer: ROSMsgVisualization
    fill_trajectory_velocity_values: bool
    control_loop_max_hz: float
    simulation_max_hz: float
    exception: Optional[Exception]
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state

    @property
    def tree(self):
        return self.tree_config.tree


def raise_to_blackboard(exception):
    GiskardBlackboard().exception = exception


def has_blackboard_exception():
    return get_blackboard_exception() is not None


def get_blackboard_exception() -> Optional[Exception]:
    try:
        return GiskardBlackboard().exception
    except Exception:
        return None


def clear_blackboard_exception():
    raise_to_blackboard(None)


T = TypeVar("T", bound=Callable)


def catch_and_raise_to_blackboard(function=None, *, skip_on_exception=True):
    def decorator(func):
        if skip_on_exception:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if has_blackboard_exception():
                    return Status.FAILURE
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not isinstance(e, DontPrintStackTrace):
                        traceback.print_exc()
                    raise_to_blackboard(e)
                    return Status.FAILURE
            return wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not isinstance(e, DontPrintStackTrace):
                        traceback.print_exc()
                    if not has_blackboard_exception():
                        raise_to_blackboard(e)
                    return Status.FAILURE
            return wrapper

    if function is None:
        return decorator
    else:
        return decorator(function)
