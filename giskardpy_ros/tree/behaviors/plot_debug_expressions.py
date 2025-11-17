from threading import Lock

import numpy as np

from giskardpy.model.trajectory import Trajectory
from giskardpy_ros.tree.behaviors.plot_trajectory import PlotTrajectory
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_state import WorldState

plot_lock = Lock()


class PlotDebugExpressions(PlotTrajectory):

    def __init__(self, name, wait=True, normalize_position: bool = False, **kwargs):
        super().__init__(
            name=name, normalize_position=normalize_position, wait=wait, **kwargs
        )
        # self.path_to_data_folder += 'debug_expressions/'
        # create_path(self.path_to_data_folder)

    def split_traj(self, traj) -> Trajectory:
        """
        if the trajectory has entries that describe vectors or matrices, split them into separate entries
        """
        new_traj = Trajectory()
        for js in traj:
            new_js = WorldState()
            for name, js_ in js.items():
                if isinstance(js_[0], np.ndarray):
                    if len(js_.position.shape) == 1:
                        for x in range(js_.position.shape[0]):
                            tmp_name = PrefixedName(f"{name}|{x}")
                            new_js[tmp_name].position = js_.position[x]
                            new_js[tmp_name].velocity = js_.velocity[x]
                    else:
                        for x in range(js_.position.shape[0]):
                            for y in range(js_.position.shape[1]):
                                tmp_name = PrefixedName(f"{name}|{x}_{y}")
                                new_js[tmp_name].position = js_.position[x, y]
                                new_js[tmp_name].velocity = js_.velocity[x, y]
                else:
                    new_js[name] = js_
                new_traj.append(new_js)

        return new_traj

    def plot(self):
        raise NotImplementedError("needs fixing")
        # trajectory = god_map.debug_expression_manager.raw_traj_to_traj(
        #     GiskardBlackboard().executor.qp_controller.config.control_dt
        #     or GiskardBlackboard().executor.qp_controller.config.mpc_dt
        # )
        # if trajectory and len(trajectory) > 0:
        #     sample_period = GiskardBlackboard().executor.qp_controller.config.mpc_dt
        #     traj = self.split_traj(trajectory)
        #     try:
        #         traj.plot_trajectory(
        #             path_to_data_folder=self.path_to_data_folder,
        #             sample_period=sample_period,
        #             file_name=f"debug.pdf",
        #             filter_0_vel=False,
        #             **self.kwargs,
        #         )
        #     except Exception:
        #         traceback.print_exc()
        # get_middleware().logwarn("failed to save debug.pdf")
