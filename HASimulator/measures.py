
from typing import Any, List, Union, Tuple, Optional

import numpy as np

try:
    from habitat.config import Config
except ImportError:
    from yacs.config import CfgNode as Config
from habitat.core.embodied_task import Action, EmbodiedTask, Measure
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.core.utils import try_cv2_import
from numpy import ndarray

cv2 = try_cv2_import()


def euclidean_distance(
    pos_a: Union[List[float], ndarray], pos_b: Union[List[float], ndarray]
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)


@registry.register_measure
class DistanceToHuman(Measure):
    """The measure calculates a distance towards the human."""

    cls_uuid: str = "distance_to_human"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = {}
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, *args: Any, **kwargs: Any
    ):
        state = self._sim.get_agent_state()
        current_position = state.position
        current_rotation = state.rotation
        human_positions = getattr(self._sim, "_human_positions", None) or {}
        for name, (position, rotation) in human_positions.items():
            distance_to_human = euclidean_distance(
                    current_position,
                    position,
                )
            import quaternion
            import math

            agent = self._sim.get_agent(0)
            action = agent.agent_config.action_space[1]

            if agent.controls.is_body_action(action.name):
                did_collide = agent.controls.action(
                    agent.scene_node, action.name, action.actuation, apply_filter=True
                )
            else:
                for v in agent._sensors.values():
                    habitat_sim.errors.assert_obj_valid(v)
                    agent.controls.action(
                        v.object, action.name, action.actuation, apply_filter=False
                    )
            
            move_position = self._sim.get_agent_state().position
            
            move = move_position - current_position
            human_to_agent = position - current_position
            move = move / (np.sqrt((move**2).sum()) + 1e-8)
            human_to_agent = human_to_agent / (np.sqrt((human_to_agent**2).sum()) + 1e-8)
            cos = (move*human_to_agent).sum()
            assert cos <=1 and cos >= -1

            theta = math.degrees(math.acos(cos))

            agent.set_state(state)
            # if did_collide:
            #     print(move_position, current_position)
            #     print(euclidean_distance(move_position, current_position))
            #     raise

            self._metric[name] = (distance_to_human, theta)

        self._previous_position = current_position


@registry.register_measure
class DistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "distance_to_goal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = None
        if self._config.DISTANCE_TO == "VIEW_POINTS":
            self._episode_view_points = [
                view_point.agent_state.position
                for goal in episode.goals
                for view_point in goal.view_points
            ]
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, *args: Any, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            if self._config.DISTANCE_TO == "POINT":
                distance_to_target = euclidean_distance(
                    current_position,
                    [goal.position for goal in episode.goals],
                )
            elif self._config.DISTANCE_TO == "VIEW_POINTS":
                distance_to_target = euclidean_distance(
                    current_position, self._episode_view_points
                )
            else:
                logger.error(
                    f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
                )

            self._previous_position = current_position
            self._metric = distance_to_target


@registry.register_measure
class CollisionsDetail(Measure):
    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._metric = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "collisions_detail"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = None

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        if self._metric is None:
            self._metric = {"count": 0, "is_collision": False, 'steps': []}
        self._metric["is_collision"] = False
        if self._sim.previous_step_collided:
            self._metric["count"] += 1
            self._metric["is_collision"] = True
            self._metric["steps"].append(True)
        else:
            self._metric["steps"].append(False)