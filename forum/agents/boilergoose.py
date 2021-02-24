"""
https://www.kaggle.com/ihelon/hungry-geese-agents-comparison
"""
import dataclasses
from dataclasses import dataclass
from typing import List, NamedTuple, Set, Dict, Optional, Tuple, Callable
import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from abc import ABC, abstractmethod
import sys
import traceback


trans_action_map: Dict[Tuple[int, int], Action] = {
    (-1, 0): Action.NORTH,
    (1, 0): Action.SOUTH,
    (0, 1): Action.EAST,
    (0, -1): Action.WEST,
}


class Pos(NamedTuple):
    x: int
    y: int

    def __repr__(self):
        return f"[{self.x}:{self.y}]"


@dataclass
class Goose:
    head: Pos = dataclasses.field(init=False)
    poses: List[Pos]

    def __post_init__(self):
        self.head = self.poses[0]

    def __repr__(self):
        return "Goose(" + "-".join(map(str, self.poses)) + ")"

    def __iter__(self):
        return iter(self.poses)

    def __len__(self):
        return len(self.poses)


def field_idx_to_pos(field_idx: int, *, num_cols: int, num_rows: int) -> Pos:
    x = field_idx // num_cols
    y = field_idx % num_cols

    if not (0 <= x < num_rows and 0 <= y < num_cols):
        raise ValueError("Illegal field_idx {field_idx} with x={x} and y={y}")

    return Pos(x, y)


class Geometry:
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.size_x, self.size_y)

    def prox(self, pos: Pos) -> Set[Pos]:
        return {
            self.translate(pos, direction)
            for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]
        }

    def translate(self, pos: Pos, diff: Tuple[int, int]) -> Pos:
        x, y = pos
        dx, dy = diff
        return Pos((x + dx) % self.size_x, (y + dy) % self.size_y)

    def trans_to(self, pos1: Pos, pos2: Pos) -> Tuple[int, int]:
        dx = pos2.x - pos1.x
        dy = pos2.y - pos1.y

        if dx <= self.size_x // 2:
            dx += self.size_x

        if dx > self.size_x // 2:
            dx -= self.size_x

        if dy <= self.size_y // 2:
            dy += self.size_y

        if dy > self.size_y // 2:
            dy -= self.size_y

        return (dx, dy)

    def action_to(self, pos1, pos2):
        diff = self.trans_to(pos1, pos2)

        result = trans_action_map.get(diff)

        if result is None:
            raise ValueError(f"Cannot step from {pos1} to {pos2}")

        return result

@dataclass
class State:
    food: Set[Pos]
    geese: Dict[int, Goose]
    index: int
    step: int
    geo: Geometry

    field: np.ndarray = dataclasses.field(init=False)
    my_goose: Goose = dataclasses.field(init=False)
    danger_poses: Set[Pos] = dataclasses.field(init=False)

    def __post_init__(self):
        self.field = np.full(fill_value=0, shape=self.geo.shape)
        for goose in self.geese.values():
            for pos in goose.poses[:-1]:  # not considering tail!
                self.field[pos.x, pos.y] = 1

            if self.geo.prox(goose.head) & self.food:
                tail = goose.poses[-1]
                self.field[tail.x, tail.y] = 1


        self.my_goose = self.geese[self.index]

        self.danger_poses = {
            pos
            for i, goose in self.geese.items()
            if i != self.index
            for pos in self.geo.prox(goose.head)
        }

    @classmethod
    def from_obs_conf(cls, obs, conf):
        num_cols = conf["columns"]
        num_rows = conf["rows"]
        step = obs["step"]
        index = obs["index"]

        geese = {
            idx: Goose(
                poses=[
                    field_idx_to_pos(idx, num_cols=num_cols, num_rows=num_rows)
                    for idx in goose_data
                ]
            )
            for idx, goose_data in enumerate(obs["geese"])
            if goose_data
        }

        food = {
            field_idx_to_pos(idx, num_cols=num_cols, num_rows=num_rows)
            for idx in obs["food"]
        }

        return cls(
            food=food,
            geese=geese,
            index=index,
            step=step,
            geo=Geometry(size_x=num_rows, size_y=num_cols),
        )

    def __repr__(self):
        return (
            f"State(step:{self.step}, index:{self.index}, Geese("
            + ",".join(f"{idx}:{len(goose.poses)}" for idx, goose in self.geese.items())
            + f"), food:{len(self.food)})"
        )

@dataclass
class FloodfillResult:
    field_dist: np.ndarray
    frontiers: List[List[Tuple[int, int]]]


def flood_fill(is_occupied: np.ndarray, seeds: List[Pos]) -> FloodfillResult:
    """
    Flood will start with distance 0 at seeds and only flow where is_occupied[x,y]==0
    """
    size_x, size_y = is_occupied.shape

    field_dist = np.full(fill_value=-1, shape=(size_x, size_y))

    frontier = [(s.x, s.y) for s in seeds]

    frontiers = [frontier]

    for seed in seeds:
        field_dist[seed] = 0

    dist = 1

    while frontier:
        new_frontier: List[Tuple[int, int]] = []
        for x, y in frontier:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = (x + dx) % size_x
                new_y = (y + dy) % size_y
                if is_occupied[new_x, new_y] == 0 and field_dist[new_x, new_y] == -1:
                    field_dist[new_x, new_y] = dist
                    new_frontier.append((new_x, new_y))
        frontier = new_frontier
        frontiers.append(frontier)
        dist += 1

    return FloodfillResult(field_dist=field_dist, frontiers=frontiers)


def get_dist(
    floodfill_result: FloodfillResult, test_func: Callable[[Tuple[int, int]], bool]
) -> Optional[int]:
    for dist, frontier in enumerate(floodfill_result.frontiers):
        for pos in frontier:
            if test_func(pos):
                return dist

    return None

class BaseAgent(ABC):
    def __init__(self):
        self.last_pos: Optional[Pos] = None

    def __call__(self, obs, conf):
        try:
            state = State.from_obs_conf(obs, conf)

            next_pos = self.step(state)

            action = state.geo.action_to(state.my_goose.head, next_pos)

            self.last_pos = state.my_goose.head

            return action.name
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            raise

    @abstractmethod
    def step(self, state: State) -> Pos:
        """
        return: next position

        Implement this
        """
        pass

    def next_poses(self, state: State) -> Set[Pos]:
        head_next_poses = state.geo.prox(state.my_goose.head)

        result = {
            pos
            for pos in head_next_poses
            if pos != self.last_pos and state.field[pos] == 0
        }

        return result

from operator import itemgetter
import random

class FloodGoose(BaseAgent):
    def __init__(self, min_length=13):
        super().__init__()
        self.min_length = min_length

    def step(self, state):
        result = None

        if len(state.my_goose) < self.min_length:
            result = self.goto(state, lambda pos:pos in state.food)
        elif len(state.my_goose) >= 3:
            result = self.goto(state, lambda pos:pos==state.my_goose.poses[-1])

        if result is None:
            result = self.random_step(state)

        return result

    def goto(self, state, test_func):
        result = None

        pos_dists = {}
        for pos in self.next_poses(state):
            flood = flood_fill(state.field, [pos])
            dist = get_dist(flood, test_func)
            if dist is not None:
                pos_dists[pos] = dist

        if pos_dists:
            closest_pos, _ = min(pos_dists.items(), key=itemgetter(1))

            if closest_pos not in state.danger_poses:
                result = closest_pos

        return result


    def random_step(self, state):
        next_poses = self.next_poses(state) - state.danger_poses - state.food
        if not next_poses:
            next_poses = self.next_poses(state) - state.danger_poses

            if not next_poses:
                next_poses = self.next_poses(state)

                if not next_poses:
                    next_poses = state.geo.prox(state.my_goose.head)

        result = random.choice(list(next_poses))

        return result


agent = FloodGoose(min_length=8)

def call_agent(obs, conf):
    return agent(obs, conf)