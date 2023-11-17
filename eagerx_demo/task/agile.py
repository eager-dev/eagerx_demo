from typing import Any, Dict, List
import numpy as np
import numpy.random as rnd
import xml.etree.ElementTree as ET
import tempfile
import os
from urdf_parser_py.urdf import URDF
import pybullet as pb
from pybullet_utils.bullet_client import BulletClient

import eagerx
import eagerx_demo
from eagerx_utility.utils import launch_node
from eagerx.core.specs import EngineStateSpec


PATH_TO_URDF = os.path.join(os.path.dirname(__file__), "assets")

COLORS = {
    "blue": [078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0, 1.0],
    "red": [255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0, 1.0],
    "green": [089.0 / 255.0, 169.0 / 255.0, 079.0 / 255.0, 1.0],
    "orange": [242.0 / 255.0, 142.0 / 255.0, 043.0 / 255.0, 1.0],
    "yellow": [237.0 / 255.0, 201.0 / 255.0, 072.0 / 255.0, 1.0],
    "purple": [176.0 / 255.0, 122.0 / 255.0, 161.0 / 255.0, 1.0],
    "pink": [255.0 / 255.0, 157.0 / 255.0, 167.0 / 255.0, 1.0],
    "cyan": [118.0 / 255.0, 183.0 / 255.0, 178.0 / 255.0, 1.0],
    "brown": [156.0 / 255.0, 117.0 / 255.0, 095.0 / 255.0, 1.0],
    # 'white': [255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0, 1.0],
    "gray": [186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0, 1.0],
}

BOLTS_HOLDER = {
    "first": [0.0, 0.0, 0.075],  # todo: assumes no rotation --> adapt to pose
    "second": [0.04, 0.0, 0.075],
    "third": [0.08, 0.0, 0.075],
    "fourth": [0.12, 0.0, 0.075],
}
Z = 0.164
BOLTS_ENGINE = {
    "interior_no_edge_big": [-0.063, 0.002, Z],
    "interior_no_edge_medium": [-0.065, 0.065, Z],
    "interior_with_edge_1": [-0.03, 0.012, Z + 0.015],
    "interior_with_edge_2": [-0.010, 0.042, Z + 0.015],
    "interior_with_edge_3": [-0.03, 0.068, Z + 0.015],
    "outer_ring_1": [0.085, 0.0865, Z + 0.015],
    "outer_ring_2": [0.085, -0.05, Z + 0.015],
    "outer_ring_3": [-0.022, -0.0875, Z + 0.015],
    "outer_ring_4": [0.055, -0.0875, Z + 0.015],
}


def modify_urdf(urdf_path, scale=None, color=None):
    # Parse the existing URDF file
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Update the scale and color
    for elem in root.iter("mesh"):
        elem.attrib["filename"] = os.path.join(os.path.split(urdf_path)[0], elem.attrib["filename"])
        if scale is not None:
            existing_scale = elem.attrib.get("scale", "0.001 0.001 0.001").split()
            new_scale = [str(float(existing_scale[i]) * scale[i]) for i in range(3)]
            elem.attrib["scale"] = " ".join(new_scale)

    if color is not None:
        for elem in root.iter("color"):
            elem.attrib["rgba"] = f"{color[0]} {color[1]} {color[2]} {color[3]}"

    # Create a new temporary URDF file with updated properties
    with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as tmp:
        tree.write(tmp.name)
        tmp_urdf_path = tmp.name

    return tmp_urdf_path


class AgileTaskState(eagerx.EngineState):
    @classmethod
    def make(cls, engine_pos=None, holder_pos=None, use_colors: List[str] = None):
        spec = cls.get_specification()
        spec.config.holder_pos = holder_pos if holder_pos is not None else [0.3, 0.1, 0.0]
        spec.config.engine_pos = engine_pos if engine_pos is not None else [0.3, -0.1, 0.0]  # -0.075
        spec.config.use_colors = use_colors if use_colors is not None else list(COLORS.keys())
        return spec

    def initialize(self, spec: EngineStateSpec, simulator: Any):
        self.simulator = simulator
        self._p = simulator["client"]
        self.ids = []
        self.use_colors = {k: COLORS[k] for k in spec.config.use_colors}

        # Load hole
        urdf_engine = modify_urdf(f"{PATH_TO_URDF}/engine_top_case.urdf", [1, 1, 1])
        urdf_holder = modify_urdf(f"{PATH_TO_URDF}/bolt_holder.urdf", [1, 1, 1])
        urdf_bolt = modify_urdf(f"{PATH_TO_URDF}/bolt_short.urdf", [1, 1, 1])
        # urdf_rockerarm = modify_urdf(f"{PATH_TO_URDF}/rockerarm.urdf", [1, 1, 1])

        # Bolt holder
        self._holder_pos = spec.config.holder_pos
        self.holder_id = self._p.loadURDF(urdf_holder, self._holder_pos, [0, 0, 0, 1], useFixedBase=True)
        self.bolt_ids = {}
        for k, v in BOLTS_HOLDER.items():
            pos_bolt = [v[0] + self._holder_pos[0], v[1] + self._holder_pos[1], v[2] + self._holder_pos[2]]
            bolt_id = self._p.loadURDF(urdf_bolt, pos_bolt, [0, 1, 0, 0], useFixedBase=False)
            self.bolt_ids[k] = bolt_id

        # Engine top case
        self._engine_pos = spec.config.engine_pos
        self._p.loadURDF(urdf_engine, self._engine_pos, [0, 0, 0, 1], useFixedBase=True)

        # UNCOMMENT TO PLACE BOLTS IN THE ENGINE
        # for k, v in BOLTS_ENGINE.items():
        #     pos = [v[0]+self._engine_pos[0], v[1]+self._engine_pos[1], v[2]+self._engine_pos[2]]
        #     bolt_id = self._p.loadURDF(urdf_bolt, pos, [0, 1, 0, 0], useFixedBase=False)
        #     self.bolt_ids.append(bolt_id)

        # # create numpy random number generator seeded with spec.config.seed
        # self.bolt_ids = []
        # self.hole_ids = []
        # for i in range(self.num_bolts):
        #     bolt_ids = add_bolt_in_hole(self._p, [0.0, 0.0, 0.0], 1.0, COLORS["red"], add_bolt=True)
        #     self.bolt_ids.append((bolt_ids["bolt"], bolt_ids["hole"]))
        #     hole_ids = add_bolt_in_hole(self._p, [0.0, 0.0, 0.0], 1.0, COLORS["red"], add_bolt=False)
        #     self.hole_ids.append((hole_ids["bolt"], hole_ids["hole"]))
        #
        reset_task(self._p, self._holder_pos, self.bolt_ids, self.use_colors)

    def reset(self, state: np.ndarray):
        must_reset_task = state
        if must_reset_task:
            reset_task(self._p, self._holder_pos, self.bolt_ids, self.use_colors)


def reset_task(p: BulletClient, holder_pos, bolt_ids, colors: Dict[str, List[float]]):
    """
    Resets the task by moving the bolts to random positions within the workspace.
    """
    # Move each bolt to its new position
    for k, bolt_id in bolt_ids.items():
        pos_bolt = BOLTS_HOLDER[k]
        pos_bolt = [pos_bolt[0] + holder_pos[0], pos_bolt[1] + holder_pos[1], pos_bolt[2] + holder_pos[2]]
        p.resetBasePositionAndOrientation(bolt_id, pos_bolt, [0, 1, 0, 0])

    # Change the color of each bol
    iter_colors = iter(colors.values())
    for i, bolt_ids in enumerate(bolt_ids.values()):
        try:
            c = next(iter_colors)
        except StopIteration:
            iter_colors = iter(colors.values())
            c = next(iter_colors)
        p.changeVisualShape(bolt_ids, -1, rgbaColor=c)


def world_with_table_and_plane(bullet_client):
    z = -0.66

    urdf_table = modify_urdf(f"{PATH_TO_URDF}/table.urdf", [1, 1, 1])
    bullet_client.loadURDF(urdf_table, [0, 0, z], [0, 0, 0, 1], useFixedBase=True)

    # ground plane
    _ = bullet_client.loadURDF(
        "plane.urdf",
        [0, 0, z],
        useFixedBase=True,
    )
