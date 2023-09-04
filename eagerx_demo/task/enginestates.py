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


def modify_urdf(urdf_path, scale=None, color=None):
    # Parse the existing URDF file
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Update the scale and color
    for elem in root.iter('mesh'):
        elem.attrib['filename'] = os.path.join(os.path.split(urdf_path)[0], elem.attrib['filename'])
        if scale is not None:
            existing_scale = elem.attrib.get('scale', '1 1 1').split()
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


def add_bolt_in_hole(
    p: BulletClient, pos: List[float], scale: float, color: List[float], add_bolt: bool = True
) -> Dict[str, int]:
    x, y, z = pos
    ids = {"bolt": None, "hole": None}

    # Load bolt
    if add_bolt:
        urdf_bolt = modify_urdf(f"{PATH_TO_URDF}/bolt-template.urdf", [scale, scale, scale], color=None)
        bolt = p.loadURDF(
            urdf_bolt,
            [x, y, z + 0.03 * scale],
            [0, 0, 0, 1],
            useFixedBase=False,
        )
        ids["bolt"] = bolt

    # Load hole
    urdf_hole = modify_urdf(f"{PATH_TO_URDF}/hollow_cylinder_L50R24T11H13-template.urdf", [1, 1, 0.5], color)

    hole = p.loadURDF(
        urdf_hole,
        [x, y, z],
        [0, 0, 0, 1],
        useFixedBase=True,
    )
    ids["hole"] = hole
    return ids


def generate_random_coordinates(rng, n, workspace, d):
    """
    Generates 'n' random (x, y) coordinates within a specified workspace,
    where each point is at least 'd' distance apart from every other point.

    Parameters:
        n (int): Number of points to generate.
        workspace (list): [xmin, xmax, ymin, ymax] specifying the area within which to generate points.
        d (float): Minimum distance between any two points.

    Returns:
        numpy.ndarray: An array of shape (n, 2), where each row represents the (x, y) coordinates of a point.
    """
    xmin, xmax, ymin, ymax = workspace
    points = np.empty((0, 2), dtype=np.float32)

    iter = 0
    while points.shape[0] < n:
        iter += 1
        if iter > 100 * n:
            raise ValueError(
                "Could not generate enough points. Try increasing the workspace area or decreasing the minimum distance between points."
            )
        # Generate a random point within the workspace
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        new_point = np.array([[x, y]])

        # Check if this point is at least 'd' distance away from all existing points
        if points.shape[0] == 0:
            points = np.vstack([points, new_point])
        else:
            distances = np.sqrt(np.sum((points - new_point) ** 2, axis=1))
            if np.all(distances >= d):
                points = np.vstack([points, new_point])

    return points


class TaskState(eagerx.EngineState):
    @classmethod
    def make(
        cls, num_bolts: int = 4, use_colors: List[str] = None, workspace: List[float] = None, distance: float = 0.05, seed=None
    ):
        """

        :param num_bolts: Number of bolts to use.
        :param use_colors: Colors to use. If None, all colors are used.
        :param workspace: [x_min, x_max, y_min, y_max]
        :return:
        """
        spec = cls.get_specification()
        spec.config.num_bolts = num_bolts
        spec.config.use_colors = use_colors if use_colors is not None else list(COLORS.keys())
        spec.config.workspace = workspace if workspace is not None else [0.2, 0.5, -0.5, 0.5]
        spec.config.distance = distance
        spec.config.seed = seed if seed is not None else rnd.randint(0, 1000000)
        return spec

    def initialize(self, spec: EngineStateSpec, simulator: Any):
        self.simulator = simulator
        self._p = simulator["client"]
        self.ids = []
        self.num_bolts = spec.config.num_bolts
        self.use_colors = {k: COLORS[k] for k in spec.config.use_colors}
        self.workspace = spec.config.workspace
        self.distance = spec.config.distance

        # create numpy random number generator seeded with spec.config.seed
        self.rng = rnd.RandomState(spec.config.seed)
        self.bolt_ids = []
        self.hole_ids = []
        for i in range(self.num_bolts):
            bolt_ids = add_bolt_in_hole(self._p, [0.0, 0.0, 0.0], 1.0, COLORS["red"], add_bolt=True)
            self.bolt_ids.append((bolt_ids["bolt"], bolt_ids["hole"]))
            hole_ids = add_bolt_in_hole(self._p, [0.0, 0.0, 0.0], 1.0, COLORS["red"], add_bolt=False)
            self.hole_ids.append((hole_ids["bolt"], hole_ids["hole"]))

        reset_task(self._p, self.rng, self.bolt_ids, self.hole_ids, self.workspace, self.distance, self.use_colors)

    def reset(self, state: np.ndarray):
        must_reset_task = state
        if must_reset_task:
            reset_task(self._p, self.rng, self.bolt_ids, self.hole_ids, self.workspace, self.distance, self.use_colors)


def reset_task(p: BulletClient, rng, bolt_ids, hole_ids, workspace, distance, colors: Dict[str, List[float]]):
    """
    Resets the task by moving the bolts to random positions within the workspace.
    """
    # Generate random coordinates for each bolt
    random_coordinates = generate_random_coordinates(rng, len(bolt_ids) + len(hole_ids), workspace, distance)
    pos_bolt = random_coordinates[: len(bolt_ids)]
    pos_hole = random_coordinates[len(bolt_ids) :]

    # Move each bolt to its new position
    for i, (bolt_id, hole_id) in enumerate(bolt_ids):
        x, y = pos_bolt[i]  # if i > 0 else [0.3, 0]
        p.resetBasePositionAndOrientation(bolt_id, [x, y, 0.03], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(hole_id, [x, y, 0], [0, 0, 0, 1])

    # Move each hole to its new position
    for i, (_, hole_id) in enumerate(hole_ids):
        x, y = pos_hole[i]  # if i > 0 else [0.4, 0]
        p.resetBasePositionAndOrientation(hole_id, [x, y, 0], [0, 0, 0, 1])

    # Change the color of each bol
    iter_colors = iter(colors.values())
    [next(iter_colors) for _ in range(rng.randint(0, len(colors.values())))]
    for i, (_, hole_id) in enumerate(bolt_ids + hole_ids):
        try:
            c = next(iter_colors)
        except StopIteration:
            iter_colors = iter(colors.values())
            c = next(iter_colors)
        p.changeVisualShape(hole_id, -1, rgbaColor=c)
