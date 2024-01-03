"""Image dataset."""

import os
import pickle
import warnings
from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset

from eagerx_demo.cliport.utils import utils

# Names as strings, REVERSE-sorted so longer (more specific) names are first.
TASK_NAMES = [""]

OBJECTS = ["bolt", "screw", "thing", "item", "fastener", "one"]
PICK_ACTIONS = ["pick", "grab", "take", "get"]
PLACE_ACTIONS = ["place", "put", "set", "drop"]
PREAMBLES = [
    "can you",
    "please",
    "will you",
    "could you",
    "would you",
    "can you please",
    "please can you",
    "please will you",
    "will you please",
    "could you please",
    "please could you",
    "could you please",
    "would you please",
    "please would you",
    "would you please",
    "can you please",
    "please can you",
    "please will you",
    "will you please",
    "could you please",
    "please could you",
    "could you please",
    "would you please",
    "please would you",
    "would you please",
]
GREETINGS = ["hello", "hi", "hey", "howdy", "greetings", "good morning", "good afternoon", "good evening", "yo"]
GREETINGS = GREETINGS + [greeting + " robot" for greeting in GREETINGS]
TOP_LEFTS = ["top left", "upper left", "left top", "left upper"]
TOP_RIGHTS = ["top right", "upper right", "right top", "right upper"]
BOTTOM_LEFTS = ["bottom left", "lower left", "left bottom", "left lower", "lower left hand corner"]
BOTTOM_RIGHTS = ["bottom right", "lower right", "right bottom", "right lower", "lower right hand corner"]
MIDDLE_LEFTS = ["middle left", "left middle", "center left", "left center", "upper one of the lower left corner"]
MIDDLE_RIGHTS = ["middle right", "right middle", "center right", "right center", "upper one of the lower right corner"]
LEFT_TUBES = [
    "left tube",
    "left pipe",
    "left cylinder",
    "tube on the left",
    "pipe on the left",
    "cylinder on the left",
    "lefthand tube",
    "lefthand pipe",
    "lefthand cylinder",
    "tube on the lefthand side",
    "pipe on the lefthand side",
    "cylinder on the lefthand side",
    "tube on the left hand side",
    "pipe on the left hand side",
    "cylinder on the left hand side",
]
RIGHT_TUBES = [
    "right tube",
    "right pipe",
    "right cylinder",
    "tube on the right",
    "pipe on the right",
    "cylinder on the right",
    "righthand tube",
    "righthand pipe",
    "righthand cylinder",
    "tube on the righthand side",
    "pipe on the righthand side",
    "cylinder on the righthand side",
    "tube on the right hand side",
    "pipe on the right hand side",
    "cylinder on the right hand side",
]
MIDDLE_TUBES = [
    "middle tube",
    "middle pipe",
    "middle cylinder",
    "tube in the middle",
    "pipe in the middle",
    "cylinder in the middle",
]
HOLES = ["hole", "opening", "insertion point", "cavity"]
ALL = (
    OBJECTS
    + PICK_ACTIONS
    + PLACE_ACTIONS
    + PREAMBLES
    + GREETINGS
    + TOP_LEFTS
    + TOP_RIGHTS
    + BOTTOM_LEFTS
    + BOTTOM_RIGHTS
    + MIDDLE_LEFTS
    + MIDDLE_RIGHTS
    + LEFT_TUBES
    + RIGHT_TUBES
    + MIDDLE_TUBES
    + HOLES
)


def augment_language(language):
    """Augment language with new words and phrases."""
    if language.startswith(" ") or language.startswith(","):
        language = language[1:]
    # make sure language is lowercase
    language = language.lower()
    # remove punctuation
    language = language.replace(",", "")
    language = language.replace(".", "")
    language = language.replace("!", "")
    language = language.replace("?", "")

    # divide language into words
    words = language.split(" ")

    # check for PREAMBLES
    preamble_in_language = False
    for preamble in PREAMBLES:
        if preamble in language:
            preamble_words = preamble.split(" ")
            all_words_in_language = True
            for preamble_word in preamble_words:
                if preamble_word not in words:
                    all_words_in_language = False
                    break
            if all_words_in_language:
                text_without_preamble = deepcopy(ALL)
                text_without_preamble.remove(preamble)
                for text in text_without_preamble:
                    if preamble in text:
                        if text not in language:
                            if np.random.random() < 0.5:
                                random_preamble = np.random.choice(PREAMBLES)
                                language = language.replace(preamble, random_preamble)
                            preamble_in_language = True
                            break
            if preamble_in_language:
                break
    greeting_in_language = False
    for greeting in GREETINGS:
        if greeting in language:
            greeting_words = greeting.split(" ")
            all_words_in_language = True
            for greeting_word in greeting_words:
                if greeting_word not in words:
                    all_words_in_language = False
                    break
            if all_words_in_language:
                text_without_greeting = deepcopy(ALL)
                text_without_greeting.remove(greeting)
                for text in text_without_greeting:
                    if greeting in text:
                        if text not in language:
                            if np.random.random() < 0.5:
                                random_greeting = np.random.choice(GREETINGS)
                                if not preamble_in_language:
                                    if np.random.random() < 0.5:
                                        random_greeting = np.random.choice(PREAMBLES) + " " + random_greeting
                                language = language.replace(greeting, random_greeting)
                                greeting_in_language = True
                                break
            if greeting_in_language:
                break
    if not greeting_in_language:
        if not preamble_in_language:
            if np.random.random() < 0.5:
                random_preamble = np.random.choice(PREAMBLES)
                language = random_preamble + " " + language
        if np.random.random() < 0.5:
            random_greeting = np.random.choice(GREETINGS)
            language = random_greeting + " " + language
    for obj in OBJECTS:
        if obj in language:
            obj_words = obj.split(" ")
            all_words_in_language = True
            for obj_word in obj_words:
                if obj_word not in words:
                    all_words_in_language = False
                    break
            if all_words_in_language:
                text_without_obj = deepcopy(ALL)
                text_without_obj.remove(obj)
                for text in text_without_obj:
                    if obj in text:
                        if text not in language:
                            random_obj = np.random.choice(OBJECTS)
                            language = language.replace(obj, random_obj)
                            break
            if all_words_in_language:
                break
    for pick_action in PICK_ACTIONS:
        if pick_action in language:
            pick_action_words = pick_action.split(" ")
            all_words_in_language = True
            for pick_action_word in pick_action_words:
                if pick_action_word not in words:
                    all_words_in_language = False
                    break
            if all_words_in_language:
                text_without_pick_action = deepcopy(ALL)
                text_without_pick_action.remove(pick_action)
                for text in text_without_pick_action:
                    if pick_action in text:
                        if text not in language:
                            random_pick_action = np.random.choice(PICK_ACTIONS)
                            language = language.replace(pick_action, random_pick_action)
                            break
            if all_words_in_language:
                break
    for place_action in PLACE_ACTIONS:
        if place_action in language:
            place_action_words = place_action.split(" ")
            all_words_in_language = True
            for place_action_word in place_action_words:
                if place_action_word not in words:
                    all_words_in_language = False
                    break
            if all_words_in_language:
                text_without_place_action = deepcopy(ALL)
                text_without_place_action.remove(place_action)
                for text in text_without_place_action:
                    if place_action in text:
                        if text not in language:
                            random_place_action = np.random.choice(PLACE_ACTIONS)
                            language = language.replace(place_action, random_place_action)
                            break
            if all_words_in_language:
                break
    locations = [
        TOP_LEFTS,
        TOP_RIGHTS,
        BOTTOM_LEFTS,
        BOTTOM_RIGHTS,
        MIDDLE_LEFTS,
        MIDDLE_RIGHTS,
        LEFT_TUBES,
        RIGHT_TUBES,
        MIDDLE_TUBES,
    ]
    for location in locations:
        for loc in location:
            if loc in language:
                loc_words = loc.split(" ")
                all_words_in_language = True
                for loc_word in loc_words:
                    if loc_word not in words:
                        all_words_in_language = False
                        break
                if all_words_in_language:
                    text_without_loc = deepcopy(ALL)
                    text_without_loc.remove(loc)
                    for text in text_without_loc:
                        if loc in text:
                            if text not in language:
                                random_loc = np.random.choice(location)
                                language = language.replace(loc, random_loc)
                                break
                if all_words_in_language:
                    break
        if all_words_in_language:
            break
    for hole in HOLES:
        if hole in language:
            hole_words = hole.split(" ")
            all_words_in_language = True
            for hole_word in hole_words:
                if hole_word not in words:
                    all_words_in_language = False
                    break
            if all_words_in_language:
                text_without_hole = deepcopy(ALL)
                text_without_hole.remove(hole)
                for text in text_without_hole:
                    if hole in text:
                        if text not in language:
                            random_hole = np.random.choice(HOLES)
                            language = language.replace(hole, random_hole)
                            break
            if all_words_in_language:
                break
    return language


class RavensDataset(Dataset):
    """A simple image dataset class."""

    def __init__(self, path, cfg, pix_size, in_shape, cam_config, bounds, n_demos=0, augment=False):
        """A simple RGB-D image dataset."""
        self._path = path

        self.cfg = cfg
        self.sample_set = []
        self.max_seed = -1
        self.n_episodes = 0
        self.images = self.cfg["dataset"]["images"]
        self.cache = self.cfg["dataset"]["cache"]
        self.n_demos = n_demos
        self.augment = augment

        self.aug_theta_sigma = (
            self.cfg["dataset"]["augment"]["theta_sigma"] if "augment" in self.cfg["dataset"] else 60
        )  # legacy code issue: theta_sigma was newly added
        self.pix_size = pix_size
        self.in_shape = in_shape
        self.cam_config = cam_config
        self.bounds = bounds

        # Track existing dataset if it exists.
        color_path = os.path.join(self._path, "action")
        if os.path.exists(color_path):
            for fname in sorted(os.listdir(color_path)):
                if ".pkl" in fname:
                    seed = int(fname[(fname.find("-") + 1) : -4])
                    self.n_episodes += 1
                    self.max_seed = max(self.max_seed, seed)

        self._cache = {}

        if self.n_demos > 0:
            self.images = self.cfg["dataset"]["images"]
            self.cache = self.cfg["dataset"]["cache"]

            # Check if there sufficient demos in the dataset
            if self.n_demos > self.n_episodes:
                raise Exception(
                    f"Requested training on {self.n_demos} demos, but only {self.n_episodes} demos exist in the dataset path: {self._path}."
                )

            episodes = np.random.choice(range(self.n_episodes), self.n_demos, False)
            self.set(episodes)

    def add(self, seed, episode):
        """Add an episode to the dataset.

        Args:
          seed: random seed used to initialize the episode.
          episode: list of (obs, act, reward, info) tuples.
        """
        color, depth, action, reward, info = [], [], [], [], []
        for obs, act, r, i in episode:
            color.append(obs["color"])
            depth.append(obs["depth"])
            action.append(act)
            reward.append(r)
            info.append(i)

        color = np.uint8(color)
        depth = np.float32(depth)

        def dump(data, field):
            field_path = os.path.join(self._path, field)
            if not os.path.exists(field_path):
                os.makedirs(field_path)
            fname = f"{self.n_episodes:06d}-{seed}.pkl"  # -{len(episode):06d}
            with open(os.path.join(field_path, fname), "wb") as f:
                pickle.dump(data, f)

        dump(color, "color")
        dump(depth, "depth")
        dump(action, "action")
        dump(reward, "reward")
        dump(info, "info")

        self.n_episodes += 1
        self.max_seed = max(self.max_seed, seed)

    def set(self, episodes):
        """Limit random samples to specific fixed set."""
        self.sample_set = episodes

    def load(self, episode_id, images=True, cache=False):
        def load_field(episode_id, field, fname):

            # Check if sample is in cache.
            if cache:
                if episode_id in self._cache:
                    if field in self._cache[episode_id]:
                        return self._cache[episode_id][field]
                else:
                    self._cache[episode_id] = {}

            # Load sample from files.
            path = os.path.join(self._path, field)
            data = pickle.load(open(os.path.join(path, fname), "rb"))
            if cache:
                self._cache[episode_id][field] = data
            return data

        # Get filename and random seed used to initialize episode.
        seed = None
        path = os.path.join(self._path, "action")
        for fname in sorted(os.listdir(path)):
            if f"{episode_id:06d}" in fname:
                seed = int(fname[(fname.find("-") + 1) : -4])

                # Load data.
                color = load_field(episode_id, "color", fname)
                depth = load_field(episode_id, "depth", fname)
                action = load_field(episode_id, "action", fname)
                reward = load_field(episode_id, "reward", fname)
                info = load_field(episode_id, "info", fname)

                # Reconstruct episode.
                episode = []
                for i in range(len(action)):
                    obs = {"color": color[i], "depth": depth[i]} if images else {}
                    episode.append((obs, action[i], reward[i], info[i]))
                return episode, seed

    def get_image(self, obs, cam_config=None):
        """Stack color and height images image."""

        # if self.use_goal_image:
        #   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
        #   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
        #   input_image = np.concatenate((input_image, goal_image), axis=2)
        #   assert input_image.shape[2] == 12, input_image.shape

        if cam_config is None:
            cam_config = self.cam_config

        # Get color and height maps from RGB-D images.
        cmap, hmap = utils.get_fused_heightmap(obs, cam_config, self.bounds, self.pix_size)
        img = np.concatenate((cmap, hmap[Ellipsis, None], hmap[Ellipsis, None], hmap[Ellipsis, None]), axis=2)
        assert img.shape == self.in_shape, img.shape
        return img

    def process_sample(self, datum, augment=True):
        # Get training labels from data sample.
        (obs, act, _, info) = datum
        img = self.get_image(obs)

        p0, p1 = None, None
        p0_theta, p1_theta = None, None
        perturb_params = None

        if act:
            p0_xyz, p0_xyzw = act["pose0"]
            p1_xyz, p1_xyzw = act["pose1"]
            p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
            p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
            p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
            p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
            p1_theta = p1_theta - p0_theta
            p0_theta = 0

        # Data augmentation.
        if augment:
            img, _, (p0, p1), perturb_params = utils.perturb(img, [p0, p1], theta_sigma=self.aug_theta_sigma)

        sample = {"img": img, "p0": p0, "p0_theta": p0_theta, "p1": p1, "p1_theta": p1_theta, "perturb_params": perturb_params}

        # Add language goal if available.
        if "lang_goal" not in info:
            warnings.warn("No language goal. Defaulting to 'task completed.'")

        if info and "lang_goal" in info:
            sample["lang_goal"] = augment_language(info["lang_goal"])
        else:
            sample["lang_goal"] = "task completed."

        return sample

    def process_goal(self, goal, perturb_params):
        # Get goal sample.
        (obs, act, _, info) = goal
        img = self.get_image(obs)

        p0, p1 = None, None
        p0_theta, p1_theta = None, None

        # Data augmentation with specific params.
        if perturb_params:
            img = utils.apply_perturbation(img, perturb_params)

        sample = {"img": img, "p0": p0, "p0_theta": p0_theta, "p1": p1, "p1_theta": p1_theta, "perturb_params": perturb_params}

        # Add language goal if available.
        if "lang_goal" not in info:
            warnings.warn("No language goal. Defaulting to 'task completed.'")

        if info and "lang_goal" in info:
            sample["lang_goal"] = info["lang_goal"]
        else:
            sample["lang_goal"] = "task completed."

        return sample

    def __len__(self):
        # return len(self.sample_set)
        return self.n_episodes

    def __getitem__(self, idx):
        # Choose random episode.
        episode_id = np.random.choice(range(self.n_episodes))
        episode, _ = self.load(episode_id, self.images, self.cache)

        # Is the task sequential like stack-block-pyramid-seq?
        is_sequential_task = "-seq" in self._path.split("/")[-1]

        # Return random observation action pair (and goal) from episode.
        i = np.random.choice(range(len(episode)))
        sample, goal = episode[i], episode[-1]

        # Process sample.
        sample = self.process_sample(sample, augment=self.augment)
        goal = self.process_goal(goal, perturb_params=sample["perturb_params"])

        return sample, goal
