from typing import Dict

import gym
import numpy as np
import torch

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.utils import MinMaxNormalizer, dict_apply

from cleandiffuser_ex.utils import suppress_output


def compose(*fns):

    def _fn(x):
        for fn in fns:
            x = fn(x)
        return x

    return _fn


def get_preprocess_fn(fn_names, env):
    fns = [eval(name)(env) for name in fn_names]
    return compose(*fns)


def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env


def maze2d_set_terminals(env):
    env = load_environment(env) if type(env) == str else env
    goal = np.array(env._target)
    threshold = 0.5

    def _fn(dataset):
        xy = dataset['observations'][:,:2]
        distances = np.linalg.norm(xy - goal, axis=-1)
        at_goal = distances < threshold
        timeouts = np.zeros_like(dataset['timeouts'])

        ## timeout at time t iff
        ##      at goal at time t and
        ##      not at goal at time t + 1
        timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]

        timeout_steps = np.where(timeouts)[0]
        path_lengths = timeout_steps[1:] - timeout_steps[:-1]

        print(
            f'[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | '
            f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
        )

        dataset['timeouts'] = timeouts
        return dataset

    return _fn


class D4RLMaze2DDataset(BaseDataset):
    def __init__(
            self,
            dataset: Dict[str, np.ndarray],
            preprocess_fn=None,
            horizon: int = 1,
            max_path_length: int = 40000,
            discount: float = 0.99,
    ):
        super().__init__()

        dataset = preprocess_fn(dataset)
        observations, actions, rewards, timeouts, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["timeouts"],
            dataset["terminals"])
        self.timeouts = timeouts
        rewards -= 1  # -1 for each step and 0 for reaching the goal
        self.normalizers = {
            "state": MinMaxNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)

        self.horizon = horizon
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

        n_paths = np.sum(np.logical_or(terminals, timeouts))
        self.seq_obs = np.zeros((n_paths, max_path_length, self.o_dim), dtype=np.float32)
        self.seq_act = np.zeros((n_paths, max_path_length, self.a_dim), dtype=np.float32)
        self.tml_and_not_timeout = []
        self.indices = []

        path_lengths, ptr = [], 0
        path_idx = 0
        for i in range(timeouts.shape[0]):
            if timeouts[i] or terminals[i]:
                path_lengths.append(i - ptr + 1)

                self.seq_obs[path_idx, :i - ptr + 1] = normed_observations[ptr:i + 1]
                self.seq_act[path_idx, :i - ptr + 1] = actions[ptr:i + 1]

                max_start = min(path_lengths[-1] - 1, max_path_length - horizon)
                max_start = min(max_start, path_lengths[-1] - horizon)

                self.indices += [(path_idx, start, start + horizon) for start in range(max_start + 1)]

                ptr = i + 1
                path_idx += 1

        self.path_lengths = np.array(path_lengths)
        self.tml_and_not_timeout = np.array(self.tml_and_not_timeout, dtype=np.int64)

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]

        seq_obs = self.seq_obs[path_idx, start:end]
        unnormed_seq_obs = self.normalizers['state'].unnormalize(seq_obs)

        goal = unnormed_seq_obs[-1, :2] 
        dist = np.linalg.norm(unnormed_seq_obs[:, :2] - goal, axis=1, keepdims=True)

        reward = np.where(dist <= 0.5, 0, -1)
        val = np.sum(reward).reshape(1)

        data = {
            'obs': {
                'state': seq_obs},
            'act': self.seq_act[path_idx, start:end],
            'rew': reward, 
            'val': val, 
        }

        torch_data = dict_apply(data, torch.tensor)

        return torch_data


class MultiHorizonD4RLMaze2DDataset(BaseDataset):
    def __init__(
            self,
            dataset: Dict[str, np.ndarray],
            preprocess_fn=None,
            horizons=(17, 17),
            max_path_length: int = 40000,
            discount: float = 0.99,
    ):
        super().__init__()

        dataset = preprocess_fn(dataset)
        observations, actions, rewards, timeouts, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["timeouts"],
            dataset["terminals"])
        self.timeouts = timeouts
        rewards -= 1  # -1 for each step and 0 for reaching the goal
        self.normalizers = {
            "state": MinMaxNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)

        self.horizons = horizons
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

        n_paths = np.sum(np.logical_or(terminals, timeouts))
        self.seq_obs = np.zeros((n_paths, max_path_length, self.o_dim), dtype=np.float32)
        self.seq_act = np.zeros((n_paths, max_path_length, self.a_dim), dtype=np.float32)
        self.indices = [[] for _ in range(len(horizons))]

        path_lengths, ptr = [], 0
        path_idx = 0
        for i in range(timeouts.shape[0]):
            if timeouts[i] or terminals[i]:
                path_lengths.append(i - ptr + 1)

                self.seq_obs[path_idx, :i - ptr + 1] = normed_observations[ptr:i + 1]
                self.seq_act[path_idx, :i - ptr + 1] = actions[ptr:i + 1]

                max_starts = []
                for horizon in horizons:
                    max_start = min(path_lengths[-1] - 1, max_path_length - horizon)
                    max_start = min(max_start, path_lengths[-1] - horizon)
                    max_starts.append(max_start)

                for k in range(len(horizons)):
                    self.indices[k] += [(path_idx, start, start + horizons[k]) for start in range(max_starts[k] + 1)]

                ptr = i + 1
                path_idx += 1

        self.path_lengths = np.array(path_lengths)
        self.len_each_horizon = [len(indices) for indices in self.indices]

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return max(self.len_each_horizon)

    def __getitem__(self, idx: int):

        indices = [
            int(self.len_each_horizon[i] * (idx / self.len_each_horizon[-1])) for i in range(len(self.horizons))]

        torch_datas = []

        for i, horizon in enumerate(self.horizons):

            path_idx, start, end = self.indices[i][indices[i]]

            data = {
                'obs': {
                    'state': self.seq_obs[path_idx, start:end]},
                'act': self.seq_act[path_idx, start:end]}

            torch_data = dict_apply(data, torch.tensor)

            torch_datas.append({
                "horizon": horizon,
                "data": torch_data,
            })

        return torch_datas
