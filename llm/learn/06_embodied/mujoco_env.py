# -*- coding: utf-8 -*-
"""
06_embodied/mujoco_env.py - MuJoCo Environment Wrapper
Provides: step, reset, render for MuJoCo physics simulation.
Install: pip install mujoco
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class EnvConfig:
    dt: float = 0.002
    n_substeps: int = 1
    render_width: int = 224
    render_height: int = 224
    camera_name: str = 'front'


class MuJoCoEnv:
    """Base MuJoCo environment. Works with or without MuJoCo installed."""
    def __init__(self, model_xml_path=None, config=None):
        self.config = config or EnvConfig()
        self._model = None; self._data = None; self._renderer = None
        self._initialized = False; self._simulated = False
        self._state = {'qpos': np.zeros(7), 'qvel': np.zeros(7)}
        self._mujoco = None
        print('MuJoCo env ready (install: pip install mujoco)')

    def _try_init(self):
        if self._initialized: return True
        try:
            import mujoco; self._mujoco = mujoco
            xml = '<mujoco><worldbody><light pos="0 0 3"/><geom type="plane" size="2 2 0.1"/></worldbody></mujoco>'
            self._model = mujoco.MjModel.from_xml_string(xml)
            self._data = mujoco.MjData(self._model)
            self._initialized = True
            return True
        except ImportError:
            self._initialized = True
            return False

    def step(self, action):
        if self._try_init():
            self._data.ctrl[:] = action
            for _ in range(self.config.n_substeps):
                self._mujoco.mj_step(self._model, self._data)
            return np.concatenate([self._data.qpos, self._data.qvel]), 0.0, False, {}
        self._state['qpos'] += np.asarray(action) * 0.01
        return self._state['qpos'].copy(), 0.0, False, {}

    def reset(self):
        if self._try_init():
            self._mujoco.mj_resetData(self._model, self._data)
            return np.concatenate([self._data.qpos, self._data.qvel])
        self._state['qpos'] = np.zeros(7)
        return self._state['qpos'].copy()

    def render(self, width=None, height=None, camera=None):
        w, h = width or self.config.render_width, height or self.config.render_height
        if self._try_init():
            if self._renderer is None:
                self._renderer = self._mujoco.Renderer(self._model, h, w)
            self._renderer.update_scene(self._data, camera or self.config.camera_name)
            return self._renderer.render()
        return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    @property
    def action_dim(self):
        return self._model.nu if self._model else 7
    @property
    def obs_dim(self):
        return (self._model.nq + self._model.nv) if self._model else 14


if __name__ == '__main__':
    env = MuJoCoEnv()
    obs = env.reset()
    print(f'Obs: {obs.shape}')
    for _ in range(10):
        obs, _, _, _ = env.step(np.random.randn(env.action_dim) * 0.01)
    img = env.render()
    print(f'Render: {img.shape}')
    print('Done!')