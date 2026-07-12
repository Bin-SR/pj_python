# -*- coding: utf-8 -*-
"""
06_embodied/mujoco_env.py - MuJoCo Environment Wrapper
Provides: step, reset, render for MuJoCo physics simulation.
Install: pip install mujoco

Viewer support: launch_viewer() opens an interactive MuJoCo window.
"""
import time
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class EnvConfig:
    dt: float = 0.002
    n_substeps: int = 1
    render_width: int = 224
    render_height: int = 224
    camera_name: str = 'front_cam'


class MuJoCoEnv:
    """Base MuJoCo environment. Works with or without MuJoCo installed."""

    def __init__(self, model_xml_path=None, config=None):
        self.config = config or EnvConfig()
        self._model = None
        self._data = None
        self._renderer = None
        self._viewer = None           # Interactive viewer window
        self._initialized = False
        self._simulated = False
        self._state = {'qpos': np.zeros(7), 'qvel': np.zeros(7)}
        self._mujoco = None
        self._viewer_running = False
        print('MuJoCo env ready (install: pip install mujoco)')

    # ================================================================
    # Initialization
    # ================================================================
    def _try_init(self):
        if self._initialized:
            return True
        try:
            import mujoco
            self._mujoco = mujoco
            scene_path = "C:/Disk/Data/New_Project/VScode_project/py/mj/model/franka_emika_panda/scene2.xml"
            self._model = mujoco.MjModel.from_xml_path(scene_path)
            self._data = mujoco.MjData(self._model)
            self._initialized = True
            print(f'MuJoCo initialized. nq={self._model.nq}, nv={self._model.nv}, nu={self._model.nu}')
            return True
        except ImportError:
            self._initialized = True
            print('MuJoCo not installed. Running in simulation-only mode.')
            print('Install: pip install mujoco')
            return False

    # ================================================================
    # Viewer (interactive window)
    # ================================================================
    def launch_viewer(self) -> bool:
        """
        打开 MuJoCo 交互式可视化窗口。

        使用 mujoco.viewer.launch_passive() 创建一个原生窗口，
        实时显示机器人仿真状态。窗口会持续运行直到手动关闭。

        Returns:
            True if viewer launched successfully, False otherwise
        """
        if not self._try_init():
            print('Cannot launch viewer: MuJoCo not installed.')
            return False

        if self._viewer is not None and self._viewer_running:
            print('Viewer already running.')
            return True

        try:
            # launch_passive: 被动模式，由用户代码控制 sync 时机
            self._viewer = self._mujoco.viewer.launch_passive(
                self._model, self._data,
                show_left_ui=True,   # 显示左侧控制面板
                show_right_ui=True,  # 显示右侧信息面板
            )
            self._viewer_running = True
            print('MuJoCo viewer launched. Close the window or press Esc to exit.')
            return True
        except Exception as e:
            print(f'Failed to launch viewer: {e}')
            print('  Make sure you have a display and mujoco is properly installed.')
            print('  Try: pip install mujoco')
            return False

    def sync_viewer(self):
        """
        同步可视化窗口（每步仿真后调用）。

        如果 viewer 被用户关闭，自动标记为非运行状态。
        """
        if self._viewer is not None and self._viewer_running:
            if self._viewer.is_running():
                self._viewer.sync()
            else:
                self._viewer_running = False
                print('Viewer window closed by user.')

    def close_viewer(self):
        """关闭可视化窗口。"""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            self._viewer_running = False
            print('Viewer closed.')

    def is_viewer_running(self) -> bool:
        """检查可视化窗口是否还在运行。"""
        return self._viewer_running

    # ================================================================
    # Simulation
    # ================================================================
    def step(self, action, n_substeps: int = None, sync_viewer: bool = True):
        """
        执行一步（或多步）物理仿真。

        Args:
            action: 控制指令，shape=(nu,)，对于位置控制模式就是目标关节角度
            n_substeps: 本次 step 内的物理子步数（None 则使用 config 默认值）
            sync_viewer: 是否在步进后同步可视化窗口

        Returns:
            (observation, reward, done, info)
        """
        if self._try_init():
            # 设置控制信号（位置控制模式下即目标关节角度）
            ctrl_len = min(len(action), self._model.nu)
            self._data.ctrl[:ctrl_len] = action[:ctrl_len]

            substeps = n_substeps if n_substeps is not None else self.config.n_substeps
            for _ in range(substeps):
                self._mujoco.mj_step(self._model, self._data)

            if sync_viewer:
                self.sync_viewer()

            obs = np.concatenate([self._data.qpos.copy(), self._data.qvel.copy()])
            return obs, 0.0, False, {}
        else:
            self._state['qpos'] += np.asarray(action) * 0.01
            return self._state['qpos'].copy(), 0.0, False, {}

    def reset(self):
        if self._try_init():
            self._mujoco.mj_resetData(self._model, self._data)
            self.sync_viewer()
            return np.concatenate([self._data.qpos.copy(), self._data.qvel.copy()])
        self._state['qpos'] = np.zeros(7)
        return self._state['qpos'].copy()

    def render(self, width=None, height=None, camera=None):
        """离线渲染一帧图像（用于 VLA 视觉输入，非可视化窗口）。"""
        w, h = width or self.config.render_width, height or self.config.render_height
        if self._try_init():
            if self._renderer is None:
                self._renderer = self._mujoco.Renderer(self._model, h, w)
            self._renderer.update_scene(
                self._data,
                camera=camera or self.config.camera_name
            )
            return self._renderer.render()
        return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    @property
    def action_dim(self):
        return self._model.nu if self._model else 7

    @property
    def obs_dim(self):
        return (self._model.nq + self._model.nv) if self._model else 14

    @property
    def data(self):
        """直接访问 MuJoCo mjData（高级用户）。"""
        return self._data

    @property
    def model(self):
        """直接访问 MuJoCo mjModel（高级用户）。"""
        return self._model


if __name__ == '__main__':
    print('=' * 60)
    print('MuJoCo Environment Demo (with viewer)')
    print('=' * 60)

    env = MuJoCoEnv()
    obs = env.reset()
    print(f'Obs: {obs.shape}')

    # 启动可视化
    env.launch_viewer()

    # 运行仿真循环
    try:
        for i in range(500):
            action = np.random.randn(env.action_dim) * 0.01
            obs, _, _, _ = env.step(action, n_substeps=5)

            if not env.is_viewer_running():
                print('Viewer closed, stopping.')
                break

            if i % 100 == 0:
                print(f'  Step {i}')

    except KeyboardInterrupt:
        print('Interrupted.')
    finally:
        env.close_viewer()

    print('Done!')