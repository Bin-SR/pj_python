# -*- coding: utf-8 -*-
"""
06_embodied/mujoco_env.py - MuJoCo Environment Wrapper
Provides: step, reset, render for MuJoCo physics simulation.
Install: pip install mujoco glfw

Viewer support: launch_viewer() opens an interactive MuJoCo window.
Uses glfw + mujoco direct rendering (compatible with all MuJoCo 3.x).
"""
import time
import sys
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
        self._renderer = None      # 离线渲染器（用于 VLA 视觉输入）
        self._initialized = False
        self._state = {'qpos': np.zeros(7), 'qvel': np.zeros(7)}
        self._mujoco = None

        # ---- 可视化窗口相关 ----
        self._window = None        # glfw 窗口句柄
        self._gl_context = None    # MuJoCo OpenGL 上下文
        self._scene = None         # MuJoCo 场景对象
        self._camera = None        # 自由摄像机
        self._viewer_running = False
        self._viewer_width = 1200
        self._viewer_height = 900

        print('MuJoCo env ready (install: pip install mujoco glfw)')

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
    # Viewer — 使用 glfw + mujoco 直接渲染（兼容所有 MuJoCo 3.x）
    # ================================================================
    def launch_viewer(self, width=1200, height=900, title="MuJoCo - Franka Panda"):
        """
        打开 MuJoCo 交互式可视化窗口。

        使用 glfw + mujoco 原生渲染 API，兼容所有 MuJoCo 3.x 版本。
        不需要 mujoco.viewer 子模块。

        Args:
            width, height: 窗口尺寸
            title: 窗口标题

        Returns:
            True if viewer launched successfully, False otherwise
        """
        if not self._try_init():
            print('Cannot launch viewer: MuJoCo not installed.')
            return False

        if self._viewer_running:
            print('Viewer already running.')
            return True

        try:
            import glfw
            self._glfw = glfw
        except ImportError:
            print('glfw not installed. Install: pip install glfw')
            print('Or try: conda install -c conda-forge glfw')
            return False

        try:
            # ---- Step 1: 初始化 glfw ----
            if not self._glfw.init():
                print('Failed to initialize GLFW.')
                return False

            # ---- Step 2: 创建窗口 ----
            self._viewer_width = width
            self._viewer_height = height
            self._window = self._glfw.create_window(width, height, title, None, None)
            if self._window is None:
                self._glfw.terminate()
                print('Failed to create GLFW window.')
                return False

            self._glfw.make_context_current(self._window)
            self._glfw.swap_interval(1)  # 垂直同步

            # ---- Step 3: 初始化 MuJoCo 渲染资源 ----
            # 场景对象
            self._scene = self._mujoco.MjvScene(self._model, maxgeom=10000)

            # OpenGL 上下文
            self._gl_context = self._mujoco.MjrContext(
                self._model, self._mujoco.mjtFontScale.mjFONTSCALE_150
            )

            # 自由摄像机（可用鼠标旋转/缩放）
            self._camera = self._mujoco.MjvCamera()
            self._camera.type = self._mujoco.mjtCamera.mjCAMERA_FREE
            # 设置初始视角
            self._camera.lookat[:] = [0.3, 0.0, 0.5]   # 看向场景中心
            self._camera.distance = 2.0                   # 距离
            self._camera.azimuth = 160.0                  # 水平角
            self._camera.elevation = -25.0                # 俯仰角

            # 摄像机控制状态
            self._cam_button_left = False
            self._cam_button_middle = False
            self._cam_button_right = False
            self._cam_last_x = 0
            self._cam_last_y = 0

            # 注册鼠标和键盘回调
            self._glfw.set_mouse_button_callback(self._window, self._mouse_button_callback)
            self._glfw.set_cursor_pos_callback(self._window, self._mouse_move_callback)
            self._glfw.set_scroll_callback(self._window, self._scroll_callback)
            self._glfw.set_key_callback(self._window, self._key_callback)

            self._viewer_running = True
            print(f'MuJoCo viewer launched ({width}x{height}).')
            print('  Mouse: Left-drag=rotate, Right-drag=translate, Scroll=zoom')
            print('  Keys:  Esc=close, R=reset, Space=pause')
            return True

        except Exception as e:
            print(f'Failed to launch viewer: {e}')
            self._cleanup_viewer()
            return False

    def _mouse_button_callback(self, window, button, action, mods):
        """鼠标按键回调：用于摄像机控制。"""
        if action == self._glfw.PRESS:
            x, y = self._glfw.get_cursor_pos(window)
            self._cam_last_x = x
            self._cam_last_y = y
            if button == self._glfw.MOUSE_BUTTON_LEFT:
                self._cam_button_left = True
            elif button == self._glfw.MOUSE_BUTTON_MIDDLE:
                self._cam_button_middle = True
            elif button == self._glfw.MOUSE_BUTTON_RIGHT:
                self._cam_button_right = True
        else:  # RELEASE
            if button == self._glfw.MOUSE_BUTTON_LEFT:
                self._cam_button_left = False
            elif button == self._glfw.MOUSE_BUTTON_MIDDLE:
                self._cam_button_middle = False
            elif button == self._glfw.MOUSE_BUTTON_RIGHT:
                self._cam_button_right = False

    def _mouse_move_callback(self, window, xpos, ypos):
        """鼠标移动回调：旋转/平移摄像机。"""
        if self._camera is None:
            return

        dx = xpos - self._cam_last_x
        dy = ypos - self._cam_last_y
        self._cam_last_x = xpos
        self._cam_last_y = ypos

        # 不修改摄像机（避免干扰 scene update）
        # 直接更新摄像机参数
        if self._cam_button_left:
            # 左键拖动 = 旋转
            self._camera.azimuth += dx * 0.3
            self._camera.elevation += dy * 0.3
            self._camera.elevation = max(-90, min(90, self._camera.elevation))
        elif self._cam_button_right:
            # 右键拖动 = 平移
            self._camera.lookat[0] += dx * 0.003 * self._camera.distance
            self._camera.lookat[1] -= dy * 0.003 * self._camera.distance
        elif self._cam_button_middle:
            # 中键 = 同时平移
            self._camera.lookat[0] += dx * 0.003 * self._camera.distance
            self._camera.lookat[1] -= dy * 0.003 * self._camera.distance

    def _scroll_callback(self, window, xoffset, yoffset):
        """滚轮回调：缩放。"""
        if self._camera is not None:
            self._camera.distance *= (1.0 - yoffset * 0.1)
            self._camera.distance = max(0.1, min(20.0, self._camera.distance))

    def _key_callback(self, window, key, scancode, action, mods):
        """键盘回调。"""
        if action == self._glfw.PRESS:
            if key == self._glfw.KEY_ESCAPE:
                self._viewer_running = False
            elif key == self._glfw.KEY_R:
                # 重置视角
                self._camera.lookat[:] = [0.3, 0.0, 0.5]
                self._camera.distance = 2.0
                self._camera.azimuth = 160.0
                self._camera.elevation = -25.0

    def sync_viewer(self):
        """
        渲染一帧到可视化窗口。

        必须在渲染循环中周期性调用（通常每步仿真后）。
        如果窗口被关闭，自动标记为非运行状态。
        """
        if not self._viewer_running or self._window is None:
            return

        if self._glfw.window_should_close(self._window):
            self._viewer_running = False
            print('Viewer window closed by user.')
            return

        try:
            # Step 1: 更新 MuJoCo 场景
            option = self._mujoco.MjvOption()
            self._mujoco.mjv_updateScene(
                self._model, self._data,
                option, None,
                self._camera,
                self._mujoco.mjtCatBit.mjCAT_ALL,
                self._scene
            )

            # Step 2: 渲染到窗口
            viewport = self._mujoco.MjrRect(0, 0, self._viewer_width, self._viewer_height)
            self._mujoco.mjr_render(viewport, self._scene, self._gl_context)

            # Step 3: 交换缓冲区 + 处理事件
            self._glfw.swap_buffers(self._window)
            self._glfw.poll_events()

        except Exception as e:
            print(f'Render error: {e}')
            self._viewer_running = False

    def close_viewer(self):
        """关闭可视化窗口。"""
        self._viewer_running = False
        self._cleanup_viewer()

    def _cleanup_viewer(self):
        """清理 GL 资源。"""
        if self._gl_context is not None:
            self._gl_context.free()
            self._gl_context = None
        if self._window is not None:
            self._glfw.destroy_window(self._window)
            self._window = None
        self._scene = None
        self._camera = None
        print('Viewer closed.')

    def is_viewer_running(self):
        """检查可视化窗口是否还在运行。"""
        return self._viewer_running

    # ================================================================
    # Simulation
    # ================================================================
    def step(self, action, n_substeps=None, sync_viewer=True):
        """
        执行一步（或多步）物理仿真。

        Args:
            action: 控制指令，shape=(nu,)，位置控制模式下即目标关节角度
            n_substeps: 本次 step 内的物理子步数（None 则使用 config 默认值）
            sync_viewer: 是否在步进后同步可视化窗口

        Returns:
            (observation, reward, done, info)
        """
        if self._try_init():
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
        """直接访问 MuJoCo mjData。"""
        return self._data

    @property
    def model(self):
        """直接访问 MuJoCo mjModel。"""
        return self._model


if __name__ == '__main__':
    print('=' * 60)
    print('MuJoCo Environment Demo (with viewer)')
    print('=' * 60)

    env = MuJoCoEnv()
    obs = env.reset()

    # 启动可视化
    if env.launch_viewer():
        try:
            for i in range(1000):
                action = np.random.randn(env.action_dim) * 0.01
                obs, _, _, _ = env.step(action, n_substeps=5)

                if not env.is_viewer_running():
                    break
                if i % 100 == 0:
                    print(f'  Step {i}')
        except KeyboardInterrupt:
            print('Interrupted.')
        finally:
            env.close_viewer()
    else:
        print('Viewer unavailable, running headless...')
        for i in range(100):
            env.step(np.zeros(env.action_dim))

    print('Done!')