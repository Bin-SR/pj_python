# -*- coding: utf-8 -*-
"""
06_embodied/go2_robot.py - Unitree Go2 Quadruped Robot
Provides: trot gait, velocity control, navigation, VLA integration.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mujoco_env import MuJoCoEnv


class Go2Robot(MuJoCoEnv):
    """Unitree Go2 quadruped: 12-DoF, trot gait."""

    FR = 0; FL = 1; RR = 2; RL = 3

    def __init__(self, model_xml_path=None, config=None):
        super().__init__(model_xml_path, config)
        self._base_position = np.zeros(3)
        self._gait_phase = 0.0
        print('Unitree Go2 ready.')

    def generate_trot_gait(self, phase, step_height=0.05, step_length=0.1):
        """Trot gait: diagonal pairs move together."""
        phase = phase % (2 * np.pi)
        diag1_swing = np.sin(phase) > 0
        diag2_swing = np.sin(phase + np.pi) > 0
        foot_pos = {}
        for leg in range(4):
            swing = diag1_swing if leg in [self.FR, self.RL] else diag2_swing
            if swing:
                foot_pos[leg] = np.array([step_length, 0.0, -step_height])
            else:
                foot_pos[leg] = np.array([0.0, 0.0, 0.0])
        return foot_pos

    def velocity_control(self, vx, vy, vyaw, duration=1.0, dt=0.02):
        """Control with velocity command."""
        steps = int(duration / dt)
        trajectory = []
        for i in range(steps):
            t = i * dt
            phase = 2 * np.pi * t * 2.0
            fp = self.generate_trot_gait(phase)
            trajectory.append({'time': t, 'foot_positions': fp})
        return trajectory

    def navigate_to(self, target_pos, current_pos=None):
        """Navigate to target position."""
        if current_pos is None:
            current_pos = self._base_position.copy()
        delta = target_pos - current_pos
        dist = np.linalg.norm(delta[:2])
        angle = np.arctan2(delta[1], delta[0])
        vx = min(dist, 0.5) * np.cos(angle)
        vy = min(dist, 0.5) * np.sin(angle)
        dur = dist / max(np.sqrt(vx**2 + vy**2), 0.01)
        traj = self.velocity_control(vx, vy, 0.0, duration=min(dur, 2.0))
        self._base_position = target_pos.copy()
        print(f'Navigated to {target_pos} ({dist:.2f}m)')
        return traj

    def execute_vla_navigation(self, instruction, camera_image=None):
        """VLA navigation: instruction -> target -> navigate."""
        il = instruction.lower()
        if 'blue' in il: target = np.array([2.0, 0.0, 0.0])
        elif 'red' in il: target = np.array([1.0, 1.0, 0.0])
        elif 'green' in il: target = np.array([-1.0, 2.0, 0.0])
        else: target = np.array([1.0, 0.0, 0.0])
        print(f'VLA Nav: instruction -> target {target}')
        return self.navigate_to(target)


if __name__ == '__main__':
    print('=' * 60)
    print('Unitree Go2 Demo')
    print('=' * 60)
    go2 = Go2Robot()
    gait = go2.generate_trot_gait(0.0)
    print(f'Trot gait: { {k: v.tolist() for k, v in gait.items()} }')
    traj = go2.velocity_control(0.3, 0.0, 0.0, 0.5)
    print(f'Vel control: {len(traj)} steps')
    nav = go2.navigate_to(np.array([2.0, 1.0, 0.0]))
    print(f'Navigation: {len(nav)} steps')
    vla = go2.execute_vla_navigation('go to the blue marker')
    print(f'VLA nav: {len(vla)} steps')
    print('Done!')