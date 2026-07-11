# -*- coding: utf-8 -*-
"""
06_embodied/franka_panda.py - Franka Emika Panda 7-DoF Robotic Arm
Provides: FK, IK, position/velocity control, pick-and-place, VLA integration.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mujoco_env import MuJoCoEnv, EnvConfig


class FrankaPanda(MuJoCoEnv):
    """Franka Emika Panda: 7 joints + gripper."""

    JOINT_LIMITS = np.array([
        [-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
        [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.8973, 2.8973],
    ])

    def __init__(self, model_xml_path=None, config=None):
        super().__init__(model_xml_path, config)
        self._gripper_open = 0.04
        self._gripper_closed = 0.0
        print('Franka Panda arm ready.')

    def forward_kinematics(self, joint_angles):
        """Compute end-effector position from joint angles (simplified FK)."""
        q = np.asarray(joint_angles)
        l1, l2, l3 = 0.333, 0.316, 0.25
        x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]) + l3 * np.cos(q[0] + q[1] + q[3])
        y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1]) + l3 * np.sin(q[0] + q[1] + q[3])
        z = 0.5 + 0.1 * np.sin(q[2])
        return np.array([x, y, z])

    def inverse_kinematics(self, target_pos, initial_guess=None):
        """Compute joint angles via numerical IK (gradient descent)."""
        if initial_guess is None:
            initial_guess = np.zeros(7)
        q = np.asarray(initial_guess, dtype=np.float64).copy()
        target = np.asarray(target_pos)
        for _ in range(200):
            current = self.forward_kinematics(q)
            error = target - current
            if np.linalg.norm(error) < 0.001:
                break
            eps = 0.001
            J = np.zeros((3, 7))
            for j in range(7):
                q_plus = q.copy()
                q_plus[j] += eps
                J[:, j] = (self.forward_kinematics(q_plus) - current) / eps
            dq = np.linalg.pinv(J) @ error * 0.1
            q += dq
            for j in range(7):
                q[j] = np.clip(q[j], self.JOINT_LIMITS[j, 0], self.JOINT_LIMITS[j, 1])
        return q

    def move_to(self, target_pos, steps=50):
        """Move end-effector to target with linear interpolation."""
        current_q = np.zeros(7)
        if hasattr(self, '_state'):
            current_q = self._state['qpos'].copy()
        target_q = self.inverse_kinematics(target_pos, current_q)
        trajectory = []
        for t in range(steps):
            alpha = (t + 1) / steps
            trajectory.append(current_q + alpha * (target_q - current_q))
        return trajectory

    def pick_and_place(self, pick_pos, place_pos):
        """Execute a pick-and-place task."""
        approach = pick_pos + np.array([0, 0, 0.1])
        t1 = self.move_to(approach, 30)
        t2 = self.move_to(pick_pos, 20)
        self._grasped_object = 'block'
        t3 = self.move_to(approach, 20)
        place_above = place_pos + np.array([0, 0, 0.1])
        t4 = self.move_to(place_above, 50)
        t5 = self.move_to(place_pos, 20)
        self._grasped_object = None
        t6 = self.move_to(place_above, 20)
        return t1 + t2 + t3 + t4 + t5 + t6

    def execute_vla_action(self, action_vector):
        """Execute VLA-predicted action: [dx, dy, dz, roll, pitch, yaw, gripper]."""
        dx, dy, dz = action_vector[:3]
        current_q = np.zeros(7)
        if hasattr(self, '_state'):
            current_q = self._state['qpos'].copy()
        current_pos = self.forward_kinematics(current_q)
        target = current_pos + np.array([dx, dy, dz]) * 0.05
        return self.move_to(target, 10)


if __name__ == '__main__':
    print('=' * 60)
    print('Franka Panda Demo')
    print('=' * 60)
    panda = FrankaPanda()
    pos = panda.forward_kinematics(np.zeros(7))
    print(f'FK (zero joints): {pos}')
    target = np.array([0.5, 0.0, 0.4])
    q = panda.inverse_kinematics(target)
    print(f'IK target: {target} -> joints: {q}')
    print(f'IK error: {np.linalg.norm(panda.forward_kinematics(q) - target):.4f}')
    traj = panda.pick_and_place(np.array([0.4, 0.2, 0.2]), np.array([0.6, -0.1, 0.3]))
    print(f'Pick-place trajectory: {len(traj)} steps')
    print('Done!')