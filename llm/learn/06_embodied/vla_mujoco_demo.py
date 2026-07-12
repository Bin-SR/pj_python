# -*- coding: utf-8 -*-
"""
06_embodied/vla_mujoco_demo.py - VLA + MuJoCo 可视化集成演示

端到端具身智能流程（带实时可视化）：
  Camera -> Perception -> VLM -> Action Head -> Robot -> MuJoCo Viewer

运行前确保：
  pip install mujoco numpy
  （PyTorch 可选，本 demo 使用 mock VLA 不依赖 PyTorch）

使用方式：
  python vla_mujoco_demo.py          # 运行 panda 演示
  python vla_mujoco_demo.py --go2    # 运行 Go2 演示
"""
import numpy as np
import sys, os, time, argparse

sys.path.insert(0, os.path.dirname(__file__))
from franka_panda import FrankaPanda
from go2_robot import Go2Robot
from perception import VisualPerceptionPipeline


# ================================================================
# VLA Pipeline (带可视化)
# ================================================================
class EmbodiedVLAPipeline:
    """
    端到端 VLA pipeline：感知 → 理解 → 动作 → 执行。

    每一步都会在 MuJoCo 可视化窗口中实时显示。
    """

    def __init__(self, robot_type='panda', perception=None,
                 launch_viewer=True):
        self.robot_type = robot_type
        self.perception = perception or VisualPerceptionPipeline()

        if robot_type == 'panda':
            self.robot = FrankaPanda()
        elif robot_type == 'go2':
            self.robot = Go2Robot()
        else:
            raise ValueError(f'Unknown robot type: {robot_type}')

        # 启动可视化
        if launch_viewer:
            self.robot.launch_viewer()
            time.sleep(0.5)  # 等窗口初始化

        self.state = 'idle'
        self.step_count = 0
        print(f'Embodied VLA Pipeline ready ({robot_type}).')

    def run_task(self, instruction: str, max_steps: int = 100):
        """
        执行自然语言任务指令。

        每一步的流程：
          1. 渲染当前画面 → 2. 感知检测物体 → 3. VLA 决策动作 → 4. 执行并可视化
        """
        print(f"\n{'='*60}")
        print(f"Task: \"{instruction}\"")
        print(f"Robot: {self.robot_type}")
        print(f"{'='*60}")

        self.state = 'running'
        step = 0
        ee_traj = []  # 记录末端轨迹

        while step < max_steps and self.robot.is_viewer_running():
            step += 1
            self.step_count = step

            # -------------------------------------------------------
            # Step 1: 感知 — 渲染当前画面 + 检测物体
            # -------------------------------------------------------
            image = self.robot.render()
            scene_desc = self.perception.format_for_vlm(image)

            # -------------------------------------------------------
            # Step 2: 决策 — VLA 根据指令和场景决定动作
            # （这里使用规则模拟 VLA，替换为真实 VLA 模型即可）
            # -------------------------------------------------------
            action, done = self._vla_decision(instruction, image, step)

            # -------------------------------------------------------
            # Step 3: 执行 — 将动作发送给机器人并驱动仿真
            # -------------------------------------------------------
            if self.robot_type == 'panda':
                self.robot.execute_vla_action(action, step_delay=0.005)
                # 记录末端位置
                ee_pos = self.robot.get_current_ee_pos()
                ee_traj.append(ee_pos)
            elif self.robot_type == 'go2':
                self.robot.execute_vla_navigation(instruction, image)

            # 打印状态
            if step % 5 == 0 or done:
                print(f"  Step {step:3d}: {scene_desc.split(chr(10))[0]}")
                if self.robot_type == 'panda':
                    print(f"         EE pos: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")

            if done:
                break

        print(f"\nTask complete after {step} steps.")
        print(f"Viewer still open: {self.robot.is_viewer_running()}")
        return {'steps': step, 'ee_trajectory': ee_traj}

    def _vla_decision(self, instruction: str, image: np.ndarray, step: int):
        """
        VLA 决策：根据指令和当前场景决定下一步动作。

        目前使用规则模拟（不依赖 PyTorch）。
        替换为真实的 VLA 模型即可实现端到端学习控制。

        返回: (action_vector, done_flag)
        """
        il = instruction.lower()

        if self.robot_type == 'panda':
            # ---- 任务: 捡起红色方块放到桌上 ----
            if 'pick' in il and ('red' in il or 'block' in il):
                if step <= 10:
                    # 阶段1: 移动到方块上方
                    return np.array([0.5, 0.2, 0.35, 0, 0, 0, 1.0]), False
                elif step <= 15:
                    # 阶段2: 下降抓取
                    return np.array([0.5, 0.2, 0.05, 0, 0, 0, 0.5]), False
                elif step <= 20:
                    # 阶段3: 抬起来
                    return np.array([0.5, 0.2, 0.4, 0, 0, 0, 1.0]), False
                elif step <= 35:
                    # 阶段4: 移动到桌子位置
                    return np.array([0.6, -0.1, 0.4, 0, 0, 0, 1.0]), False
                elif step <= 40:
                    # 阶段5: 放下
                    return np.array([0.6, -0.1, 0.05, 0, 0, 0, 0.0]), False
                else:
                    # 完成
                    return np.array([0.0, 0.0, 0.5, 0, 0, 0, 0.0]), True

            elif 'home' in il:
                return np.array([0.0, 0.0, 0.5, 0, 0, 0, 0.0]), step >= 15
            else:
                # 默认: 保持原位
                return np.array([0.0, 0.0, 0.5, 0, 0, 0, 0.0]), step >= 30

        else:
            # Go2 导航
            return np.zeros(3), step >= 20

    def cleanup(self):
        """清理资源，关闭可视化窗口。"""
        self.robot.close_viewer()
        print('Pipeline cleaned up.')


# ================================================================
# 预置场景
# ================================================================
class VLAMujocoScenario:
    """预置的 VLA + MuJoCo 演示场景。"""

    @staticmethod
    def panda_pick_and_place(launch_viewer=True):
        """Franka Panda 拾取放置任务（带可视化）。"""
        print('\n' + '=' * 60)
        print('Scenario: Franka Panda Pick-and-Place (VLA + Viewer)')
        print('=' * 60)
        print('A MuJoCo window should open. Watch the robot arm move!')
        print('Press Esc or close the window to stop.')
        print()

        pipeline = EmbodiedVLAPipeline(
            robot_type='panda',
            launch_viewer=launch_viewer
        )

        try:
            result = pipeline.run_task(
                'pick up the red block and place it on the table',
                max_steps=50
            )
        except KeyboardInterrupt:
            print('\nInterrupted by user.')
        finally:
            pipeline.cleanup()

        return result

    @staticmethod
    def go2_visual_navigation(launch_viewer=True):
        """Unitree Go2 视觉导航任务（带可视化）。"""
        print('\n' + '=' * 60)
        print('Scenario: Unitree Go2 Visual Navigation (VLA + Viewer)')
        print('=' * 60)

        pipeline = EmbodiedVLAPipeline(
            robot_type='go2',
            launch_viewer=launch_viewer
        )

        try:
            result = pipeline.run_task('go to the blue marker', max_steps=30)
        except KeyboardInterrupt:
            print('\nInterrupted by user.')
        finally:
            pipeline.cleanup()

        return result

    @staticmethod
    def interactive_control():
        """
        交互式控制模式：手动输入关节角度，实时观察机械臂运动。

        用于调试和理解关节空间与任务空间的关系。
        """
        print('\n' + '=' * 60)
        print('Interactive Panda Control')
        print('=' * 60)
        print('Enter 7 joint angles (radians) separated by spaces.')
        print('Example: 0 0 0 -1.5 0 1.5 0')
        print('Type "q" to quit, "home" for home position.\n')

        panda = FrankaPanda()
        panda.launch_viewer()
        time.sleep(0.5)

        try:
            while panda.is_viewer_running():
                cmd = input('Joint angles (7 values): ').strip()
                if cmd.lower() == 'q':
                    break
                if cmd.lower() == 'home':
                    target = np.zeros(7)
                else:
                    try:
                        target = np.array([float(x) for x in cmd.split()])
                        if len(target) != 7:
                            print('  Need exactly 7 values!')
                            continue
                    except ValueError:
                        print('  Invalid input!')
                        continue

                print(f'  Moving to: {target}')
                panda.move_joints(target, steps=80, step_delay=0.008)

        except KeyboardInterrupt:
            print('\nInterrupted.')
        finally:
            panda.close_viewer()

    @staticmethod
    def system_overview():
        """打印系统架构概览。"""
        print('=' * 60)
        print('Embodied VLA System Overview')
        print('=' * 60)
        print()
        print('Architecture:')
        print('  Layer 1: Perception')
        print('    Camera -> render() -> RGB Image (224x224)')
        print('    Object Detector -> scene description')
        print()
        print('  Layer 2: Understanding (VLM)')
        print('    Image tokens + Instruction tokens -> Transformer')
        print('    -> Hidden states with multimodal understanding')
        print()
        print('  Layer 3: Action (Action Head)')
        print('    Hidden states -> Action vector [dx,dy,dz,roll,pitch,yaw,grip]')
        print()
        print('  Layer 4: Execution (MuJoCo + Viewer)')
        print('    Action -> IK -> joint trajectory -> mj_step() -> viewer.sync()')
        print('    -> Visual feedback loop')
        print()
        print('Visualization:')
        print('  mujoco.viewer.launch_passive() -> native window')
        print('  Each step: set ctrl, mj_step(), viewer.sync()')
        print()
        print('RTX 3050: ~5M params, ~40MB VRAM, 10-50Hz control')


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VLA + MuJoCo Demo')
    parser.add_argument('--go2', action='store_true', help='Run Go2 demo instead of Panda')
    parser.add_argument('--interactive', action='store_true', help='Interactive joint control mode')
    parser.add_argument('--no-viewer', action='store_true', help='Run without viewer (headless)')
    args = parser.parse_args()

    use_viewer = not args.no_viewer

    VLAMujocoScenario.system_overview()

    if args.interactive:
        VLAMujocoScenario.interactive_control()
    elif args.go2:
        VLAMujocoScenario.go2_visual_navigation(launch_viewer=use_viewer)
    else:
        VLAMujocoScenario.panda_pick_and_place(launch_viewer=use_viewer)

    print('\nAll demos complete!')
    print('\nNext steps:')
    print('  1. Replace _vla_decision() with a real VLA model')
    print('  2. Train behavior cloning on collected demo data')
    print('  3. Add more complex tasks (stacking, assembly)')
    print('  4. Integrate real camera feed instead of render()')