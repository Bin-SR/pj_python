import cv2
import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

import numpy as np
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState

scene_path = "C:/Disk/Data/New_Project/VScode_project/py/mj/model/franka_emika_panda/scene2.xml"
# go2_path = "C:/Disk/Data/New_Project/VScode_project/py/mj/model/unitree_h1/scene.xml"
# panda_scene_path = "/home/ub22/ros2_ws/src/my_mujoco_learn/models/panda_scene.xml"

class my_mujoco_env:
    def __init__(self):
        self.mj_model = mujoco.MjModel.from_xml_path(scene_path)
        self.mj_data  = mujoco.MjData(self.mj_model)
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.mj_model.opt.timestep = 0.005  # 200Hz
        self.locker = threading.Lock()

        self.renderer = mujoco.Renderer(self.mj_model)

        viewer_thread = Thread(target=self.PhysicsViewerThread)
        sim_thread = Thread(target=self.SimulationThread)

        viewer_thread.start()
        sim_thread.start()

    def SimulationThread(self):
        while self.viewer.is_running():
            step_start = time.perf_counter()

            self.locker.acquire()

            mujoco.mj_step(self.mj_model, self.mj_data)

            self.locker.release()

            time_until_next_step = self.mj_model.opt.timestep - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def PhysicsViewerThread(self):
        while self.viewer.is_running():
            self.locker.acquire()
            self.viewer.sync()
            self.locker.release()
            time.sleep(0.02)
        exit()
    
    def model(self) -> mujoco.MjModel:
        return self.mj_model
    
    def data(self) -> mujoco.MjData:
        return self.mj_data
    
    def viewer(self) -> mujoco.viewer:
        return self.viewer
    
    def PrintSceneInformation(self):
        print(" ")

        print("<<------------- Link ------------->> ")
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                print("link_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Joint ------------->> ")
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                print("joint_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Actuator ------------->>")
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i
            )
            if name:
                print("actuator_index:", i, ", name:", name)
        print(" ")

    def get_rgb(self):

        self.renderer.update_scene(self.mj_data, camera="front_cam")
        img = self.renderer.render()

        # cv2.imshow("org", img)
        # cv2.waitKey(1)

        return img

TEST = 1
if TEST: 
    ARM_JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7"
    ]  
    ACTUATOR_NAMES = [
    "actuator1", "actuator2", "actuator3", "actuator4",
    "actuator5", "actuator6", "actuator7", "actuator8"
    ]
    if __name__ == "__main__":
        _env = my_mujoco_env()
        mj_model = _env.model()
        mj_data = _env.data()

        _arm_joint_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES]
        arm_qpos_adr = np.array([mj_model.jnt_qposadr[jid] for jid in _arm_joint_ids], dtype=np.int32)

        _actuator_ids = np.array([mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATOR_NAMES], dtype=np.int32)
        print(type(_arm_joint_ids))
        print(type(arm_qpos_adr))
        print(_actuator_ids)

        _cube_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
        cube_pos = mj_data.body(_cube_body_id).xpos.copy()
        print(cube_pos)

        while False:
            qpos = mj_data.qpos[arm_qpos_adr].copy()
            print(np.round(qpos, 3))
            time.sleep(1)
            # img = _env.get_rgb()
            # cv2.imshow("test", img)
            # cv2.waitKey(1)