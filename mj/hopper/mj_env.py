import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState

# scene_path = "C:/Disk/Data/New_Project/VScode_project/py/mj/model/franka_emika_panda/scene.xml"
hopper_path = "C:/Disk/Data/New_Project/VScode_project/py/mj/model/hopper/hopper.xml"
# go2_path = "C:/Disk/Data/New_Project/VScode_project/py/mj/model/unitree_h1/scene.xml"
# panda_scene_path = "/home/ub22/ros2_ws/src/my_mujoco_learn/models/panda_scene.xml"

class my_mujoco_env:
    def __init__(self):
        self.mj_model = mujoco.MjModel.from_xml_path(hopper_path)
        self.mj_data  = mujoco.MjData(self.mj_model)
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.mj_model.opt.timestep = 0.005  # 200Hz
        self.locker = threading.Lock()


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
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i)
            if name is None:
                print("actuator_index:", i, ", name:", name)
        print(" ")

if __name__ == "__main__":
    TEST = 1
    if TEST == 1:
        print("**********mj_env : TEST**********")
        env = my_mujoco_env()
        mj_model = env.model()
        mj_data = env.data()

        env.PrintSceneInformation()

        print(mj_model.nq)
        print(mj_model.nv)
        print(mj_model.nu)

        print(mj_data.qpos)
        print(mj_data.qvel)
        print(mj_data.ctrl)
