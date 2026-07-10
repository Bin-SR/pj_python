#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS Noetic + Ubuntu20
Gazebo中识别并跟踪橙色球
摄像头: /kinect2/hd/image_color

功能：
1. 使用OpenAI库调用 DeepSeek 模型（兼容OpenAI接口）
2. OpenCV识别橙色球
3. ROS控制机器人跟踪目标

运行前安装:
pip install openai opencv-python

记得修改:
API_KEY
BASE_URL
MODEL_NAME
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from openai import OpenAI

# ==============================
# DeepSeek API 配置
# ==============================
API_KEY = "你的api_key"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

MODEL_NAME = "deepseek-chat"

# ==============================
# 全局变量
# ==============================
bridge = CvBridge()
frame = None


# ==============================
# 摄像头回调函数
# ==============================
def image_callback(msg):
    global frame
    frame = bridge.imgmsg_to_cv2(msg, "bgr8")


# ==============================
# 获取图像
# ==============================
def get_image():
    global frame
    return frame


# ==============================
# 调用 DeepSeek 模型（可选）
# 用于输出任务状态
# ==============================
def ask_deepseek(prompt_text):

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        return str(e)


# ==============================
# 检测橙色球
# 返回中心坐标
# ==============================
def detect_target(img):

    if img is None:
        return None, None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 橙色HSV范围（可调）
    lower = np.array([5, 120, 120])
    upper = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # 去噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None

    # 最大轮廓
    c = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(c)

    if area < 100:
        return None, None

    M = cv2.moments(c)

    if M["m00"] == 0:
        return None, None

    target_x = int(M["m10"] / M["m00"])
    target_y = int(M["m01"] / M["m00"])

    return target_x, target_y


# ==============================
# 跟踪目标
# ==============================
def track_target(pub, target_x, target_y, width, height):

    cmd = Twist()

    if target_x is None:
        # 没目标时旋转搜索
        cmd.angular.z = 0.3
        pub.publish(cmd)
        return

    # 图像中心
    center_x = width // 2
    center_y = height // 2

    # 误差
    error_x = target_x - center_x
    error_y = target_y - center_y

    # 比例控制参数
    kp_ang = 0.003
    kp_lin = 0.002

    # 左右转向
    cmd.angular.z = -kp_ang * error_x

    # 前进后退（球在图像偏上说明远）
    cmd.linear.x = kp_lin * (-error_y)

    # 限幅
    cmd.angular.z = max(min(cmd.angular.z, 0.6), -0.6)
    cmd.linear.x = max(min(cmd.linear.x, 0.4), -0.4)

    pub.publish(cmd)


# ==============================
# 主程序
# ==============================
def main():

    rospy.init_node("orange_ball_tracker")

    rospy.Subscriber("/kinect2/hd/image_color",
                     Image,
                     image_callback)

    pub = rospy.Publisher("/cmd_vel",
                          Twist,
                          queue_size=10)

    rate = rospy.Rate(10)

    rospy.loginfo("启动橙色球跟踪节点")

    # DeepSeek 输出启动信息
    text = ask_deepseek("请用一句话说明机器人正在执行橙色球跟踪任务")
    rospy.loginfo(text)

    while not rospy.is_shutdown():

        img = get_image()

        if img is not None:

            h, w, _ = img.shape

            target_x, target_y = detect_target(img)

            # 显示检测结果
            if target_x is not None:
                cv2.circle(img,
                           (target_x, target_y),
                           8,
                           (0, 255, 0),
                           -1)

            cv2.circle(img,
                       (w // 2, h // 2),
                       6,
                       (255, 0, 0),
                       -1)

            cv2.imshow("tracking", img)
            cv2.waitKey(1)

            track_target(pub,
                         target_x,
                         target_y,
                         w,
                         h)

        rate.sleep()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
import rospy
from openai import OpenAI
from move_api import move_forward, move_backward, turn_left, turn_right

client = OpenAI(
    api_key="你的DeepSeek_API_KEY",
    base_url="https://api.deepseek.com"
)

rospy.init_node("deepseek_robot")

system_prompt = """
你是移动机器人控制器。

你只能使用以下函数：

move_forward(距离)
move_backward(距离)
turn_left(角度)
turn_right(角度)

规则：
1. 只输出Python函数调用代码
2. 不要解释
3. 一行一个动作
"""

while not rospy.is_shutdown():

    user_cmd = input("请输入任务：")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_cmd}
        ]
    )

    code = response.choices[0].message.content

    print("\n模型输出：")
    print(code)

    exec(code)