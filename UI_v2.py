import cv2
import numpy as np
import time
import math
import platform
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from openpose import pyopenpose as op

# 常量定义
IMAGE_PATH_SHOW_1 = "./img/shoulder_shen_1.png"
IMAGE_PATH_SHOW_2 = "./img/shoulder_shen_2.png"
IMAGE_PATH_SHOW_3 = "./img/shoulder_waizhan_1.png"  # 胳膊外展图片1
IMAGE_PATH_SHOW_4 = "./img/shoulder_waizhan_2.png"  # 胳膊外展图片2
IMAGE_PATH_SHOW_5 = "./img/elbow_1.png"  # 肘部图片1
IMAGE_PATH_SHOW_6 = "./img/elbow_2.png"  # 肘部图片2
CAMERA_INDEX = 0  # 默认使用第一个摄像头
FPS = 30
MODEL_FOLDER = "openpose/models/"
NET_RESOLUTION = "480x-1"

# 计算三个关节点之间的角度
def angle_between_points(p0, p1, p2):
    a = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
    b = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    c = (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2
    if a * b == 0:
        return -1.0
    return math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi

# 计算两个向量之间的角度
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 * norm_v2 == 0:
        return -1.0
    cos_theta = dot_product / (norm_v1 * norm_v2)
    return math.acos(cos_theta) * 180 / math.pi

# 获取特定关节的三个关键点
def get_angle_point(human, pos_list):
    pnts = []
    for i in pos_list:
        if human[i][0] == 0 and human[i][1] == 0:  # 如果坐标为0，表示关键点丢失
            return []
        pnts.append((int(human[i][0]), int(human[i][1])))
    return pnts

# 获取特定关节的四个关键点
def get_four_points(human, pos_list):
    pnts = []
    for i in pos_list:
        if human[i][0] == 0 and human[i][1] == 0:  # 如果坐标为0，表示关键点丢失
            return []
        pnts.append((int(human[i][0]), int(human[i][1])))
    return pnts

# 绘制骨架和角度
def draw_from_numpy(img_ori, skel):
    img = img_ori.copy()
    pairs = [(1, 8), (1, 0), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6),
             (6, 7), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
             (0, 15), (15, 17), (0, 16), (16, 18), (14, 19), (19, 20), (11, 22), (22, 23)]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 85, 170], [85, 170, 255], [170, 255, 85]]
    
    body_parts = []
    
    # 绘制关节点
    for kk in range(len(skel)):
        k = skel[kk]
        body_part = (int(k[0]), int(k[1]))
        if body_part != (0, 0):
            cv2.circle(img, body_part, 4, (0, 0, 255), 4)  # 绘制关节点
        body_parts.append(body_part)

    # 绘制骨骼连线
    for idx, pair in enumerate(pairs):
        p1 = body_parts[pair[0]]
        p2 = body_parts[pair[1]]
        if p1 != (0, 0) and p2 != (0, 0):
            color = colors[idx % len(colors)]
            cv2.line(img, p1, p2, color, 4)  # 绘制连线

    #此处删除了绘制角度的代码，因为在UI中不再需要绘制角度
    return img

# 更新UI的函数
def update_ui():
    global countdown, current_task
    ret, frame = dev.read()  # 从摄像头读取帧
    if not ret:
        root.after(100, update_ui)
        return

    vis_output = frame.copy()

    # OpenPose处理
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # 只在开始动作时显示倒计时
    if countdown > 0:
        countdown_label.config(text=f"倒计时: {countdown}")
    elif countdown == 0:
        # 开始动作检测并显示结果
        action_result = perform_action_check(datum.poseKeypoints, current_task)
        action_results[current_task] = action_result
        show_action_result(current_task)
        current_task += 1
        if current_task < 3:
            # 等待2秒后继续下一个动作
            root.after(2000, start_next_task)

    # 更新实时图像
    img = Image.fromarray(cv2.cvtColor(vis_output, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # 每隔一段时间更新一次
    root.after(100, update_ui)

# 动作检测
def perform_action_check(pose_keypoints, task):
    if task == 0:
        # 测量胳膊后伸展角度
        return check_shoulder_extension(pose_keypoints)
    elif task == 1:
        # 测量胳膊外展角度
        return check_shoulder_abduction(pose_keypoints)
    elif task == 2:
        # 测量肘部角度
        return check_elbow_angle(pose_keypoints)
    return False

# 胳膊后伸展角度检查
def check_shoulder_extension(pose_keypoints):
    # 根据已知骨架关键点，计算胳膊后伸展角度
    # 假设计算方法和之前的类似
    # 计算并返回是否符合条件
    if pose_keypoints is None or len(pose_keypoints) == 0:
        return False
    
    shoulder_right = get_angle_point(pose_keypoints[0], [2, 3, 4])
    shoulder_left = get_angle_point(pose_keypoints[0], [5, 6, 7])
    
    if not shoulder_right or not shoulder_left:
        return False
    
    angle_right = angle_between_points(*shoulder_right)
    angle_left = angle_between_points(*shoulder_left)
    
    # 判断是否符合标准，例如角度大于150度
    if angle_right > 150 or angle_left > 150:
        return True
    return False

# 胳膊外展角度检查
def check_shoulder_abduction(pose_keypoints):
    # 计算脊柱与胳膊外展向量的夹角，检查是否大于170度
    if pose_keypoints is None or len(pose_keypoints) == 0:
        return False
    
    shoulder_right = get_four_points(pose_keypoints[0], [1, 2, 3, 4])
    shoulder_left = get_four_points(pose_keypoints[0], [1, 5, 6, 7])
    
    if not shoulder_right or not shoulder_left:
        return False
    
    spine_vector = np.array(shoulder_right[1]) - np.array(shoulder_right[0])
    arm_vector_right = np.array(shoulder_right[3]) - np.array(shoulder_right[2])
    arm_vector_left = np.array(shoulder_left[3]) - np.array(shoulder_left[2])
    
    angle_right = angle_between_vectors(spine_vector, arm_vector_right)
    angle_left = angle_between_vectors(spine_vector, arm_vector_left)
    
    # 判断是否符合标准，例如角度大于170度
    if angle_right > 170 or angle_left > 170:
        return True
    return False

# 肘部角度检查
def check_elbow_angle(pose_keypoints):
    if pose_keypoints is None or len(pose_keypoints) == 0:
        return False
    
    elbow_right = get_angle_point(pose_keypoints[0], [2, 3, 4])
    elbow_left = get_angle_point(pose_keypoints[0], [5, 6, 7])
    
    if not elbow_right or not elbow_left:
        return False
    
    angle_right = angle_between_points(*elbow_right)
    angle_left = angle_between_points(*elbow_left)
    
    # 判断是否符合标准，例如角度小于30度
    if angle_right < 30 or angle_left < 30:
        return True
    return False

# 显示每个动作结果
def show_action_result(task):
    if task == 0:
        result_label.config(text=f"胳膊后伸展: {action_results[0]}")
        show_img1_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_1).resize((256, 340)))
        show_img2_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_2).resize((256, 340)))
        show_label1.config(image=show_img1_tk)
        show_label1.image = show_img1_tk
        show_label2.config(image=show_img2_tk)
        show_label2.image = show_img2_tk
    elif task == 1:
        result_label.config(text=f"胳膊外展: {action_results[1]}")
        show_img3_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_3).resize((256, 340)))
        show_img4_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_4).resize((256, 340)))
        show_label1.config(image=show_img3_tk)
        show_label1.image = show_img3_tk
        show_label2.config(image=show_img4_tk)
        show_label2.image = show_img4_tk
    elif task == 2:
        result_label.config(text=f"肘部角度: {action_results[2]}")
        show_img5_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_5).resize((256, 340)))
        show_img6_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_6).resize((256, 340)))
        show_label1.config(image=show_img5_tk)
        show_label1.image = show_img5_tk
        show_label2.config(image=show_img6_tk)
        show_label2.image = show_img6_tk

# 启动下一个动作
def start_next_task():
    global countdown
    countdown = 5  # 设置倒计时为5秒
    update_ui()

# 更新倒计时
def update_countdown():
    global countdown
    if countdown > 0:
        countdown -= 1
    root.after(1000, update_countdown)

# 创建主窗口
def create_main_window():
    global result_label, countdown_label, show_label1, show_label2
    # 创建主窗口
    root = tk.Tk()
    root.title("动作检测与评估")

    # 左边显示实时图像和骨架
    left_frame = ttk.Frame(root)
    left_frame.pack(side=tk.LEFT, padx=10, pady=10)

    image_label = ttk.Label(left_frame)
    image_label.pack()

    # 右边显示角度和提示图片
    right_frame = ttk.Frame(root)
    right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    result_label = ttk.Label(right_frame, text="动作结果：")
    result_label.pack(pady=10)

    countdown_label = ttk.Label(right_frame, text="倒计时: 5")
    countdown_label.pack(pady=10)

    show_label1 = ttk.Label(right_frame)
    show_label1.pack(side=tk.LEFT, pady=10, padx=10)

    show_label2 = ttk.Label(right_frame)
    show_label2.pack(side=tk.LEFT, pady=10, padx=10)

    return root, image_label, result_label, countdown_label, show_label1, show_label2

# 初始化OpenPose
def init_openpose():
    # OpenPose参数设置
    params = dict()
    params["model_folder"] = MODEL_FOLDER
    params["net_resolution"] = NET_RESOLUTION
    # 启动OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    return opWrapper

# 初始化摄像头
def init_camera():
    if platform.system().lower() == 'windows':
        dev = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        dev = cv2.VideoCapture(CAMERA_INDEX)
    dev.set(cv2.CAP_PROP_FPS, FPS)# 设置摄像头FPS
    return dev


# 启动程序
if __name__ == "__main__":
    # 初始化OpenPose
    opWrapper = init_openpose()

    # 初始化RGB摄像头
    dev = init_camera()

    # 初始化变量
    action_results = [None, None, None]
    countdown = 5
    current_task = 0

    # 创建主窗口
    root, image_label, result_label, countdown_label, show_label1, show_label2 = create_main_window()

    # 启动倒计时
    update_countdown()

    # 启动更新UI的函数
    update_ui()

    # 运行主循环
    root.mainloop()

    # 释放资源
    cv2.destroyAllWindows()
    dev.release()
