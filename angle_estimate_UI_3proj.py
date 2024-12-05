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
CAMERA_INDEX = 0 # 默认使用第一个摄像头
FPS = 30
MODEL_FOLDER = "openpose/models/"
NET_RESOLUTION = "480x-1"

'''数学上的向量角度计算和骨骼点读取'''
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


'''不同测量项目的夹角计算函数，用来简介update_ui函数'''
# 计算肩膀、肘部、手腕之间的角度（胳膊外展）
def calculate_abduction_angle(skel):
    pnts = get_four_points(skel, [1, 5, 6, 7])  # 获取肩部、肘部、手腕的四个关键点
    if len(pnts) == 4:
        shoulder_to_elbow_vector = np.array(pnts[1]) - np.array(pnts[0])
        elbow_to_wrist_vector = np.array(pnts[2]) - np.array(pnts[1])
        return angle_between_vectors(shoulder_to_elbow_vector, elbow_to_wrist_vector)
    return -1

# 计算肘部合拢角度
def calculate_elbow_convergence_angle(skel):
    pnts = get_four_points(skel, [1, 2, 3, 4])  # 获取肩部、肘部、手腕的四个关键点
    if len(pnts) == 4:
        shoulder_to_elbow_vector = np.array(pnts[1]) - np.array(pnts[0])
        elbow_to_wrist_vector = np.array(pnts[2]) - np.array(pnts[1])
        return angle_between_vectors(shoulder_to_elbow_vector, elbow_to_wrist_vector)
    return -1

'''主要绘制函数'''
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

    # 计算并显示关节角度
    joint_angles = {
        'neck': [1, 0, 8],  # 颈部
        'left_shoulder': [1, 5, 6],  # 左肩
        'right_shoulder': [1, 2, 3],  # 右肩
        'left_elbow': [5, 6, 7],  # 左肘
        'right_elbow': [2, 3, 4],  # 右肘
        'left_wrist': [6, 7, 4],  # 左手腕
        'right_wrist': [3, 4, 2],  # 右手腕
        'left_hip': [8, 12, 13],  # 左髋
        'right_hip': [8, 9, 10],  # 右髋
        'left_knee': [12, 13, 14],  # 左膝
        'right_knee': [9, 10, 11],  # 右膝
        'left_ankle': [13, 14, 12],  # 左脚踝
        'right_ankle': [10, 11, 9],  # 右脚踝
        'chest': [1, 19, 8],  # 胸部
    }

    for joint_name, pos_list in joint_angles.items():
        pnts = get_angle_point(skel, pos_list)
        if len(pnts) == 3:
            angle = angle_between_points(pnts[0], pnts[1], pnts[2])
            if angle != -1:
                cv2.putText(img, f"{joint_name} angle: {int(angle)}", pnts[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img

# 更新UI的函数
def update_ui():
    global current_project, measurement_results
    ret, frame = dev.read()  # 从摄像头读取帧
    if not ret:
        root.after(100, update_ui)
        return

    vis_output = frame.copy()

    # OpenPose处理
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # 绘制骨架和角度
    if datum.poseKeypoints is not None:# 如果检测到人体
        for i in range(datum.poseKeypoints.shape[0]):# 遍历每个人体
            skel = datum.poseKeypoints[i, :25, :2].astype(np.int32)  # 使用25个关键点
            vis_output = draw_from_numpy(vis_output, skel)

            if current_project == 0:  # 项目1：胳膊后伸
                # 计算胳膊大臂和身躯脊柱主干的夹角
                pnts_left_arm = get_four_points(skel, [1, 5, 6, 8])
                # pnts_right_arm = get_four_points(skel, [1, 2, 3, 8])

                if len(pnts_left_arm) == 4:
                    neck_to_hip_vector_1 = np.array(pnts_left_arm[3]) - np.array(pnts_left_arm[0])
                    shoulder_to_elbow_vector = np.array(pnts_left_arm[2]) - np.array(pnts_left_arm[1])
                    angle_left = angle_between_vectors(neck_to_hip_vector_1, shoulder_to_elbow_vector)
                    if angle_left != -1:
                        measurement_results[0] = int(angle_left)  # 更新测量结果
                        update_angle_label(angle_left, "左胳膊和身躯夹角: {}°".format(int(angle_left)), "后伸")
            
            elif current_project == 1:  # 项目2：胳膊外展
                # 测量胳膊外展角度
                pnts_proj2_left = get_four_points(skel, [1, 5, 7, 8])
                # pnts_proj2_right = get_four_points(skel, [1, 2, 3, 4])

                if len(pnts_proj2_left) == 4:
                    shoulder_to_wrist_vector = np.array(pnts_proj2_left[2]) - np.array(pnts_proj2_left[1])
                    neck_to_hip_vector_2 = np.array(pnts_proj2_left[3]) - np.array(pnts_proj2_left[0])
                    abduction_angle_left = angle_between_vectors(shoulder_to_wrist_vector, neck_to_hip_vector_2)

                    if abduction_angle_left != -1:
                        measurement_results[1] = int(abduction_angle_left)  # 更新测量结果
                        update_angle_label(abduction_angle_left, "左胳膊外展角度: {}°".format(int(abduction_angle_left)), "外展")

            elif current_project == 2:  # 项目3：肘部合拢
                # 测量肘部合拢角度
                elbow_convergence_angle_left = calculate_elbow_convergence_angle(skel)#简化测试，只测量左肘部

                if elbow_convergence_angle_left != -1:
                    measurement_results[2] = int(elbow_convergence_angle_left)  # 更新测量结果
                    update_angle_label(elbow_convergence_angle_left, "左肘部合拢角度: {}°".format(int(elbow_convergence_angle_left)), "肘部")

    # 更新实时图像
    img = Image.fromarray(cv2.cvtColor(vis_output, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # 每隔一定时间更新一次
    root.after(100, update_ui)

def update_angle_label(angle, text, project_type):
    if project_type == "后伸":
        # print("后伸")
        if angle > 50:
            angle_label.config(text=text, foreground="green")
            root.config(bg="green")
            show_label1.config(borderwidth=2, relief="solid")
            show_label2.config(borderwidth=0, relief="flat")
            show_img1_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_1).resize((256, 340)))
            show_img2_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_2).resize((256, 340)))
        else:
            angle_label.config(text=text, foreground="red")
            root.config(bg="red")
            show_label1.config(borderwidth=0, relief="flat")
            show_label2.config(borderwidth=2, relief="solid")
            show_img1_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_1).resize((256, 340)))
            show_img2_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_2).resize((256, 340)))
    
    elif project_type == "外展":
        # print("外展")
        if angle > 170:
            angle_label.config(text=text, foreground="green")
            root.config(bg="green")
            show_label1.config(borderwidth=2, relief="solid")
            show_label2.config(borderwidth=0, relief="flat")
            show_img1_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_3).resize((256, 340)))
            show_img2_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_4).resize((256, 340)))
        else:
            angle_label.config(text=text, foreground="red")
            root.config(bg="red")
            show_label1.config(borderwidth=0, relief="flat")
            show_label2.config(borderwidth=2, relief="solid")
            show_img1_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_3).resize((256, 340)))
            show_img2_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_4).resize((256, 340)))

    elif project_type == "肘部":
        # print("肘部")
        if angle < 30:
            angle_label.config(text=text, foreground="green")
            root.config(bg="green")
            show_label1.config(borderwidth=2, relief="solid")
            show_label2.config(borderwidth=0, relief="flat")
            show_img1_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_5).resize((256, 340)))
            show_img2_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_6).resize((256, 340)))
        else:
            angle_label.config(text=text, foreground="red")
            root.config(bg="red")
            show_label1.config(borderwidth=0, relief="flat")
            show_label2.config(borderwidth=2, relief="solid")
            show_img1_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_5).resize((256, 340)))
            show_img2_tk = ImageTk.PhotoImage(Image.open(IMAGE_PATH_SHOW_6).resize((256, 340)))
    
    show_label1.config(image=show_img1_tk)
    show_label1.image = show_img1_tk
    show_label2.config(image=show_img2_tk)
    show_label2.image = show_img2_tk


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

# 添加项目状态和测量结果
current_project = 0  # 0: 后伸, 1: 外展, 2: 肘部
measurement_results = [-1, -1, -1]  # 存储三个项目的角度结果

def save_and_next_project(event):
    global current_project, measurement_results
    # 保存当前项目角度
    if current_project == 0:
        print(f"后伸测量保存：{measurement_results[0]}°")
    elif current_project == 1:
        print(f"外展测量保存：{measurement_results[1]}°")
    elif current_project == 2:
        print(f"肘部测量保存：{measurement_results[2]}°")

    # 切换到下一个项目
    current_project += 1
    if current_project > 2:  # 测量结束
        print("所有项目测量完成！")
        print(f"最终测量结果：后伸 {measurement_results[0]}°, 外展 {measurement_results[1]}°, 肘部 {measurement_results[2]}°")
        root.quit()

# 创建主窗口
def create_main_window():
    # 创建主窗口
    root = tk.Tk()
    root.title("实时图像和骨架检测")

    # 左边显示实时图像和骨架
    left_frame = ttk.Frame(root)
    left_frame.pack(side=tk.LEFT, padx=10, pady=10)

    image_label = ttk.Label(left_frame)
    image_label.pack()

    # 右边显示角度和提示图片
    right_frame = ttk.Frame(root)
    right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    angle_label = ttk.Label(right_frame, text="胳膊和身躯夹角: -°")
    angle_label.pack(pady=10)

    # indicator_label = tk.Label(right_frame, width=10, height=2)  # 使用 tk.Label
    # indicator_label.pack(pady=10)

    show_label1 = ttk.Label(right_frame)
    show_label1.pack(side=tk.LEFT, pady=10, padx=10)

    show_label2 = ttk.Label(right_frame)
    show_label2.pack(side=tk.LEFT, pady=10, padx=10)

    return root, image_label, angle_label, show_label1, show_label2

# 监听ESC键退出
def on_key_press(event):
    if event.keysym == 'Escape':
        root.quit()

if __name__ == "__main__":
    # 初始化OpenPose
    opWrapper = init_openpose()

    # 初始化RGB摄像头
    dev = init_camera()

    # 创建主窗口
    root, image_label, angle_label, show_label1, show_label2 = create_main_window()
    # root.bind("<Key>", on_key_press)# 绑定ESC键退出

    # 绑定按键切换项目
    root.bind("<space>", save_and_next_project)  # 按空格键保存并切换项目

    # 启动更新UI的函数
    update_ui()

    # 运行主循环
    root.mainloop()

    # 释放资源
    cv2.destroyAllWindows()
    dev.release()