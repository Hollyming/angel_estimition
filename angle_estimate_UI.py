from openpose import pyopenpose as op
import cv2
import numpy as np
import time
import math
import platform
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# 计算两个向量之间的角度
def angle_between_points(p0, p1, p2):
    a = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
    b = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    c = (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2
    if a * b == 0:
        return -1.0
    return math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi

# 获取特定关节的三个关键点
def get_angle_point(human, pos_list):
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
    if datum.poseKeypoints is not None:
        for i in range(datum.poseKeypoints.shape[0]):
            skel = datum.poseKeypoints[i, :25, :2].astype(np.int32)  # 使用25个关键点
            vis_output = draw_from_numpy(vis_output, skel)

            # 计算胳膊和身体夹角
            pnts = get_angle_point(skel, [1, 2, 3])
            if len(pnts) == 3:
                angle = angle_between_points(pnts[0], pnts[1], pnts[2])
                if angle != -1:
                    angle_label.config(text=f"胳膊和身体夹角: {int(angle)}°")
                    if angle > 50:
                        angle_label.config(foreground="green")
                        root.config(bg="green")
                        # indicator_label.config(bg="green")
                        show_img1_tk = ImageTk.PhotoImage(Image.open("./img/show_1.png").resize((256, 340)))
                        show_img2_tk = ImageTk.PhotoImage(Image.open("./img/show_2.png").resize((256, 340)))
                        show_label1.config(image=show_img1_tk)
                        show_label1.image = show_img1_tk
                        show_label2.config(image=show_img2_tk)
                        show_label2.image = show_img2_tk
                        show_label1.config(borderwidth=2, relief="solid")
                        show_label2.config(borderwidth=0, relief="flat")
                    else:
                        angle_label.config(foreground="red")
                        root.config(bg="red")
                        # indicator_label.config(bg="red")
                        show_img1_tk = ImageTk.PhotoImage(Image.open("./img/show_1.png").resize((256, 340)))
                        show_img2_tk = ImageTk.PhotoImage(Image.open("./img/show_2.png").resize((256, 340)))
                        show_label1.config(image=show_img1_tk)
                        show_label1.image = show_img1_tk
                        show_label2.config(image=show_img2_tk)
                        show_label2.image = show_img2_tk
                        show_label1.config(borderwidth=0, relief="flat")
                        show_label2.config(borderwidth=2, relief="solid")

    # 更新实时图像
    img = Image.fromarray(cv2.cvtColor(vis_output, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # 每隔一定时间更新一次
    root.after(100, update_ui)

if __name__ == "__main__":
    # OpenPose参数设置
    params = dict()
    params["model_folder"] = "openpose/models/"
    params["net_resolution"] = "480x-1"

    # 启动OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # 初始化RGB摄像头
    cam_idx = 0  # 默认使用第一个摄像头
    if platform.system().lower() == 'windows':
        dev = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    else:
        dev = cv2.VideoCapture(cam_idx)
    
    # 设置摄像头FPS
    dev.set(cv2.CAP_PROP_FPS, 30)

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

    angle_label = ttk.Label(right_frame, text="胳膊和身体夹角: -°")
    angle_label.pack(pady=10)

    # indicator_label = tk.Label(right_frame, width=10, height=2)  # 使用 tk.Label
    # indicator_label.pack(pady=10)

    show_label1 = ttk.Label(right_frame)
    show_label1.pack(side=tk.LEFT, pady=10, padx=10)

    show_label2 = ttk.Label(right_frame)
    show_label2.pack(side=tk.LEFT, pady=10, padx=10)

    # 启动更新UI的函数
    update_ui()

    # 运行主循环
    root.mainloop()

    # 释放资源
    cv2.destroyAllWindows()
    dev.release()