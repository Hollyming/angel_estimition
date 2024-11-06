from openpose import pyopenpose as op
import cv2
import numpy as np
import time
import math
import platform

# 计算角度函数
def angle_between_points(p0, p1, p2):
    a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
    b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
    if a * b == 0:
        return -1.0
    return math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi

# 获取特定关节的三个关键点
def get_angle_point(human, pos_list):
    pnts = []
    for i in pos_list:
        if human[i][0] == 0 and human[i][1] == 0:  # 如果坐标为0, 表示关键点丢失
            return []
        pnts.append((int(human[i][0]), int(human[i][1])))
    return pnts

# 绘制骨架和角度
def draw_from_numpy(img_ori, skel):
    img = img_ori.copy()
    pairs = [(1, 8), (1, 0), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6),
             (6, 7), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    body_parts = []

    # 绘制关节点
    for kk in range(len(skel)):
        k = skel[kk]
        body_part = (int(k[0]), int(k[1]))
        if body_part != (0, 0):
            cv2.circle(img, body_part, 4, (0, 0, 255), 4)
        body_parts.append(body_part)

    # 绘制骨骼连线
    for idx, pair in enumerate(pairs):
        p1 = body_parts[pair[0]]
        p2 = body_parts[pair[1]]
        if p1 != (0, 0) and p2 != (0, 0):
            color = colors[idx % len(colors)]
            cv2.line(img, p1, p2, color, 4)

    # 计算并显示角度
    joint_angles = {
        'neck': [1, 0, 8],
        'left_shoulder': [1, 5, 6],
        'right_shoulder': [1, 2, 3],
        'left_elbow': [5, 6, 7],
        'right_elbow': [2, 3, 4],
        'left_wrist': [6, 7, 4],
        'right_wrist': [3, 4, 2],
        'left_hip': [8, 12, 13],
        'right_hip': [8, 9, 10],
        'left_knee': [12, 13, 14],
        'right_knee': [9, 10, 11],
        'left_ankle': [13, 14, 12],
        'right_ankle': [10, 11, 9],
    }


    for joint_name, pos_list in joint_angles.items():
        pnts = get_angle_point(skel, pos_list)
        if len(pnts) == 3:
            angle = angle_between_points(pnts[0], pnts[1], pnts[2])
            if angle != -1:
                cv2.putText(img, f"{joint_name} angle: {int(angle)}", pnts[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img

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
    

    while dev.isOpened():
        ret, frame = dev.read()  # 从摄像头读取帧
        if not ret:
            time.sleep(0.1)
            continue

        vis_output = frame.copy()

        # OpenPose处理
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # 绘制骨架和角度
        if datum.poseKeypoints is not None:
            for i in range(datum.poseKeypoints.shape[0]):
                skel = datum.poseKeypoints[i, :15, :2].astype(np.int32)
                vis_output = draw_from_numpy(vis_output, skel)

        # 显示带有骨架的输出
        cv2.imshow("Skeleton", vis_output)

        # 按'q'或ESC键退出
        k = cv2.waitKey(2) & 0xff
        if k == 27 or k == ord('q'):
            break

    cv2.destroyAllWindows()
    dev.release()
