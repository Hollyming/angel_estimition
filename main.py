from openpose import pyopenpose as op
import cv2
import numpy as np
import time
import platform

def draw_from_numpy(img_ori, skel):
    img = img_ori.copy()
    pairs = [(1, 8), (1, 0), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6),
             (6, 7), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    body_parts = []
    for kk in range(len(skel)):
        k = skel[kk]
        body_part = (int(k[0]), int(k[1]))
        if body_part != (0, 0):
            # Use a specific color from the colors list
            # color = colors[kk % len(colors)]  # Use modulo to cycle through colors
            # cv2.circle(img, body_part, 4, color, 4)
            cv2.circle(img, body_part, 4, (0, 0, 255), 4)
        body_parts.append(body_part)
    # for pair in pairs:
    #     p1 = body_parts[pair[0]]
    #     p2 = body_parts[pair[1]]
    #     if p1 != (0, 0) and p2 != (0, 0):
    #         cv2.line(img, p1, p2, (0, 0, 255), 4)
    for idx, pair in enumerate(pairs):
        p1 = body_parts[pair[0]]
        p2 = body_parts[pair[1]]
        if p1 != (0, 0) and p2 != (0, 0):
            color = colors[idx % len(colors)]  # 选择一个循环颜色
            cv2.line(img, p1, p2, color, 4)
    return img

if __name__ == "__main__":
    # OpenPose parameters
    params = dict()
    params["model_folder"] = "openpose/models/"
    params["net_resolution"] = "480x-1"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Initialize RGB camera
    cam_idx = 0  # Assuming using the default camera
    if platform.system().lower() == 'windows':
        dev = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    else:
        dev = cv2.VideoCapture(cam_idx)
    
    # Set camera FPS (optional)
    dev.set(cv2.CAP_PROP_FPS, 30)

    while dev.isOpened():
        ret, frame = dev.read()  # Read frame from RGB camera
        if not ret:
            time.sleep(0.1)
            continue

        # Create a copy of the frame for visualization
        vis_output = frame.copy()

        # OpenPose processing
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        output = datum.cvOutputData

        # Draw skeleton on the original image
        if datum.poseKeypoints is not None:
            for i in range(datum.poseKeypoints.shape[0]):
                skel = datum.poseKeypoints[i, :15, :2].astype(np.int32)
                vis_output = draw_from_numpy(vis_output, skel)

        # Display the output with skeleton overlay
        cv2.imshow("Skeleton", vis_output)

        # Exit on 'q' or ESC
        k = cv2.waitKey(2) & 0xff
        if k == 27 or k == ord('q'):
            break

    cv2.destroyAllWindows()
    dev.release()
