import sys
import cv2
import os
import time
from sys import platform
import numpy as np
import math

dir_path = os.path.dirname(os.path.realpath(__file__))

K = np.array([[640, 0, 480], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
D = np.array([-0.06, -0.2, 0.0, 0.0], dtype=np.float32)

try:
    if platform == "win32":
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        sys.path.append('../../python');
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

def undistort_frame(frame):
    """
    輸入原始廣角畫面，輸出拉直後的畫面
    """
    if frame is None: return None
    # 使用 OpenCV 函數進行校正
    return cv2.undistort(frame, K, D)

# Cam 1 (front)
def analyze_shoulder_tilt(image, keypoints_list):
    """
    輸入: 影像, 關鍵點列表
    輸出: 畫好線條的影像
    """
    if keypoints_list is None or len(keypoints_list) == 0:
        return image
        
    # 取第一個人的關鍵點
    keypoints = keypoints_list[0]
    
    # 5: 左肩, 2: 右肩
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[2] 
    
    # 確保信心度 (Confidence Score) 足夠
    if left_shoulder[2] > 0.1 and right_shoulder[2] > 0.1:
        x1, y1 = int(left_shoulder[0]), int(left_shoulder[1])
        x2, y2 = int(right_shoulder[0]), int(right_shoulder[1])

        delta_y = abs(y1 - y2)
        delta_x = abs(x1 - x2)

        if delta_x == 0:
            slope_percent = 0
        else:
            slope_percent = (delta_y / delta_x) * 100

        if slope_percent < 2.0:
            color = (0, 255, 0)      # Green
            status = "Good"
        elif slope_percent < 5.0:
            color = (0, 255, 255)    # Yellow
            status = "Tilt"
        else:
            color = (0, 0, 255)      # Red
            status = "Bad"

        text = f"Slope: {slope_percent:.1f}% ({status})"
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2) - 20
        
        cv2.line(image, (x1, y1), (x2, y2), color, 2)

        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        # 避免文字跑出畫面，做個簡單邊界檢查
        if mid_y - h - 10 < 0: mid_y = h + 20

        cv2.rectangle(image, (mid_x - 10, mid_y - h - 10), (mid_x + w + 10, mid_y + 10), (0,0,0), -1)
        cv2.putText(image, text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return image

# Cam 2 (side)
def analyze_spine_lean(image, keypoints_list):
    """
    修改版：針對側面優化。
    不抓不穩定的中線 (Neck-MidHip)，改抓「同側肩膀」連「同側臀部」。
    """
    if keypoints_list is None or len(keypoints_list) == 0:
        cv2.putText(image, "No Person", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return image

    keypoints = keypoints_list[0]
    
    # 定義關鍵點索引 (BODY_25)
    # 2:右肩, 9:右臀 | 5:左肩, 12:左臀
    r_shoulder, r_hip = keypoints[2], keypoints[9]
    l_shoulder, l_hip = keypoints[5], keypoints[12]

    # --- 判斷哪一側可見度高 ---
    # 我們檢查哪一組 (肩+臀) 的信心分數總和比較高
    score_right = r_shoulder[2] + r_hip[2]
    score_left = l_shoulder[2] + l_hip[2]
    
    # 設定最低門檻 (如果連一側都沒抓清楚，就放棄)
    threshold = 0.2

    target_shoulder = None
    target_hip = None
    side_name = ""

    if score_right > score_left and score_right > threshold:
        target_shoulder = r_shoulder
        target_hip = r_hip
        side_name = "Right Side"
    elif score_left > score_right and score_left > threshold:
        target_shoulder = l_shoulder
        target_hip = l_hip
        side_name = "Left Side"
    
    # --- 開始計算 ---
    if target_shoulder is not None and target_hip is not None:
        sx, sy = int(target_shoulder[0]), int(target_shoulder[1])
        hx, hy = int(target_hip[0]), int(target_hip[1])

        # 計算向量 (從臀部指向肩膀)
        dx = sx - hx
        dy = hy - sy # 讓 Y 軸向上為正

        # 計算角度 (相對於垂直線)
        angle = 0
        if dy != 0:
            angle = math.degrees(math.atan(abs(dx) / dy))

        # 判斷標準
        if angle < 10:
            status, color = "Good", (0, 255, 0)
        elif angle < 20:
            status, color = "Leaning", (0, 255, 255)
        else:
            status, color = "Hunchback", (0, 0, 255)

        # --- 繪圖 ---
        # 1. 畫出身體連線 (肩到臀)
        cv2.line(image, (hx, hy), (sx, sy), color, 4)
        
        # 2. 畫出垂直參考線
        cv2.line(image, (hx, hy), (hx, hy - 150), (200, 200, 200), 2)

        # 3. 關鍵點
        cv2.circle(image, (sx, sy), 6, (255, 0, 0), -1)
        cv2.circle(image, (hx, hy), 6, (0, 0, 255), -1)

        # 4. 文字
        cv2.putText(image, f"Angle: {int(angle)} ({side_name})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, status, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    else:
        # 如果真的抓不到，提示使用者轉一下角度
        cv2.putText(image, "Rotate Body Slightly", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, "(Side Not Clear)", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return image

def main():
    params = dict()
    params["model_folder"] = "../../../models/"
    params["net_resolution"] = "-1x160"
    params["model_pose"] = "BODY_25"
    params["number_people_max"] = 1

    try:
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
    except Exception as e:
        print(f"OpenPose 初始化失敗: {e}")
        return

    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    cap1.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap2.set(cv2.CAP_PROP_FOURCC, fourcc)

    width, height = 640, 480

    cap1.set(3, width)
    cap1.set(4, height)
    cap2.set(3, width)
    cap2.set(4, height)

    if not cap1.isOpened():
        print("無法開啟鏡頭 0")
    if not cap2.isOpened():
        print("無法開啟鏡頭 1")

    print("開始執行... 按 'q' 離開")

    prev_time = time.time()
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 and not ret2:
            print("兩支鏡頭都無法讀取")
            break

        if not ret1: frame1 = np.zeros((height, width, 3), dtype=np.uint8)
        if not ret2: frame2 = np.zeros((height, width, 3), dtype=np.uint8)

        frame2 = undistort_frame(frame2)

        # CAM 1
        datum1 = op.Datum()
        datum1.cvInputData = frame1
        opWrapper.emplaceAndPop(op.VectorDatum([datum1]))
        output1 = analyze_shoulder_tilt(datum1.cvOutputData, datum1.poseKeypoints)

        # CAM 2
        datum2 = op.Datum()
        datum2.cvInputData = frame2
        opWrapper.emplaceAndPop(op.VectorDatum([datum2]))
        output2 = analyze_spine_lean(datum2.cvOutputData, datum2.poseKeypoints)

        cv2.putText(output1, "Cam 1: Front (Shoulder)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output2, "Cam 2: Side (Hunchback)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        display_h = 540
        scale = display_h / height 
        display_w = int(width * scale)

        img1_small = cv2.resize(output1, (display_w, display_h))
        img2_small = cv2.resize(output2, (display_w, display_h))

        # 合併
        combined_image = np.hstack((img1_small, img2_small))

        cv2.imshow("Posture Analysis", combined_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
