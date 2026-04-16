import cv2
import numpy as np

# 匯入我們的三個獨立類別
from helper import Helper
from cam_front import CamFront
from cam_side import CamSide

def main():
    # 1. 初始化各個組件
    pose = Helper()
    front = CamFront()
    side = CamSide()

    # 2. 開啟相機
    if not front.open() or not side.open():
        print("無法開啟相機，程式結束。")
        return

    print(">>> 系統啟動！按 'c' 校準，按 'q' 離開。")

    while True:
        ret1, frame1 = front.cap.read()
        ret2, frame2 = side.cap.read()
        if not ret1 or not ret2: break

        # vis_f = cv2.undistort(frame1, front.K, front.D)
        # kps_f, _ = pose.detect(vis_f)

        img_f, kps_f = front.process_frame(frame1, pose) # 假設我們加了這個方法，或修改原方法
        # 3. 讓每個相機各自處理自己的畫面
        #    注意：我們把 pose 傳進去，讓相機自己去呼叫 detect
        img_s = side.process_frame(frame2, pose, kps_front=kps_f)

        if img_f is None or img_s is None:
            print("讀取影像錯誤")
            break

        # 4. 畫面拼接 (Resize side to match front height if needed)
        if img_f.shape != img_s.shape:
            img_s = cv2.resize(img_s, (img_f.shape[1], img_f.shape[0]))
        
        combined = np.hstack((img_f, img_s))
        cv2.imshow("Dual Camera System", combined)

        # 5. 按鍵處理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # 若 Front Cam 有回傳可用的校準值，就設定回去
            front.set_calibration()
            
            # 2. 校準頸部角度 (Side) - 把現在姿勢歸零
            side.set_calibration()

    # 6. 清理
    front.close()
    side.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()