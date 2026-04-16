import cv2
import numpy as np
import math
import time

class CamFront:
    def __init__(self):
        self.id = 0
        self.cap = None
        
        # 內參矩陣
        self.K = np.array([[916.86916, 0.0, 632.03174], [0.0, 917.45823, 392.08485], [0.0, 0.0, 1.0]], dtype=np.float32)
        self.D = np.array([0.05881, -0.09019, 0.00016, 0.00172, 0.0209], dtype=np.float32)
        
        # 狀態變數
        self.standard_shoulder_y = None # 校準線
        self.hunch_thresh = 20

        # NEW 轉身狀態
        self.turning_dx = None

        # NEW 起坐
        self.standard_neck_y = None
        self.is_standing = False
        self.standing_color = (0,255,0)

        # NEW 重置用參數
        self.current_shoulder_y = None
        self.current_neck_y = None
        self.current_dx = None

    def open(self):
        print(f">>> 開啟 Front Cam (ID: {self.id})...")
        self.cap = cv2.VideoCapture(self.id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(3, 640); self.cap.set(4, 480)
        time.sleep(1.0)
        return self.cap.isOpened()

    def close(self):
        if self.cap: self.cap.release()
    
    def set_calibration(self):
        """ 由主程式呼叫 設定校準值 """
        if self.current_shoulder_y is not None:
            self.standard_shoulder_y = self.current_shoulder_y
            print(f"[Front] 校準完成，肩膀標準線 Y = {self.standard_shoulder_y:.1f}")
        if self.current_neck_y is not None:
            self.standard_neck_y = self.current_neck_y
            print(f"[Front] 校準完成，頸部標準線 Y = {self.standard_neck_y:.1f}")
        if self.current_dx is not None:
            self.turning_dx = self.current_dx
            print(f"[Front] 校準完成，轉身基準 dx = {self.turning_dx:.1f}")

    def process_frame(self, pose_estimator):
        """ 讀取 -> 去畸變 -> 偵測 -> 計算邏輯 -> 繪圖 """
        if not self.cap: return None, None
        
        ret, raw_frame = self.cap.read()
        if not ret: return None, None

        # 去畸變
        img = cv2.undistort(raw_frame, self.K, self.D)

        # 架偵測
        kps, drawn_img = pose_estimator.detect(img)
        if drawn_img is None: drawn_img = img

        if kps is not None and len(kps) > 0:
            p1 = kps[0]
            f_neck = p1[1][:2] # NEW 1頸部
            f_rsh, f_lsh = p1[2][:2], p1[5][:2] # 2右肩 5左肩

            # 輸出用資訊
            angle = 0.0
            tilt_status = "N/A"
            tilt_color = (255, 255, 255)
            hunch_txt = "Press 'c'"
            hunch_color = (255, 255, 0)
            if self.standard_shoulder_y is not None:
                hunch_txt = "N/A"
                hunch_color = (255, 255, 255)
            turning = False
            turning_color = (0,255,0)

            if f_rsh[0] > 0 and f_lsh[0] > 0:
                # 傾斜
                dy = f_lsh[1] - f_rsh[1]
                dx = f_lsh[0] - f_rsh[0]
                if dx == 0: dx = 0.0001
                angle = abs(math.degrees(math.atan2(dy, dx)))
                if angle > 90: angle = 180 - angle
                
                tilt_status = "Good"
                tilt_color = (0, 255, 0)
                if angle > 5:
                    tilt_status = "BAD"
                    tilt_color = (0, 0, 255)

                # 駝背
                avg_y = (f_lsh[1] + f_rsh[1]) / 2.0
                
                # NEW 轉身
                if self.turning_dx is not None:
                    if dx/self.turning_dx <0.85:
                        turning = True
                        turning_color = (0,0,255)
                    elif dx/self.turning_dx > 0.95:
                        turning = False
                        turning_color = (0,255,0)

                # 最低點
                lowest_y = max(f_lsh[1], f_rsh[1])
                lowest_pt = (int(f_lsh[0]), int(f_lsh[1])) if f_lsh[1] > f_rsh[1] else (int(f_rsh[0]), int(f_rsh[1]))

                if self.standard_shoulder_y is not None:
                    diff = lowest_y - self.standard_shoulder_y
                    
                    # 畫標準線與距離線
                    line_y = int(self.standard_shoulder_y)
                    cv2.line(drawn_img, (0, line_y), (640, line_y), (255, 0, 0), 1)
                    cv2.line(drawn_img, lowest_pt, (lowest_pt[0], line_y), (255, 0, 255), 3)
                    cv2.circle(drawn_img, lowest_pt, 6, (255, 0, 255), -1)

                    if diff > self.hunch_thresh:
                        hunch_txt = f"Drop: {diff:.0f} (BAD)"
                        hunch_color = (0, 0, 255)
                    elif diff < -20:
                        hunch_txt = f"High: {abs(diff):.0f}"
                        hunch_color = (255, 255, 0)
                    else:
                        hunch_txt = "Height: OK"
                        hunch_color = (0, 255, 0)
                
                # NEW 重置用參數更新
                self.current_shoulder_y = avg_y
                self.current_dx = dx

            if f_neck[0] > 0:
                # NEW 起坐
                neck_current_val = f_neck[1]
                if (neck_current_val > 0 and self.standard_neck_y is not None):
                    diff_neck_y = self.standard_neck_y - neck_current_val
                    if diff_neck_y > 80:
                        self.is_standing = True
                        self.standing_color = (0,0,255)
                    elif diff_neck_y < 40:
                        self.is_standing = False
                        self.standing_color = (0,255,0)


                # NEW 重置用參數更新
                self.current_neck_y = f_neck[1]

            cv2.rectangle(drawn_img, (0, 0), (240, 170), (10,10,10), -1)
            cv2.putText(drawn_img, f"Tilt: {angle:.1f}d ({tilt_status})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tilt_color, 2)
            cv2.putText(drawn_img, hunch_txt, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hunch_color, 2)
            # NEW
            cv2.putText(drawn_img,f"Turning: {turning}",(10,110),cv2.FONT_HERSHEY_SIMPLEX, 0.8, turning_color, 2)
            cv2.putText(drawn_img, f"Standing: {self.is_standing}",(10,150),cv2.FONT_HERSHEY_SIMPLEX,0.8,self.standing_color,2)

        return drawn_img, kps