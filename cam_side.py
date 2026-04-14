import cv2
import numpy as np
import time
import math

class CamSide:
    def __init__(self):
        self.id = 2
        self.cap = None
        
        # 內參矩陣
        self.K = np.array([[752.75825, 0.0, 638.23156], [0.0, 751.44076, 360.50859], [0.0, 0.0, 1.0]], dtype=np.float32)
        self.D = np.array([-0.33029, 0.13842, 0.00041, -0.03133], dtype=np.float32)

        # --- 正面相機內參 (三角測量需要知道對手的參數) ---
        # 為了獨立性，這裡手動複製一份 Front 的參數，或者你可以從 config 匯入
        self.K_front = np.array([[916.86916, 0.0, 632.03174], [0.0, 917.45823, 392.08485], [0.0, 0.0, 1.0]], dtype=np.float32)
        self.D_front = np.array([0.05881, -0.09019, 0.00016, 0.00172, 0.0209], dtype=np.float32)

        self.pos = (57.2427, 22.8355, 4.0658)
        self.angles = (-10.0, -45.0)
        self.R, self.T = self.compute_extrinsics()

        self.angle_offset = 0.0 
        self.current_raw_angle = 0.0

    def open(self):
        print(f">>> 開啟 Side Cam (ID: {self.id})...")
        self.cap = cv2.VideoCapture(self.id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(3, 640); self.cap.set(4, 480)
        time.sleep(1.0)
        return self.cap.isOpened()

    def close(self):
        if self.cap: self.cap.release()

    def set_calibration(self):
        """ 當使用者坐正時呼叫此函式，將當下角度設為 0 度基準 """
        self.angle_offset = self.current_raw_angle
        print(f">>> [Side] 側面校準完成！Offset 設定為: {self.angle_offset:.2f} 度")

    def compute_extrinsics(self):
        """ 根據角度與位置計算 R, T 矩陣 """
        px, py, pz = self.pos
        pitch, yaw = self.angles
        
        rad_pitch, rad_yaw = math.radians(pitch), math.radians(yaw)
        Rx = np.array([[1, 0, 0], [0, math.cos(rad_pitch), -math.sin(rad_pitch)], [0, math.sin(rad_pitch), math.cos(rad_pitch)]])
        Ry = np.array([[math.cos(rad_yaw), 0, math.sin(rad_yaw)], [0, 1, 0], [-math.sin(rad_yaw), 0, math.cos(rad_yaw)]])
        
        # R_pose = Ry @ Rx (先 Pitch 再 Yaw)
        R_stereo = (Ry @ Rx).T
        C_vec = np.array([[px], [py], [pz]], dtype=np.float32)
        T_stereo = -R_stereo @ C_vec
        return R_stereo, T_stereo

    def triangulate_point(self, pt_front, pt_side):
        """ 將 2D 點轉換為 3D 點 """
        if pt_front is None or pt_side is None: return None
        if pt_front[0] == 0 or pt_side[0] == 0: return None

        pt1_np = np.array([[[float(pt_front[0]), float(pt_front[1])]]], dtype=np.float32)
        pt2_np = np.array([[[float(pt_side[0]), float(pt_side[1])]]], dtype=np.float32)

        pt1_norm = cv2.undistortPoints(pt1_np, self.K_front, self.D_front)
        pt2_norm = cv2.undistortPoints(pt2_np, self.K, self.D)

        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((self.R, self.T))
        
        pt_4d = cv2.triangulatePoints(P1, P2, pt1_norm, pt2_norm)
        return (pt_4d[:3] / pt_4d[3]).flatten()

    def process_frame(self, pose_estimator, kps_front=None):
        """
        採用純 2D 模式，不依賴正面鏡頭，穩定性最高
        """
        if not self.cap: return None
        
        ret, raw_frame = self.cap.read()
        if not ret: return None

        # 1. 去畸變 (很重要，讓直線變直)
        img = cv2.undistort(raw_frame, self.K, self.D)

        # 2. 骨架偵測
        kps_side, drawn_img = pose_estimator.detect(img)
        if drawn_img is None: drawn_img = img

        final_angle = 0.0
        
        # 3. 計算邏輯
        if kps_side is not None and len(kps_side) > 0:
            p_side = kps_side[0]
            
            # 取得右耳(17)與右肩(2)的 2D 像素座標
            # 如果是拍左側，請改用 左耳(18) 與 左肩(5)
            s_ear = p_side[17][:2] 
            s_sh = p_side[2][:2]
            s_rw = p_side[4][:2] #NEW 4左手腕
            s_lw = p_side[7][:2] #NEW 7右手腕
            s_nose = p_side[0][:2] #NEW 0鼻子

            # 只要有點就能算，不用管正面鏡頭
            if s_ear[0] > 0 and s_sh[0] > 0:
                
                # dx: 水平距離 (前後)
                # dy: 垂直距離 (高度) - 注意 Y 軸向下為正
                pixel_dx = s_ear[0] - s_sh[0] 
                pixel_dy = s_sh[1] - s_ear[1] 

                if pixel_dy != 0:
                    # --- 步驟 A: 視角幾何補償 ---
                    # 因為側面鏡頭大概是 45 度角拍攝，前後距離(dx)會被壓縮
                    # 我們除以 sin(45) 來嘗試還原一點真實感
                    # 雖然這不是完美的 3D，但比直接算準確
                    correction_factor = 0.707 # sin(45度)
                    
                    real_dx_est = pixel_dx / correction_factor
                    real_dy_est = pixel_dy

                    # --- 步驟 B: 計算原始角度 ---
                    # 這裡算出來可能就是您看到的 22 度
                    raw_angle = math.degrees(math.atan2(real_dx_est, real_dy_est))
                    
                    # 存起來給 set_calibration 用
                    self.current_raw_angle = raw_angle

                    # --- 步驟 C: 套用校準 (歸零) ---
                    # 最終角度 = 原始角度 - 偏差值
                    # 範例： 22 - 22 = 0
                    final_angle = raw_angle - self.angle_offset

                    # --- 顯示 ---
                    status = "Good"
                    color = (0, 255, 0)
                    
                    # 判定邏輯：超過 10 度就紅字
                    if final_angle > 10: 
                        status = "Neck Fwd"
                        color = (0, 0, 255)

                    # 伸展 NEW
                    stretching = False
                    stretching_color = (0,255,0)
                    if (s_rw[1] > 0 and s_rw[1] < s_nose[1]) or (s_lw[1] > 0 and s_lw[1] < s_nose[1]):
                        stretching = True
                        stretching_color = (0,0,255)

                    # 繪製資訊背景板
                    cv2.rectangle(drawn_img, (0, 0), (360, 110), (0,0,0), -1)
                    
                    # 顯示最終角度 (大字)
                    cv2.putText(drawn_img, f"Angle: {final_angle:.1f}d", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    
                    # 顯示原始數據 (小字，讓您知道 Offset 有沒有在運作)
                    # Raw: 22.0, Off: 22.0 -> 結果就會是 0
                    info_txt = f"Raw:{raw_angle:.1f} | Off:{self.angle_offset:.1f}"
                    cv2.putText(drawn_img, info_txt, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    # NEW
                    cv2.putText(drawn_img,f"Stretching: {stretching}",(10,80),cv2.FONT_HERSHEY_SIMPLEX, 0.8, stretching_color, 2)

                    # 畫線
                    cv2.line(drawn_img, (int(s_sh[0]), int(s_sh[1])), (int(s_ear[0]), int(s_ear[1])), color, 3)

        return drawn_img