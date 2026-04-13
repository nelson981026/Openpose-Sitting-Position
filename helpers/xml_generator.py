import cv2
import numpy as np
import os

# ==========================================
# 1. 參數設定 (請填入您的數值)
# ==========================================

# --- 輸出資料夾 ---
OUTPUT_DIR = "camera_parameters"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- A. 內參 (Intrinsics) ---
# 請填入您用棋盤格校正算出來的 K 與 D
# 這裡用範例值，請務必換成您實際跑出來的數據！
K_front = np.array([
    [916.86916, 0.0, 632.03174],
    [0.0, 917.45823, 392.08485],
    [0.0, 0.0, 1.0]])
D_front = np.array([0.05881, -0.09019, 0.00016, 0.00172, 0.0209])

K_side  = np.array([
    [752.75825, 0.0, 638.23156],
    [0.0, 751.44076, 360.50859],
    [0.0, 0.0, 1.0]])
D_side  = np.array([-0.33029, 0.13842, 0.00041, -0.03133])

# --- B. 側面鏡頭的物理位置與角度 (您的定義) ---
# 位置 (公尺): 前(X) 0.028, 左(Y) 0.575, 上(Z) 0.220
pos_user = np.array([0.028018, 0.575436, 0.220303]) 

# 角度: 向下 20 度, 向右 45 度
pitch_deg = 20
yaw_deg = 45 # 注意：等等計算時會轉負號

# ==========================================
# 2. 計算側面鏡頭的外參 (Extrinsics)
# ==========================================

def calculate_extrinsics():
    # 1. 位置轉換: User(X,Y,Z) -> OpenCV(-Y,-Z,X)
    # 這是因為 OpenCV 的座標系定義與您的不同
    pos_opencv = np.array([
        -pos_user[1], # X_cv = -Y_user (向右為正)
        -pos_user[2], # Y_cv = -Z_user (向下為正)
         pos_user[0]  # Z_cv = X_user  (向前為正)
    ]).reshape(3, 1)

    # 2. 旋轉計算
    # 轉換角度為弧度 (向右轉為負，向下轉為正)
    ang_yaw = np.radians(-yaw_deg) 
    ang_pitch = np.radians(pitch_deg)

    # 繞 Y 軸 (Yaw)
    Ry = np.array([
        [np.cos(ang_yaw), 0, np.sin(ang_yaw)],
        [0, 1, 0],
        [-np.sin(ang_yaw), 0, np.cos(ang_yaw)]
    ])
    
    # 繞 X 軸 (Pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(ang_pitch), -np.sin(ang_pitch)],
        [0, np.sin(ang_pitch), np.cos(ang_pitch)]
    ])

    # 鏡頭姿態矩陣 R_cam
    R_cam = Rx @ Ry
    
    # 計算外參矩陣 R (World to Camera)
    R = R_cam.T
    
    # 計算外參向量 T (World to Camera)
    # T = -R * C
    T = -R @ pos_opencv
    
    return R, T

# 計算側面的 R 和 T
R_side, T_side = calculate_extrinsics()

# 正面的 R 和 T (設為原點)
R_front = np.eye(3)
T_front = np.zeros((3, 1))

# ==========================================
# 3. 寫入 XML 檔案 (OpenPose 格式)
# ==========================================

def write_xml(filename, K, D, R, T):
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # 使用 cv2.FileStorage 寫入
    cv_file = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
    
    cv_file.write("CameraMatrix", K)
    cv_file.write("Intrinsics", K) # 有些舊版 OpenPose 會讀這個標籤，保險起見兩個都寫
    cv_file.write("Distortion", D)
    cv_file.write("DistortionCoeffs", D)
    cv_file.write("RotationMatrix", R)
    cv_file.write("TranslationVector", T)
    
    cv_file.release()
    print(f"已生成: {filepath}")

# 產生檔案
write_xml("0.xml", K_front, D_front, R_front, T_front)
write_xml("1.xml", K_side,  D_side,  R_side,  T_side)

print("\n完成！請將 'camera_parameters' 資料夾的路徑餵給 OpenPose。")