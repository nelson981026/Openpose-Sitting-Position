import sys
import os
import cv2
import numpy as np
import time

# ==========================================
# 1. 產生 OpenPose 喜歡的完美 XML
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_FOLDER = os.path.join(BASE_DIR, "xml_final_fix")

if not os.path.exists(XML_FOLDER):
    os.makedirs(XML_FOLDER)

# 640x480 的標準參數
K = np.array([[458.0, 0, 320.0], [0, 458.0, 240.0], [0, 0, 1]])
D = np.zeros(5) 
R0 = np.eye(3); T0 = np.zeros((3,1))

theta = np.radians(-90)
R1 = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
T1 = np.array([[500.0], [0.0], [0.0]])

def save_xml_robust(name, K, D, R, T):
    path = os.path.join(XML_FOLDER, name)
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    
    # 寫入影像尺寸 (非常重要！有些版本沒這個會報錯)
    fs.write("image_width", 640)
    fs.write("image_height", 480)
    
    # 寫入多種別名，確保讀取無死角
    fs.write("CameraMatrix", K); fs.write("Intrinsics", K)
    fs.write("Distortion", D);   fs.write("DistortionCoeffs", D)
    fs.write("Rotation", R);     fs.write("RotationMatrix", R)
    fs.write("Translation", T);  fs.write("TranslationVector", T)
    fs.release()

save_xml_robust("0.xml", K, D, R0, T0)
save_xml_robust("1.xml", K, D, R1, T1)

print(f"✅ XML 已重建於: {XML_FOLDER}")
time.sleep(1.0) # 等硬碟寫入

# ==========================================
# 2. 初始化 OpenPose (關鍵參數修正)
# ==========================================
try:
    sys.path.append('../../python')
    from openpose import pyopenpose as op
except:
    print("找不到 OpenPose Library"); sys.exit()

params = dict()
params["model_folder"] = "../../../models/"
params["model_pose"] = "BODY_25"
params["number_people_max"] = 1
params["3d"] = True
params["3d_min_views"] = 2

# 【關鍵修正 1】路徑必須以斜線結尾
params["camera_parameter_path"] = XML_FOLDER + os.sep 

# 【關鍵修正 2】絕對必須開啟，否則它會忽略 1.xml 的 R/T 矩陣！
params["frame_undistort"] = True

print(f"啟動參數: 3d=True, undistort=True")

try:
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    print("OpenPose 啟動成功！ (沒崩潰就是好事)")
except Exception as e:
    print(f"啟動失敗: {e}")
    sys.exit()

# ==========================================
# 3. 執行迴圈 (防崩潰版)
# ==========================================
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(2) # 試試 2 或 1

# MJPG 設定
cap0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap0.set(3, 640); cap0.set(4, 480)
cap1.set(3, 640); cap1.set(4, 480)

if not cap0.isOpened() or not cap1.isOpened():
    print("鏡頭開啟失敗")
else:
    print("鏡頭開啟成功，開始執行 (按 q 離開)...")
    
    while True:
        ret0, img0 = cap0.read()
        ret1, img1 = cap1.read()
        
        if not ret0 or not ret1:
            print("遺失訊號...", end="\r")
            continue
        
        d0 = op.Datum(); d0.cvInputData = img0
        d1 = op.Datum(); d1.cvInputData = img1
        
        try:
            # 這裡如果 OpenPose 參數沒設好，C++ 會崩潰
            opWrapper.emplaceAndPop(op.VectorDatum([d0, d1]))
        except Exception as e:
            print(f"\nOpenPose 執行錯誤: {e}")
            break
        
        # 取得畫面 (如果 frame_undistort=True，這裡的圖會是去畸變後的)
        out0 = d0.cvOutputData
        
        # 防止 imshow 空值錯誤
        if out0 is None:
            # 如果 OpenPose 沒吐出圖，就用原圖顯示，避免 python 報錯
            out0 = img0 
            cv2.putText(out0, "OpenPose Failed", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            # 檢查 3D
            kp3d = d0.poseKeypoints3D
            if kp3d is not None and len(kp3d) > 0:
                neck = kp3d[0][1]
                info = f"Neck Z: {int(neck[2])} mm"
                cv2.putText(out0, info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"\rSUCCESS! {info}     ", end="")
            else:
                cv2.putText(out0, "Searching 3D...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Result", out0)
        if cv2.waitKey(1) == ord('q'): break

cap0.release(); cap1.release(); cv2.destroyAllWindows()