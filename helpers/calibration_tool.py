import cv2
import numpy as np
import time

# --- 設定區 ---
CAMERA_ID = 2          # 請修改這裡：0 是正面鏡頭，1 是側面鏡頭 (一次校正一顆)
BOARD_SIZE = (9, 6)    # 棋盤格的「內角點」數量 (長, 寬)
SQUARE_SIZE = 19       # 每一格的邊長 (單位: mm)，這影響最後的平移距離單位，不影響畸變修正
# -------------

def run_calibration():
    # 準備棋盤格的 3D 座標 (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    # 儲存點的陣列
    objpoints = [] # 3D 點 (世界座標)
    imgpoints = [] # 2D 點 (影像座標)

    cap = cv2.VideoCapture(CAMERA_ID)
    # 設定解析度 (必須跟您主程式一樣)
    cap.set(3, 640)
    cap.set(4, 480)

    print(f"--- 開始校正鏡頭 ID: {CAMERA_ID} ---")
    print("請拿著棋盤格在鏡頭前移動 (上下左右、遠近、傾斜)")
    print("按 's' 拍攝一張有效照片 (需要約 15-20 張)")
    print("按 'q' 結束並開始計算")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 尋找棋盤格角點
        ret_corners, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

        display_frame = frame.copy()

        # 畫出角點
        if ret_corners:
            cv2.drawChessboardCorners(display_frame, BOARD_SIZE, corners, ret_corners)
            cv2.putText(display_frame, "Ready to Capture", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Show Chessboard", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display_frame, f"Captured: {count}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Calibration', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and ret_corners:
            objpoints.append(objp)
            imgpoints.append(corners)
            count += 1
            print(f"已拍攝第 {count} 張")
            time.sleep(0.5) # 暫停一下避免連拍
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count < 10:
        print("拍攝張數不足，建議至少 10 張以上以獲得準確結果。")
        return

    # 計算校正結果
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\n" + "="*40)
    print(f"校正結果 (鏡頭 {CAMERA_ID})")
    print("="*40)
    print("請將以下數據複製回您的程式碼 (cam_side.py 或 cam_front.py)：\n")
    
    # 格式化輸出，方便您直接複製
    print("self.K = np.array([")
    print(f"    [{mtx[0][0]:.5f}, {mtx[0][1]:.5f}, {mtx[0][2]:.5f}],")
    print(f"    [{mtx[1][0]:.5f}, {mtx[1][1]:.5f}, {mtx[1][2]:.5f}],")
    print(f"    [{mtx[2][0]:.5f}, {mtx[2][1]:.5f}, {mtx[2][2]:.5f}]")
    print("], dtype=np.float32)")
    
    print("\nself.D = np.array([")
    print(f"    {dist[0][0]:.5f}, {dist[0][1]:.5f}, {dist[0][2]:.5f}, {dist[0][3]:.5f}, {dist[0][4]:.5f}")
    print("], dtype=np.float32)")
    print("="*40)

    # 計算誤差值 (越小越好，通常小於 1.0 是可接受，小於 0.1 是完美)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print(f"平均誤差 (Reprojection Error): {mean_error/len(objpoints):.5f}")

if __name__ == "__main__":
    run_calibration()