import cv2
import numpy as np
import time

# ==========================================
# 1. 設定參數
# ==========================================
BOARD_SIZE = (9, 6)    # 棋盤格內角點 (請確認您的棋盤格是 9x6 還是其他)
SQUARE_SIZE = 19       # 格子大小 (mm)，只影響 T 的單位，不影響誤差計算

# --- 請填入您之前「手動目測」的舊參數 (綠色網格法) ---
manual_k1 = -0.06 
manual_k2 = -0.2 
manual_p1 = 0.0    
manual_p2 = 0.0   

# ==========================================
# 2. 收集資料與校正流程
# ==========================================
def run_calibration_and_compare():
    # 準備 3D 物件點 (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1,2) * SQUARE_SIZE

    objpoints = [] # 3D 點
    imgpoints = [] # 2D 點

    cap = cv2.VideoCapture(2) # 開啟正面鏡頭 (如果是側面請改 1)
    if not cap.isOpened():
        print("無法開啟鏡頭")
        return

    print("--- 步驟 1: 收集校正資料 ---")
    print("請拿著棋盤格移動 (上下左右前後)")
    print("按 's' 拍照 (至少拍 5-10 張)")
    print("按 'c' 開始計算並進行比較")
    print("按 'q' 放棄")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

        # 畫圖
        disp = frame.copy()
        if ret_corners:
            cv2.drawChessboardCorners(disp, BOARD_SIZE, corners, ret_corners)
            status_text = f"Ready to Capture ({count})"
            color = (0, 255, 0)
        else:
            status_text = "No Chessboard"
            color = (0, 0, 255)
            
        cv2.putText(disp, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Calibration', disp)

        key = cv2.waitKey(1) & 0xFF
        
        # 按 S 拍照
        if key == ord('s') and ret_corners:
            objpoints.append(objp)
            imgpoints.append(corners)
            count += 1
            print(f"已擷取第 {count} 張")
            time.sleep(0.3) # 防止連按
            
        # 按 C 開始計算 (重點在這裡！)
        elif key == ord('c'):
            if count < 5:
                print("照片太少，請至少拍 5 張以上")
                continue
            
            print("\n正在進行棋盤格運算 (這就是您原本缺少的步驟)...")
            
            # --- 這裡產生 rvecs, tvecs, K0 ---
            ret, K0, D0, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            
            print("校正完成！")
            print(f"新算出的焦距 fx: {K0[0,0]:.2f}")
            
            # --- 接著直接執行比較邏輯 ---
            perform_comparison(objpoints, imgpoints, rvecs, tvecs, K0, D0)
            break
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# 3. 比較分析模組 (您原本要跑的那段)
# ==========================================
def perform_comparison(objpoints, imgpoints, rvecs, tvecs, K_new, D_new):
    print("\n" + "="*40)
    print("【實驗結果】新舊校正方法誤差比較")
    print("="*40)
    
    # 建立舊的手動參數矩陣
    D_manual = np.array([[manual_k1, manual_k2, manual_p1, manual_p2, 0]], dtype=np.float32)

    # 內部計算函式
    def calculate_error(K, D, name):
        total_error = 0
        total_points = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
            total_points += 1
        return total_error / total_points

    # 1. 計算新方法誤差
    err_new = calculate_error(K_new, D_new, "棋盤格校正 (新)")

    # 2. 計算舊方法誤差 (使用新的 K，配上舊的手動 D)
    err_old = calculate_error(K_new, D_manual, "綠色網格法 (舊)")

    # 3. 顯示結果
    print("-" * 40)
    if err_old > 0:
        improvement = ((err_old - err_new) / err_old) * 100
        print(f"★ 精度提升率: {improvement:.2f}%")
        print(f"★ 誤差降低: {err_old - err_new:.4f} pixels")
    print("="*40)
    
    # 暫停一下讓您看結果
    print("請按任意鍵結束程式...")
    cv2.waitKey(0)

if __name__ == "__main__":
    run_calibration_and_compare()