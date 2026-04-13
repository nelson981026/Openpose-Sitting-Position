import sys
import os
import cv2
import numpy as np
from sys import platform

class Helper:
    def __init__(self):
        print(">>> [Pose] 初始化 OpenPose...")
        self.is_initialized = False
        self.op = None          # 儲存 pyopenpose 模組
        self.opWrapper = None   # 儲存 OpenPose Wrapper

        # ---------------------------------------------------------
        # 1. 設定系統路徑並匯入 pyopenpose (原 init_openpose 邏輯)
        # ---------------------------------------------------------
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            if platform == "win32":
                # 設定 Windows 下 Release 資料夾路徑
                sys.path.append(dir_path + '/../../python/openpose/Release')
                os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                # 設定 Linux/Mac 路徑
                sys.path.append('../../python')
                from openpose import pyopenpose as op
            
            # 將匯入的模組存入 self，供其他方法使用
            self.op = op

        except ImportError as e:
            print('Error: OpenPose library could not be found.')
            print(e)
            return

        # ---------------------------------------------------------
        # 2. 設定參數 (原 params 設定)
        # ---------------------------------------------------------
        params = dict()
        params["model_folder"] = "../../../models/"
        params["net_resolution"] = "-1x160"
        params["model_pose"] = "BODY_25"
        params["number_people_max"] = 1

        # ---------------------------------------------------------
        # 3. 啟動 Wrapper
        # ---------------------------------------------------------
        try:
            self.opWrapper = self.op.WrapperPython()
            self.opWrapper.configure(params)
            self.opWrapper.start()
            self.is_initialized = True
            print(">>> [Pose] OpenPose 啟動成功 (Ready).")
        except Exception as e:
            print(f"Error: OpenPose Wrapper failed to start: {e}")
            self.is_initialized = False

    def detect(self, frame):
        """
        輸入影像，回傳 (keypoints, rendered_image)
        對應原本 helper.get_keypoints 的功能
        """
        if not self.is_initialized or frame is None:
            return None, None

        try:
            # 建立 Datum 物件
            datum = self.op.Datum()
            datum.cvInputData = frame
            
            # 執行偵測
            self.opWrapper.emplaceAndPop(self.op.VectorDatum([datum]))
            
            # 回傳 (骨架座標, 繪製後的圖)
            return datum.poseKeypoints, datum.cvOutputData
            
        except Exception as e:
            print(f"Detection Error: {e}")
            return None, frame