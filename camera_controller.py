#!/usr/bin/env python3

import os
import sys
from picamera2 import Picamera2, Preview
from picamera2.devices import IMX500
from libcamera import Transform

# class RPICameraController:
#     def __init__(self, camera_num=0, model_dir="models"):
#         self.model_dir = model_dir
#         self.camera_num = camera_num
#         self.camera = None
#         self.imx500 = None

#         self._initialize_camera()

#     def _initialize_camera(self):
#         """Initialize the Picamera2 with a preview window."""
#         print("[Camera] Initializing...")
#         self.camera = Picamera2(self.camera_num)
#         self.camera.start_preview(
#             Preview.QTGL,
#             x=100, y=200, width=800, height=600,
#             transform=Transform(vflip=1)
#         )
#         print("[Camera] Preview started")

#     def start(self):
#         """Start video capture or AI inference."""
#         if not self.camera:
#             print("[Error] Camera not initialized")
#             return
#         print("[Camera] Starting capture...")
#         self.camera.start()

#     def stop(self):
#         """Stop video capture and release resources."""
#         if self.camera:
#             print("[Camera] Stopping capture...")
#             self.camera.stop()

#     def change_model(self, model_name: str):
#         """Load a new AI model into IMX500."""
#         model_path = os.path.join(self.model_dir, model_name + ".rpk")

#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model '{model_name}' not found in {self.model_dir}")

#         print(f"[Camera] Loading AI model: {model_name}")
#         self.imx500 = IMX500(model_path)

#     def run_inference_loop(self):
#         """Main loop — ready to integrate detection logic."""
#         print("[Camera] Running inference loop... Press Ctrl+C to exit.")
#         try:
#             while True:
#                 # TODO: Read frame, run inference, output results
#                 pass
#         except KeyboardInterrupt:
#             print("\n[Camera] Keyboard interrupt received")
#             self.stop()


def main():
    # camera_controller = RPICameraController()

    # camera_controller.change_model("imx500_network_yolo11n_pp")
    # camera_controller.start()
    # camera_controller.run_inference_loop()

    model_dir = "./models/imx500_network_yolo11n_pp.rpk"
    imx500 = IMX500(model_dir)
    intrinsics = imx500.network_intrinsics
    
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        raise RuntimeError(f"Model task '{intrinsics.task}' is not 'object detection'")
    
    print("Network Intrinsics:")
    print(intrinsics)
        

    


if __name__ == "__main__":
    main()
