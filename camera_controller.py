#!/usr/bin/env python3

from picamera2 import Picamera2, Preview
from picamera2.devices import IMX500
from libcamera import Transform
import os


class RPICameraController:
    def __init__(self):

        self.camera = Picamera2(imx500.camera_num)
        self.camera.start_preview(Preview.QTGL, x=100, y=200, width=800, height=600, transform=Transform(vflip=1))
        # self.camera.start_preview(Preview.NULL)

    def start(self):

        self.camera.start()


    def stop(self):

        self.camera.stop()


    def change_model(self, model_name: str):

        # Check if model exists in models folder
        if not os.path.exists("/models/" + model_name + ".rpk"):
            raise FileNotFoundError("Model: " + model_name + " not found in models folder")

        self.imx500 = IMX500("/models/" + model_name + ".rpk")

        
def main():

    camera_controller = RPICameraController()
    
    try:
        camera_controller.change_model("imx500_network_yolo11n_pp")

        camera_controller.start()
        while True:
            pass  # <-- Placeholder: detection will go here

    except KeyboardInterrupt:

        camera_controller.stop()

if __name__ == "__main__":
    main()