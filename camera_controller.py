#!/usr/bin/env python3

from picamera2 import Picamera2, Preview
from libcamera import Transform

class CameraNode:
    def __init__(self):
        print("[Camera Node] Initializing camera and model...")
        self.camera = Picamera2()
        self.camera.start_preview(Preview.QTGL, x=100, y=200, width=800, height=600, transform=Transform(hflip=1))

    def start(self):
        print("[Camera Node] Starting detection loop...")
        self.camera.start()

    def stop(self):
        print("[Camera Node] Stopping camera...")
        self.camera.stop()

def main():
    print("[System] Camera Node starting...")
    camera = CameraNode()
    
    try:
        camera.start()
        # Placeholder loop (simulate running)
        while True:
            pass  # <-- Placeholder: detection will go here

    except KeyboardInterrupt:
        print("\n[System] Shutting down...")
        camera.stop()

if __name__ == "__main__":
    main()