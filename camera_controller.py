#!/usr/bin/env python3

class CameraNode:
    def __init__(self):
        print("[Camera Node] Initializing camera and model...")
        # Later: load model, initialize camera (Picamera2, etc.)

    def start(self):
        print("[Camera Node] Starting detection loop...")
        # Later: start grabbing frames and running inference

    def stop(self):
        print("[Camera Node] Stopping camera...")
        # Later: cleanup

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