from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

class RPICameraController:
    def __init__(self):
        self.picam2 = None
        self.imx500_model = None
        self.intrinsics = None
        self.running = False

    def load_model(self, model_path, labels_path=None):
        """Load or switch the AI model."""
        print(f"Loading model: {model_path}")

        # Stop camera if running
        if self.running:
            self.stop()

        # Load IMX500 model
        self.imx500_model = IMX500(model_path)
        self.intrinsics = self.imx500_model.network_intrinsics

        # Ensure intrinsics are set
        if not self.intrinsics:
            self.intrinsics = NetworkIntrinsics()
            self.intrinsics.task = "object detection"
        elif self.intrinsics.task != "object detection":
            raise RuntimeError("This model is not an object detection network.")

        # Load custom labels if provided
        if labels_path:
            with open(labels_path, "r") as f:
                self.intrinsics.labels = f.read().splitlines()

        self.intrinsics.update_with_defaults()

        # Prepare camera after model load
        self.picam2 = Picamera2(self.imx500_model.camera_num)
        print("Model loaded successfully.")

    def start(self):
        """Start camera inference."""
        if not self.picam2:
            raise RuntimeError("No model loaded — call load_model() first.")
        config = self.picam2.create_preview_configuration(
            controls={"FrameRate": self.intrinsics.inference_rate}
        )
        self.picam2.start(config, show_preview=False)
        self.running = True
        print("Camera started.")

    def stop(self):
        """Stop camera."""
        if self.picam2:
            self.picam2.stop()
            self.running = False
            print("Camera stopped.")

    def get_detections(self):
        """Return AI detections from the current frame."""
        if not self.running:
            return []

        metadata = self.picam2.capture_metadata()
        outputs = self.imx500_model.get_outputs(metadata, add_batch=True)
        if outputs is None:
            return []

        boxes, scores, labels = outputs[0][0], outputs[1][0], outputs[2][0]
        detections = [
            (box, float(score), int(label))
            for box, score, label in zip(boxes, scores, labels)
            if score > 0.5  # Default threshold
        ]
        return detections

    def close(self):
        """Cleanup"""
        self.stop()
        print("Controller closed.")


if __name__ == "__main__":

    controller = RPICameraController()
    controller.load_model("./model/imx500_network_yolo11n_pp.rpk", "coco_labels.txt")
    controller.start()

    try:
        while True:
            detections = controller.get_detections()
            print(f"Detections: {detections}")
    except KeyboardInterrupt:
        pass
    
    controller.close()