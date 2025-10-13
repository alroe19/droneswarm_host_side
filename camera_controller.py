from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
import numpy as np

class Detection:
    def __init__(self, box, category, score):
        self.box = box  # [ymin, xmin, ymax, xmax] normalized
        self.category = int(category)
        self.score = float(score)




class RPICameraController:
    def __init__(self, model_path=None, labels_path=None):

        ## Initiate IMX500 and Picamera2 variables
        self.picam2 = None
        self.imx500_active_model = None
        self.intrinsics = None
        self.running = False

        self.iou = 0.65 # Intersection over Union
        self.conf_threshold = 0.55 # Confidence threshold
        self.max_detections = 3 # Maximum number of detections

        if model_path != None and labels_path != None:
            self.load_model(model_path, labels_path)


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
            controls={"FrameRate": self.intrinsics.inference_rate},
            buffer_count=12
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

    def parse_detections(self, metadata: dict):

        """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
        bbox_normalization = self.intrinsics.bbox_normalization

        np_outputs = self.imx500_active_model.get_outputs(metadata, add_batch=True)
        __, input_h = self.imx500_active_model.get_input_size()

        # Check for output, if none return none
        if np_outputs is None:
            return None

        # Skal måske ændre hvis vi bruger en anden model som ikke er SSD
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

        detections = [
            Detection(category,
                      score,
                      box=self.imx500_active_model.convert_inference_coords(box, metadata, self.picam2),
                      )
            for box, score, category in zip(boxes, scores, classes)
            if score > self.conf_threshold
        ]
        return detections


    # def get_detections(self):
    #     """Return AI detections from the current frame."""
    #     if not self.running:
    #         return []

    #     metadata = self.picam2.capture_metadata()
    #     outputs = self.imx500_model.get_outputs(metadata, add_batch=True)
    #     if outputs is None:
    #         return []

    #     boxes, scores, labels = outputs[0][0], outputs[1][0], outputs[2][0]
    #     detections = [
    #         (box, float(score), int(label))
    #         for box, score, label in zip(boxes, scores, labels)
    #         if score > 0.5  # Default threshold
    #     ]
    #     return detections

    def close(self):
        """Cleanup"""
        self.stop()
        print("Controller closed.")


if __name__ == "__main__":

    controller = RPICameraController()
    controller.load_model("   ", "coco_labels.txt")
    controller.start()

    try:
        while True:
            detections = controller.get_detections()
            if detections:
                print(f"Detections: {detections}")
    except KeyboardInterrupt:
        pass
    
    controller.close()