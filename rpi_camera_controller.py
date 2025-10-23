
from picamera2 import Picamera2, MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class Detection:
    def __init__(self, box, category, conf):
        self.box = box  # [ymin, xmin, ymax, xmax] normalized
        self.category = category
        self.conf = conf

class RPICameraController:
    def __init__(self, model_path, labels_path):

        ## Initiate IMX500 and Picamera2 variables
        self.__picam2 = None
        self.__imx500_active_model = IMX500(model_path)
        self.__intrinsics = self.__imx500_active_model.network_intrinsics

        self.__conf_threshold = 0.55 # Confidence threshold
        self.__max_detections = 3 # Maximum number of detections

        # Ensure intrinsics are set
        if not self.__intrinsics:
            self.__intrinsics = NetworkIntrinsics()
            self.__intrinsics.task = "object detection"
        elif self.__intrinsics.task != "object detection":
            raise RuntimeError("This model is not an object detection network.")

        # Load custom labels
        with open(labels_path, "r") as f:
            self.__intrinsics.labels = f.read().splitlines()

        self.__intrinsics.update_with_defaults()

        logging.info("Model intrinsics:")
        logging.info(self.__intrinsics)

        # Prepare camera after model load
        self.__picam2 = Picamera2(self.__imx500_active_model.camera_num)
        logging.info("Model loaded successfully.")


        # Set camera configurations and start camera
        config = self.__picam2.create_preview_configuration(
            # main = {"format": "RGB888"},
            controls = {"FrameRate": self.__intrinsics.inference_rate},
            buffer_count=12
        )

        self.__picam2.start(config, show_preview=False)
        logging.info("Camera started.")

    def __parse_detections(self, metadata: dict):

        """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
        bbox_normalization = self.__intrinsics.bbox_normalization

        np_outputs = self.__imx500_active_model.get_outputs(metadata, add_batch=True)
        __, input_h = self.__imx500_active_model.get_input_size()

        # Check for output, if none return none
        if np_outputs is None:
            return None

        # Skal måske ændre hvis vi bruger en anden model som ikke er SSD
        boxes, conf, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

        detections = [
            self.__convert_detection(box, conf, category, metadata)
            for box, conf, category in zip(boxes, conf, classes)
            if conf > self.__conf_threshold
        ]
        
        return detections[:self.__max_detections] # Return only up to max_detections. Also works when actual detections are fewer than max_detections.


    def __convert_detection(self, box, conf, category, metadata):
        """Convert raw detection to Detection object with scaled box coordinates."""
        
        box = self.__imx500_active_model.convert_inference_coords(box, metadata, self.__picam2)
        return Detection(box, category, conf)


    def get_inference(self):
        """ Get inference results from the camera buffer."""

        request = self.__picam2.capture_request()
        metadata = request.get_metadata()
        detections = None
        
        # Check if metadata is present, if so the frame is valid
        if not metadata:
            request.release()
            return None
        
        detections = self.__parse_detections(metadata)

        request.release()
        return detections



    def __del__(self):
        """Destructor to stop camera when controller is deleted."""
        self.__picam2.stop()
        logging.info("Camera stopped.")


if __name__ == "__main__":

    model_path = "./models/network.rpk"
    labels_path = "./models/labels.txt"

    camera_controller = RPICameraController(model_path, labels_path)

    try:
        while True:
            detections = camera_controller.get_inference()
            # print detections box, category and confidence
            if detections:
                for det in detections:
                    print(f"Box: {det.box}, Category: {det.category}, Confidence: {det.conf}")
    except KeyboardInterrupt:
        pass