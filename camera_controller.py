#!/usr/bin/env python3

from picamera2 import Picamera2, MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from functools import lru_cache
import numpy as np
import cv2
import logging




class Detection:
    def __init__(self, box, category, conf):
        self.box = box  # [ymin, xmin, ymax, xmax] normalized
        self.category = category
        self.conf = conf


class RPICameraController:
    def __init__(self, model_path=None, labels_path=None):

        ## Initiate IMX500 and Picamera2 variables
        self.__picam2 = None
        self.__imx500_active_model = None
        self.__intrinsics = None
        self.__running = False

        self.__conf_threshold = 0.55 # Confidence threshold
        self.__max_detections = 3 # Maximum number of detections

        if model_path != None and labels_path != None:
            self.load_model(model_path, labels_path)


    def update_threshold(self, new_threshold):
        """Update the confidence threshold."""
        self.__conf_threshold = new_threshold
        logging.info(f"Updated confidence threshold: {new_threshold}")
    

    def update_max_detections(self, new_max):
        """Update the maximum number of detections."""
        self.__max_detections = new_max
        logging.info(f"Updated maximum detections: {new_max}")


    def load_model(self, model_path, labels_path=None):
        """Load or switch the AI model."""
        logging.info(f"Loading model: {model_path}")

        # Stop camera if running
        if self.__running:
            self.stop()

        # Load IMX500 model
        self.__imx500_active_model = IMX500(model_path)
        self.__intrinsics = self.__imx500_active_model.network_intrinsics

        # Ensure intrinsics are set
        if not self.__intrinsics:
            self.__intrinsics = NetworkIntrinsics()
            self.__intrinsics.task = "object detection"
        elif self.__intrinsics.task != "object detection":
            raise RuntimeError("This model is not an object detection network.")

        # Load custom labels if provided
        if labels_path:
            with open(labels_path, "r") as f:
                self.__intrinsics.labels = f.read().splitlines()

        self.__intrinsics.update_with_defaults()

        # Prepare camera after model load
        self.__picam2 = Picamera2(self.__imx500_active_model.camera_num)
        logging.info("Model loaded successfully.")


    def start(self):
        """Start camera inference."""
        if not self.__picam2:
            raise RuntimeError("No model loaded — call load_model() first.")
        
        config = self.__picam2.create_preview_configuration(
            main = {"format": "RGB888"},
            controls = {"FrameRate": self.__intrinsics.inference_rate},
            buffer_count=12
        )

        

        self.__picam2.start(config, show_preview=False)
        self.__running = True
        logging.info("Camera started.")


    def stop(self):
        """Stop camera."""
        if self.__picam2:
            self.__picam2.stop()
            self.__running = False
            logging.info("Camera stopped.")


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
            self.__make_detection(box, conf, category, metadata)
            for box, conf, category in zip(boxes, conf, classes)
            if conf > self.__conf_threshold
        ]
        
        return detections[:self.__max_detections] # Return only up to max_detections. Also works when actual detections are fewer than max_detections.
    

    def __make_detection(self, box, conf, category, metadata):
        """Convert raw detection to Detection object with scaled box coordinates."""
        box = self.__imx500_active_model.convert_inference_coords(box, metadata, self.__picam2)
        return Detection(box, category, conf)

    @lru_cache
    def __get_labels(self):
        labels = self.__intrinsics.labels

        if self.__intrinsics.ignore_dash_labels:
            labels = [label for label in labels if label and label != "-"]
        return labels

    def __draw_detections(self, detections, request, stream="main"):
        """Draw the detections for this request onto the ISP output."""
        
        labels = self.__get_labels() # Retrieve labels

        # Draw detections
        with MappedArray(request, stream) as m:
            # Create a copy of the array to draw the background with opacity
            detection_frame = m.array.copy()  # single copy per frame

            for detection in detections:
                x, y, w, h = map(int, detection.box)

                label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

                # Calculate text size and position
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_x = x + 5
                text_y = y + 15

                # Draw the background rectangle on the detection_frame
                cv2.rectangle(detection_frame,
                            (text_x, text_y - text_height),
                            (text_x + text_width, text_y + baseline),
                            (255, 255, 255),  # Background color (white)
                            cv2.FILLED)

                alpha = 0.30
                cv2.addWeighted(detection_frame, alpha, m.array, 1 - alpha, 0, m.array)

                # Draw text on top of the background
                cv2.putText(m.array, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Draw detection box
                cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

            if self.__intrinsics.preserve_aspect_ratio:
                b_x, b_y, b_w, b_h = self.__imx500_active_model.get_roi_scaled(request)
                color = (255, 0, 0)  # red
                cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0))

        return detection_frame


    def capture(self, draw_detection = False):
        """Fetch a new request and its metadata."""
        request = self.__picam2.capture_request()
        metadata = request.get_metadata()
        detections = None
        
        # Check if metadata is present, if so the frame is valid
        if not metadata:
            request.release()
            return None
        
        detections = self.__parse_detections(metadata)

        if draw_detection and detections:
            detection_frame = self.__draw_detections(detections, request)
            request.release()
            return {
                "detections": detections,
                "detection_frame": detection_frame
            }
        
        else: 
            request.release()
            return {
                "detections": detections,
                "detection_frame": None
            }


    def close(self):
        """Cleanup"""
        self.stop()
        logging.info("Controller closed.")


if __name__ == "__main__":

    controller = RPICameraController()
    controller.load_model("/models/imx500_network_yolo11n_pp.rpk", "coco_labels.txt")
    controller.start()

    try:
        while True:
            results = controller.capture(draw_detection=True)
            if results["detection_frame"] is not None:
                detection = results["detections"]
                print("Objects coordinates on image:")
                for obj in detection:
                    print(f" - {obj['label']}: {obj['box']}")
    except KeyboardInterrupt:
        pass
    
    controller.close()