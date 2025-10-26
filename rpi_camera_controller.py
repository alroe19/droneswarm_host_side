from __future__ import annotations
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from typing import List, Optional
import numpy as np
import os


class Detection:
    """Represents a single object detection result."""

    def __init__(self, box: np.ndarray, category: int, confidence: float) -> None:
        self.box = box
        self.category = category
        self.confidence = confidence

    def __repr__(self) -> str:
        return f"Detection(box={self.box}, category={self.category}, confidence={self.confidence:.2f})"


class RPICameraController:
    """Handles IMX500 model loading, camera setup, and inference processing."""

    def __init__(self, model_path: str, labels_path: str, conf_threshold: float = 0.55) -> None:
        """Initialize the Raspberry Pi camera and model."""
        self._model_path = model_path
        self._labels_path = labels_path
        self._conf_threshold = conf_threshold

        self._picam2: Optional[Picamera2] = None
        self._imx500_model = IMX500(model_path)
        self._intrinsics = self._initialize_intrinsics()
        self._setup_camera()

        # Create output directory for this run
        self._run_dir = self._get_new_run_dir()
        self._image_counter = 0 # Counter for saved images

    def _initialize_intrinsics(self) -> NetworkIntrinsics:
        """Initialize and configure network intrinsics."""
        intrinsics = self._imx500_model.network_intrinsics
        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "object detection"
        elif intrinsics.task != "object detection":
            print("Network is not an object detection task", file=sys.stderr)
            exit()

        # Load labels
        with open(self._labels_path, "r") as f:
            intrinsics.labels = f.read().splitlines()

        # Apply default settings
        intrinsics.bbox_normalization = True
        intrinsics.update_with_defaults()

        return intrinsics

    def _setup_camera(self) -> None:
        """Configure and start the Picamera2 for inference."""
        self._picam2 = Picamera2(self._imx500_model.camera_num)
        config = self._picam2.create_preview_configuration(
            # main={"format": "BGR888"},  # RGB format. Each pixel is laid out as [R, G, B], contrary to the BGR in the name.
            controls={"FrameRate": self._intrinsics.inference_rate},
            buffer_count=12
        )
        self._picam2.start(config, show_preview=False)

    def _get_new_run_dir(self) -> str:
        # Create base directory if it doesn't exist
        if not os.path.exists("Outputs"):
            os.makedirs("Outputs")

        # List all existing subdirectories starting with "run"
        existing_runs = [
            d for d in os.listdir("Outputs")
            if os.path.isdir(os.path.join("Outputs", d)) and d.startswith("run")
        ]

        # Extract numeric suffixes, e.g. run1 -> 1
        run_numbers = []
        for d in existing_runs:
            try:
                run_numbers.append(int(d.replace("run", "")))
            except ValueError:
                pass

        # Determine next run number
        next_run_number = max(run_numbers, default=0) + 1

        # Create new run directory
        folder_name = f"run{next_run_number:03d}"
        run_dir = os.path.join("Outputs", folder_name)

        os.makedirs(run_dir)

        return run_dir

    def _parse_detections(self, metadata: dict) -> Optional[List[Detection]]:
        """Parse raw model outputs into Detection objects."""
        np_outputs = self._imx500_model.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return None

        _, input_h = self._imx500_model.get_input_size()
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

        if self._intrinsics.bbox_normalization:
            boxes /= input_h

        # Convert to [x, y, w, h] and split into coordinates
        boxes = boxes[:, [1, 0, 3, 2]]
        box_splits = zip(*np.array_split(boxes, 4, axis=1))

        detections = [
            self._convert_detection(box, score, category, metadata)
            for box, score, category in zip(box_splits, scores, classes)
            if score > self._conf_threshold
        ]

        return detections

    def _convert_detection(self, box, confidence, category, metadata) -> Detection:
        """Convert model output into a scaled Detection object."""
        scaled_box = self._imx500_model.convert_inference_coords(box, metadata, self._picam2)
        return Detection(scaled_box, int(category), float(confidence))

    def save_image(self, request: any) -> None:
        """Save the captured image to the run directory and keep track of image counter."""

        image_path = os.path.join(self._run_dir, f"image_{self._image_counter:03d}.jpg")
        request.save("main", image_path)
        self._image_counter += 1

    def get_inference(self, save_image: bool = False) -> Optional[List[Detection]]:
        """Run inference and return detections."""
        if self._picam2 is None:
            raise RuntimeError("Camera has not been initialized.")

        request = self._picam2.capture_request()
        metadata = request.get_metadata()

        if save_image:
            self.save_image(request)

        # Check for valid metadata -> valid image and tensor outputs
        if not metadata:
            request.release()
            return None

        detections = self._parse_detections(metadata)
        request.release()
        return detections

    def close(self) -> None:
        """Gracefully stop the camera."""
        if self._picam2:
            self._picam2.stop()
            self._picam2 = None

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.close()


if __name__ == "__main__":
    model_path = "./models/network.rpk"
    labels_path = "./models/labels.txt"

    camera_controller = RPICameraController(model_path, labels_path)

    try:
        while True:
            detections = camera_controller.get_inference(save_image=True)
            if detections:
                for det in detections:
                    print(det)
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        camera_controller.close()
