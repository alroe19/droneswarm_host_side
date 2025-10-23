
from picamera2 import Picamera2, MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
import logging

logging.basicConfig(level=logging.INFO)


class RPICameraController:
    def __init__(self, model_path, labels_path):

        ## Initiate IMX500 and Picamera2 variables
        self.__picam2 = None
        self.__imx500_active_model = IMX500(model_path)
        self.__intrinsics = self.__imx500_active_model.network_intrinsics

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
            main = {"format": "RGB888"},
            controls = {"FrameRate": self.__intrinsics.inference_rate},
            buffer_count=12
        )

        self.__picam2.start(config, show_preview=False)
        logging.info("Camera started.")



if __name__ == "__main__":

    model_path = "path/to/your/model.blob"
    labels_path = "path/to/your/labels.txt"

    camera_controller = RPICameraController(model_path, labels_path)