from sympy import det
from rpi_camera_controller import RPICameraController, Detection
from tcp_client import TCPClient
import json


def camera_node():

    model_path = "./models/network.rpk"
    labels_path = "./models/labels.txt"
    confidence_threshold = 0.6
    save_images = True

    host = "localhost"
    port = 5050

    tcp_client = TCPClient(host=host, port=port)
    tcp_client.connect()

    img_save_interval = 9 # Save an image every 10 frames (0-9)

    camera_controller = RPICameraController(
        model_path=model_path,
        labels_path=labels_path,
        conf_threshold=confidence_threshold,
    )

    frame_count = 0

    try:
        while True:
            ## Call get_inference and set save_image to True every img_save_interval frames
            detections = camera_controller.get_inference(save_image=(frame_count % img_save_interval == 0 and save_images))
            frame_count = (frame_count + 1) % img_save_interval
            
            # The get_inference method can return multiple detections, but we only send the first one for simplicity
            # This can cause problems if multiple objects are detected, which can get the detection to flicker between them.
            # The detection object contains bbox, category, confidence attributes which is converted to a dictionary and sent as a JSON string.
            if detections:
                detection_dict = [
                    {
                        "bbox": det.box.tolist(),
                        "category": det.category,
                        "confidence": det.confidence,
                    }
                    for det in detections[0]
                ]

                tcp_client.send(json.dumps(detection_dict))

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    camera_node()