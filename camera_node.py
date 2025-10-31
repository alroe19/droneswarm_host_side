#!/usr/bin/env python3

import os
from rpi_camera_controller import RPICameraController, Detection
from tcp_client import TCPClient
import json

# TODO: Jeg tror jeg har fået opdateret klassen, men tjek lige efter. Og Dectection klassen er opdateret så den har err_x og err_y attributter
# hvilket der skal tages højde for her måske. men tjek lige efter

def camera_node():

    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = script_dir + "/models/network.rpk"
    labels_path = script_dir + "/models/labels.txt"
    img_base_path = script_dir
    confidence_threshold = 0.6
    inference_rate = 10 # How many inferences per second but also the cameras frame rate
    img_save_interval = 9 # Save an image every 10 frames (0-9)
    save_images = True

    host = "localhost"
    port = 5050

    tcp_client = TCPClient(host=host, port=port)
    tcp_client.connect()

    

    camera_controller = RPICameraController(
        model_path = model_path,
        labels_path = labels_path,
        img_base_path = img_base_path,
        conf_threshold = confidence_threshold,
        inference_rate = inference_rate  
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
                detection_dict = detections[0].__dict__

                tcp_client.send(json.dumps(detection_dict))

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    camera_node()