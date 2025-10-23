#!/usr/bin/env python3

from picamera2 import Picamera2, MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
import logging


logging.basicConfig(level=logging.INFO)


def upload_and_start_camera():

    pass


def host_node():
    logging.info("Host node started.")
    
    pass

if __name__ == "__main__":
    host_node()