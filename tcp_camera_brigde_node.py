#! /usr/bin/env python3

import rclpy 
from rclpy.node import Node
from tcp_server import TCPServer


class TCPCameraBridgeNode(Node):
    def __init__(self):
        super().__init__('tcp_camera_bridge_node')
        self.server = TCPServer(host='localhost', port=8000)
        self.server.connect()
        