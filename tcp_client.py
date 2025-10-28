
import socket
import threading


class TCPClient:
    def __init__(self, server="localhost", port=5050):
        self._server = server
        self._port = port
        self._client_socket = None
        self._connected = False

        self._header_size = 64  # Size of the header indicating message length
        self._format = 'utf-8'  # Encoding format
        self._disconnect_message = "!DISCONNECT"  # Message to signal disconnection

    def connect(self):
        """ Connect to the TCP server """
        if self._connected:
            print("Already connected to the server.")
            return

        self._client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client_socket.connect((self.__host, self.__port))
        self._connected = True
        print(f"Connected to server at {self.__host}:{self.__port}")


    def send(self, msg):
        pass
