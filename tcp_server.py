import socket
import json
from __future__ import annotations

class TCPServer:
    def __init__(self, host="localhost", port=5050) -> None:
        self._host = host
        self._port = port
        self._server_socket = None
        self._connected = False

        self._header_size = 64  # Size of the header indicating message length
        self._format = 'utf-8'  # Encoding format

    def connect(self) -> None:
        """ Start the TCP server and wait for client to connect """
        if self._connected:
            return

        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Creates a TCP socket
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow to reuse the same address and port immediately after the program is closed. Otherwise, you may get "Address already in use" error.
        self._server_socket.bind((self._host, self._port))
        self._server_socket.listen(1)  # Listen for incoming connections

        self._client_socket, addr = self._server_socket.accept()
        self._connected = True

    def receive(self) -> str | None:
        """ Receive a message from the TCP client. This is a blocking method. """
        if not self._connected:
            raise ConnectionError("No client is connected.")

        len_header = self._client_socket.recv(self._header_size).decode(self._format)
        if len_header:
            msg_length = int(len_header)
            msg = self._client_socket.recv(msg_length).decode(self._format)
            return msg
        return None

    def __del__(self):
        if self._connected:
            self._client_socket.close()
            self._server_socket.close()
            self._connected = False


if __name__ == "__main__":
    # Test script
    server = TCPServer()
    server.connect()
    print("Waiting to receive message...")
    message = json.loads(server.receive())
    print(f"Received message: {message}")
