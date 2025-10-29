
import socket
from __future__ import annotations

class TCPClient:
    def __init__(self, host="localhost", port=5050) -> None:
        self._host = host
        self._port = port
        self._client_socket = None
        self._connected = False

        self._header_size = 64  # Size of the header indicating message length
        self._format = 'utf-8'  # Encoding format

    def connect(self) -> None:
        """ Connect to the TCP server. Blocks until connection is established. """
        if self._connected:
            return

        self._client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client_socket.connect((self._host, self._port))
        self._connected = True


    def send(self, msg) -> None:
        """ Send a message to the TCP server. Blocks until the message is sent. """
        if not self._connected:
            raise ConnectionError("Not connected to the server.")

        msg_encoded = msg.encode(self._format)  # Encoded message
        msg_length = len(msg_encoded) # Length of the encoded message
        len_header = str(msg_length).encode(self._format) # Message represents the length of the message
        len_header += b' ' * (self._header_size - len(len_header))  # Padding the header to a fixed size of self._header_size

        # Send the header and the message
        self._client_socket.sendall(len_header)
        self._client_socket.sendall(msg_encoded)

    def __del__(self) -> None:
        if self._connected:
            self._client_socket.close()
            self._connected = False


if __name__ == "__main__":
    # Test script
    client = TCPClient()
    client.connect()
    client.send("Hello, Server!")
