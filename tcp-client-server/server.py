
import socket
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCPServer:
    def __init__(self, host = "0.0.0.0", port = 5050, on_msg_callback = None):
        self.__host = host
        self.__port = port
        self.__server_socket = None
        self.__client_socket = None
        self.__connected = False

        if on_msg_callback is not None and not callable(on_msg_callback):
            raise TypeError("callback must be a callable (function/method)")
        self.__on_msg_callback = on_msg_callback # Function to call when a message is received

        self.__header_size = 64  # Size of the header indicating message length
        self.__format = 'utf-8'  # Encoding format


    def start(self):
        """ Start the TCP server"""
        if self.__running:
            logger.warning("Server is already running.")
            return
        
        logger.info(f"Starting TCP server on {self.__host}:{self.__port} ...")

        self.__server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__server.bind((self.__host, self.__port))
        self.__server.listen(1)

        conn, addr = self.__server.accept()

        logger.info(f"New connection - {addr} connected.")

        # Start sender and receiver threads
        sender_thread = threading.Thread(target=self.sender_thread, args=(conn, addr), daemon=True)
        sender_thread.start()
        self.__connected = True

        self.receiver()

    def sender_thread(self, conn: socket.socket, addr):
        pass

    def receiver(self, conn: socket.socket, addr):
        
        while self.__connected:
            
    


    def send(self, msg):
        pass






