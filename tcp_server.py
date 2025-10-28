import socket


HOST = "localhost"
PORT = 5050
SERVER_ADDR = (HOST, PORT)
FORMAT = 'utf-8'
HEADER_SIZE = 64
DISCONNECT_MESSAGE = "!DISCONNECT"

def node():
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:

        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow to reuse the same address and port immediately after the program is closed
        server.bind(SERVER_ADDR) # Bind the socket to the address
        server.listen(1) # Start listening for incomming connections and only 1 pending connection allowed
        print(f"[LISTENING] Server is listening on {HOST}:{PORT}")

        # wait for a connection host side node and accept it
        conn, addr = server.accept()
        print(f"[NEW CONNECTION] {addr} connected.")


        try:
            while True:
                msg_length = conn.recv(HEADER_SIZE).decode(FORMAT)
                if msg_length:
                    msg_length = int(msg_length)
                    msg = conn.recv(msg_length).decode(FORMAT)

                    if msg == DISCONNECT_MESSAGE:
                        print(f"[DISCONNECTED] {addr} disconnected.")
                        break

                    print(f"[RECEIVED FROM {addr}] {msg}")

        finally:
            conn.close()

if __name__ == "__main__":
    node()
