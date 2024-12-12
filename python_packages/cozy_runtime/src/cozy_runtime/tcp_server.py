import logging
import socket
import threading
from typing import Callable, Dict, Optional, Union


class RequestContext:
    def __init__(self, connection: socket.socket, addr: str):
        self.connection = connection
        self.addr = addr
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    def data(self) -> bytes:
        try:
            # size = self.connection.recv(4)
            data = self.connection.recv(1024 * 1024)
            if not data:  # Handle case where connection is closed
                raise ConnectionResetError("Connection closed by client")
            return data
        except socket.timeout:
            raise TimeoutError(f"Timeout waiting for data from {self.addr}")
        except Exception as e:
            raise ConnectionError(f"Error receiving data: {e}")

    def send(self, data: Union[bytes, str]):
        if self.closed:
            raise ConnectionError("Cannot send data, connection is closed.")
        try:
            if isinstance(data, str):
                data = data.encode()
            self.connection.sendall(data)
        except Exception:
            raise

    # def send_final(self, data: Union[bytes, str]):
    #     if self.closed:
    #         raise ConnectionError("Cannot send data, connection is closed.")
    #     self.finished = True
    #     self.send(data)

    def end(self):
        if not self.closed:
            try:
                self.connection.close()
            except Exception as e:
                logging.error(f"Failed to close connection {self.addr}: {e}")
            finally:
                self.closed = True


class TCPServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8888,
        max_connections: int = 1000,
        timeout: float = 30.0,
        max_threads: int = 50,  # Added max_threads to limit number of active threads
    ):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.timeout = timeout
        self.max_connections = max_connections
        self.connections: Dict[socket.socket, str] = {}
        self.logger = self._setup_logger()
        self.lock = threading.Lock()
        self.handler = self.default_handler
        self.threads = []
        self.max_threads = max_threads

    @staticmethod
    def _setup_logger():
        logger = logging.getLogger("TCPServer")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _setup_socket(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(self.max_connections)
            self.socket.settimeout(1.0)
        except Exception as e:
            self.logger.error(f"Socket setup failed: {e}")
            raise

    def set_handler(self, handler: Callable[[RequestContext], Union[bytes, str, None]]):
        self.handler = handler

    def default_handler(self, context: RequestContext) -> None:
        context.send("No handler set.".encode())

    def start(self, callback: Optional[Callable[[str, int], None]] = None):
        try:
            self.running = True
            self._setup_socket()

            if callback is not None:
                address, port = self.socket.getsockname()
                callback(address, port)

            while self.running:
                try:
                    connection, addr = self.socket.accept()
                    connection.settimeout(self.timeout)

                    self._cleanup_threads()

                    if len(self.threads) >= self.max_threads:
                        self.logger.warning(
                            "Max thread limit reached, rejecting connection"
                        )
                        connection.close()
                        continue

                    connection_thread = threading.Thread(
                        target=self.handle_connection,
                        args=(connection, addr),
                    )
                    connection_thread.start()
                    self.threads.append(connection_thread)

                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.error(f"Error accepting connection: {e}")

        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()

        with self.lock:
            for connection in list(self.connections.keys()):
                try:
                    connection.close()
                except Exception:
                    pass
            self.connections.clear()

        self.logger.info("Server stopped")

    def handle_connection(self, connection: socket.socket, addr: str):
        with self.lock:
            self.connections[connection] = addr

        try:
            with RequestContext(connection, addr) as context:
                response = self.handler(context)
                if response:
                    context.send(response)

        except TimeoutError as e:
            self.logger.warning(f"Timeout for connection {addr}: {e}")
        except ConnectionResetError as e:
            self.logger.warning(f"Connection reset by client {addr}: {e}")
        except ConnectionError as e:
            self.logger.error(f"Connection error for {addr}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error handling connection {addr}: {e}")
        finally:
            with self.lock:
                del self.connections[connection]

    def _cleanup_threads(self):
        # Clean up finished threads to prevent resource leakage
        self.threads = [t for t in self.threads if t.is_alive()]
