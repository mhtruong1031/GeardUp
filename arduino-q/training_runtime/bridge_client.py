#!/usr/bin/env python3
"""
Bridge client for training_runtime.
Connects to Arduino MCU via Unix socket (arduino-router) or uses arduino.app_utils when on device.
Reference: arduino_uno_q_knowledge_base_and_playground/knowledge-base/Example-1/mpu/bridge_client.py
"""

import socket
import struct
import logging

logger = logging.getLogger(__name__)

try:
    import msgpack
except ImportError:
    msgpack = None


def get_bridge(socket_path="/var/run/arduino-router.sock"):
    """
    Return a Bridge instance. Prefer arduino.app_utils.Bridge when available (MPU on device).
    Otherwise use SocketBridge to arduino-router.
    """
    try:
        from arduino.app_utils import Bridge as AppBridge
        return AppBridge()
    except ImportError:
        pass
    if msgpack is None:
        raise ImportError("msgpack is required for SocketBridge. Install with: pip install msgpack")
    return SocketBridge(socket_path)


class SocketBridge:
    """Bridge client over Unix socket (msgpack RPC). Used when arduino.app_utils is not available."""

    def __init__(self, socket_path="/var/run/arduino-router.sock"):
        self.socket_path = socket_path
        self.sock = None
        self.msg_id = 0
        self.connect()

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(self.socket_path)
            logger.info("Connected to arduino-router at %s", self.socket_path)
        except Exception as e:
            logger.error("Failed to connect to arduino-router: %s", e)
            self.sock = None

    def call(self, method, *args):
        if self.sock is None:
            raise RuntimeError("Not connected to arduino-router")
        if msgpack is None:
            raise RuntimeError("msgpack required")
        self.msg_id += 1
        request = [0, self.msg_id, method, list(args)]
        data = msgpack.packb(request)
        self.sock.sendall(struct.pack(">I", len(data)))
        self.sock.sendall(data)
        length_bytes = self.sock.recv(4)
        if len(length_bytes) != 4:
            raise Exception("Failed to read response length")
        length = struct.unpack(">I", length_bytes)[0]
        response_data = b""
        while len(response_data) < length:
            chunk = self.sock.recv(length - len(response_data))
            if not chunk:
                raise Exception("Connection closed")
            response_data += chunk
        response = msgpack.unpackb(response_data, raw=False)
        if response[0] == 1:
            return response[2] if len(response) > 2 else None
        if response[0] == 2:
            err = response[2] if len(response) > 2 else "Unknown error"
            raise Exception(f"RPC error: {err}")
        raise Exception(f"Unknown response type: {response[0]}")

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
