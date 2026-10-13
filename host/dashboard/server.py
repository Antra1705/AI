#!/usr/bin/env python3
"""Dashboard server: relays the cart's telemetry to a browser (standard library only).

    python3 host/dashboard/server.py --sim              simulated cart (host/build/bin/sim live)
    python3 host/dashboard/server.py --sim --speed 2    ...running twice as fast as real time
    python3 host/dashboard/server.py --serial /dev/cu.usbserial-0001
                                                        real cart on USB serial (115200 baud)
Then open http://localhost:8080

Browser <- GET /events   Server-Sent Events, one message per line from the cart
                         (J,{...} snapshots, S,{...} simulator truth, event text)
Browser -> POST /cmd     one command line ("start", "maint test", "sim wall", ...)

Once the hardware is here, the ESP32 can serve index.html itself over WiFi with
the same /events and /cmd endpoints; the page does not change.
"""

import argparse
import http.server
import os
import queue
import socketserver
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "build", "bin", "sim")


class Hub:
    """Fan-out of cart lines to every connected browser, plus a short history."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clients = []
        self.history = []  # recent event (non-telemetry) lines for late joiners

    def publish(self, line):
        with self.lock:
            if not (line.startswith("J,") or line.startswith("S,")):
                self.history = (self.history + [line])[-200:]
            for q in list(self.clients):
                try:
                    q.put_nowait(line)
                except queue.Full:
                    pass  # slow browser: drop rather than block the cart

    def subscribe(self):
        q = queue.Queue(maxsize=2000)
        with self.lock:
            for line in self.history:
                q.put_nowait(line)
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)


class SimSource:
    def __init__(self, hub, speed):
        if not os.path.exists(SIM):
            sys.exit("Simulator not built. Run:  make -C host all")
        self.proc = subprocess.Popen([SIM, "live", "--speed", str(speed)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self.hub = hub
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.hub.publish(line.rstrip("\n"))
        self.hub.publish("# simulator stopped")

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()


class SerialSource:
    """Real cart on a USB serial port, configured with termios (no pyserial needed)."""

    def __init__(self, hub, device, baud=115200):
        import termios
        self.fd = os.open(device, os.O_RDWR | os.O_NOCTTY)
        attrs = termios.tcgetattr(self.fd)
        speed = getattr(termios, "B%d" % baud)
        attrs[0] = 0                                                  # iflag: raw
        attrs[1] = 0                                                  # oflag: raw
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL       # cflag: 8N1
        attrs[3] = 0                                                  # lflag: no echo, non-canonical
        attrs[4] = attrs[5] = speed
        attrs[6][termios.VMIN] = 1
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        self.hub = hub
        threading.Thread(target=self._read, daemon=True).start()
        time.sleep(0.5)
        self.send("log json")

    def _read(self):
        buf = b""
        while True:
            chunk = os.read(self.fd, 4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self.hub.publish(line.decode("utf-8", "replace").rstrip("\r"))

    def send(self, line):
        os.write(self.fd, (line + "\n").encode())


def make_handler(hub, source):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q = hub.subscribe()
                try:
                    while True:
                        try:
                            line = q.get(timeout=10)
                        except queue.Empty:
                            line = ""  # keep-alive comment below
                        msg = ("data: %s\n\n" % line) if line else ": ping\n\n"
                        self.wfile.write(msg.encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    hub.unsubscribe(q)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/cmd":
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length", "0"))
            line = self.rfile.read(n).decode("utf-8", "replace").strip().splitlines()
            if line and len(line[0]) < 80:
                source.send(line[0])
                hub.publish("> " + line[0])
            self.send_response(204)
            self.end_headers()

    return Handler


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sim", action="store_true", help="run the simulated cart")
    src.add_argument("--serial", metavar="DEVICE", help="real cart on this serial port")
    ap.add_argument("--speed", type=float, default=1.0, help="simulation speed (with --sim)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    hub = Hub()
    source = SimSource(hub, args.speed) if args.sim else SerialSource(hub, args.serial)
    httpd = Server(("127.0.0.1", args.port), make_handler(hub, source))
    print("Dashboard: http://localhost:%d   (Ctrl+C to stop)" % args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
