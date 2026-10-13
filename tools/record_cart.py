#!/usr/bin/env python3
"""Record raw IMU data from the real cart for training (standard library only).

Over the cart's WiFi (join "SuspensionCart" first; no cable while driving):
    python3 tools/record_cart.py gravel1.csv
Over USB serial:
    python3 tools/record_cart.py gravel1.csv --serial /dev/cu.usbserial-0001

Turns on 'log raw', saves every R,... line to the file and turns logging off again
on Ctrl+C (or after --seconds). Drive on ONE surface per recording; then label it:
    host/build/bin/replay gravel1.csv --label gravel --run 1 >> host/data/real.csv
"""

import argparse
import os
import sys
import time
import urllib.request


def wifi_lines(host):
    """Yield the cart's output lines from the dashboard stream."""
    resp = urllib.request.urlopen("http://%s/events" % host, timeout=10)
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith("data: "):
            yield line[6:]


def wifi_send(host, cmd):
    req = urllib.request.Request("http://%s/cmd" % host, data=cmd.encode(), method="POST")
    urllib.request.urlopen(req, timeout=5).read()


class SerialPort:
    def __init__(self, device, baud=115200):
        import termios

        self.fd = os.open(device, os.O_RDWR | os.O_NOCTTY)
        attrs = termios.tcgetattr(self.fd)
        speed = getattr(termios, "B%d" % baud)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[3] = 0
        attrs[4] = attrs[5] = speed
        attrs[6][termios.VMIN] = 1
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        time.sleep(0.3)

    def lines(self):
        buf = b""
        while True:
            chunk = os.read(self.fd, 4096)
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line.decode("utf-8", "replace").rstrip("\r")

    def send(self, cmd):
        os.write(self.fd, (cmd + "\n").encode())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output file, e.g. gravel1.csv")
    ap.add_argument("--host", default="192.168.4.1", help="cart address on its WiFi (default 192.168.4.1)")
    ap.add_argument("--serial", metavar="DEVICE", help="use USB serial instead of WiFi")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (default: Ctrl+C)")
    args = ap.parse_args()

    if args.serial:
        port = SerialPort(args.serial)
        lines, send = port.lines(), port.send
    else:
        lines, send = wifi_lines(args.host), lambda c: wifi_send(args.host, c)

    send("log raw")
    n, t0 = 0, time.time()
    print("recording to %s - drive now, Ctrl+C to stop" % args.out)
    try:
        with open(args.out, "w") as f:
            for line in lines:
                if line.startswith("R,"):
                    f.write(line + "\n")
                    n += 1
                    if n % 500 == 0:
                        print("  %d samples (%.0f s)" % (n, n / 100.0), flush=True)
                elif line and not line.startswith(("J,", "S,")):
                    print("  cart: " + line)  # events, faults
                if args.seconds and time.time() - t0 >= args.seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            send("log off")
        except OSError:
            pass
    print("saved %d samples (%.1f s of data) to %s" % (n, n / 100.0, args.out))
    if n == 0:
        sys.exit("no data received: is the cart on, and are you on its WiFi / the right serial port?")


if __name__ == "__main__":
    main()
