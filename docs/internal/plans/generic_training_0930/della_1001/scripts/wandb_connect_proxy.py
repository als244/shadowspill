"""Local HTTPS CONNECT relay for W&B on an offline Slurm compute node.

Run on the head node and forward its loopback port with SSH. TLS, credentials,
and request bodies remain between the SDK and W&B; this relay logs only hosts.
"""
import argparse
import select
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Relay(BaseHTTPRequestHandler):
    def do_CONNECT(self):
        host, _, port = self.path.rpartition(":")
        allowed = ("wandb.ai", "wandb.io", "googleapis.com", "amazonaws.com")
        if port != "443" or not any(host == d or host.endswith("." + d) for d in allowed):
            self.send_error(403)
            return
        try:
            upstream = socket.create_connection((host, 443), timeout=20)
        except OSError:
            self.send_error(502)
            return
        with upstream:
            self.send_response(200, "Connection established")
            self.end_headers()
            self.wfile.flush()
            sockets = (self.connection, upstream)
            while True:
                ready, _, _ = select.select(sockets, (), (), 180)
                if not ready:
                    return
                for source in ready:
                    data = source.recv(65536)
                    if not data:
                        return
                    destination = upstream if source is self.connection else self.connection
                    destination.sendall(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18375)
    args = parser.parse_args()
    print(f"W&B HTTPS relay on 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Relay).serve_forever()
