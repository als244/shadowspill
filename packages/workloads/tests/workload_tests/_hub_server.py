"""A real HTTP fixture serving immutable HF metadata and tiny tensor shards."""

import hashlib
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import unquote, urlsplit


@contextmanager
def serve_hub(directory, *, reference="fixture/llama", revision="a" * 40):
    files = {
        path.name: (path, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in directory.iterdir()
        if path.is_file()
    }
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.respond(False)

        def do_GET(self):
            self.respond(True)

        def log_message(self, *args):
            pass

        def respond(self, body):
            prefix = f"/{reference}/resolve/{revision}/"
            path = unquote(urlsplit(self.path).path)
            name = path.removeprefix(prefix)
            if not path.startswith(prefix) or name not in files:
                self.send_response(404)
                self.send_header("X-Error-Code", "EntryNotFound")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            source, checksum = files[name]
            requests.append((self.command, name))
            self.send_response(200)
            self.send_header("Content-Length", str(source.stat().st_size))
            self.send_header("ETag", f'"{checksum}"')
            self.send_header("X-Repo-Commit", revision)
            self.end_headers()
            if body:
                with source.open("rb") as stream:
                    while data := stream.read(1 << 20):
                        self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
