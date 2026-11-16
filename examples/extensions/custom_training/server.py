"""Start in an existing GPU allocation with SHADOWSPILL_TOKEN set."""

from shadowspill.server import Allocation, Server

server = Server(
    allocation=Allocation.from_environment(),
    storage_root="./custom-training-data",
    workload_packages=["example-training-definitions"],
)
server.serve(host="127.0.0.1", port=8000)
