"""Independent stdlib archive writer for checkpoint corruption tests."""

import io
import tarfile

from blake3 import blake3


def write_contents(path, contents):
    contents = {
        **contents,
        "manifest.blake3": blake3(contents["manifest.json"]).hexdigest().encode(),
    }
    with tarfile.open(path, "w:", format=tarfile.GNU_FORMAT) as archive:
        for name, data in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
