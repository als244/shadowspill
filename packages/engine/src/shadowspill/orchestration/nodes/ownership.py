"""Agent identity/device locks retained by child supervisors across agent death."""

import fcntl
import hashlib
import os
import stat
from pathlib import Path


class NodeOwnership:
    def __init__(self) -> None:
        self._files: list[int] = []
        self._names: set[str] = set()

    @property
    def descriptors(self) -> tuple[int, ...]:
        return tuple(self._files)

    def acquire(self, *names: str) -> None:
        root = Path("/tmp") / f"shadowspill-agents-{os.getuid()}"
        root.mkdir(mode=0o700, exist_ok=True)
        info = root.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise PermissionError("node ownership directory is not private")
        for name in sorted(names):
            if name in self._names:
                raise ValueError("node ownership lock was requested twice")
            path = root / hashlib.sha256(name.encode()).hexdigest()
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                os.close(fd)
                raise RuntimeError(
                    "another agent or its children still own assigned resources"
                ) from None
            self._files.append(fd)
            self._names.add(name)

    def close(self) -> None:
        # Closing our copies must not explicitly LOCK_UN the shared open-file
        # descriptions: surviving supervisors retain them until cleanup ends.
        while self._files:
            os.close(self._files.pop())
        self._names.clear()
