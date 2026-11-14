"""Small service metadata; credentials stay in connection configuration."""

from dataclasses import dataclass

from .._records import Record, text
from .requests import identifier

CONTROL_MEDIA_TYPE = "application/vnd.shadowspill.control+json"


def bearer_token(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 32 <= len(value) <= 512
        or not value.isascii()
        or any(not 33 <= ord(char) <= 126 for char in value)
    ):
        raise ValueError("service token must contain 32-512 visible ASCII characters")
    return value


@dataclass(frozen=True)
class ServiceInfo(Record):
    server_incarnation: str
    owner_id: str

    def __post_init__(self) -> None:
        identifier(self.server_incarnation, "server incarnation")
        identifier(self.owner_id, "authenticated owner ID")


@dataclass(frozen=True)
class ServiceError(Record):
    """An HTTP failure, never evidence about a training operation's effects."""

    code: str
    message: str

    def __post_init__(self) -> None:
        identifier(self.code, "service error code")
        text(self.message, "service error message")
        if len(self.message) > 1024:
            raise ValueError("service error message exceeds its bound")
