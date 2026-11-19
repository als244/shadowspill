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
    """Identity of a running service, as `ServiceClient.info`.

    Parameters
    ----------
    server_incarnation : str
        ID of this run of the service process, new each time it starts.
    owner_id : str
        The owner the connection's token authenticates as.
    """

    server_incarnation: str
    owner_id: str

    def __post_init__(self) -> None:
        identifier(self.server_incarnation, "server incarnation")
        identifier(self.owner_id, "authenticated owner ID")


@dataclass(frozen=True)
class ServiceError(Record):
    """Error record the service sends with an unsuccessful reply.

    It is a record, not an exception: `RemoteError` carries it, and a failed
    run's record holds one as its ``error``. It says nothing about the effects
    of a training operation.

    Parameters
    ----------
    code : str
        Error code, such as ``"not_found"``.
    message : str
        Description of the error, at most 1024 characters.
    """

    code: str
    message: str

    def __post_init__(self) -> None:
        identifier(self.code, "service error code")
        text(self.message, "service error message")
        if len(self.message) > 1024:
            raise ValueError("service error message exceeds its bound")
