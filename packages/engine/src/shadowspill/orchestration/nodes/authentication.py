"""Operator-issued node credentials, separate from client owner credentials."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shadowspill.server.authentication import BearerTokens
from shadowspill_client._records import text
from shadowspill_client.protocol.requests import identifier


@dataclass(frozen=True)
class NodeAccess:
    """The node a node credential authorizes, and the compute pool it joins.

    The service refuses a node daemon's join unless the daemon reports this
    node ID and pool.

    Parameters
    ----------
    node_id : str
        The node's ID: up to 128 letters, digits and ``_.:-``, starting with
        a letter or digit.
    pool : str, default "default"
        The compute pool the node joins, as clients name it in
        `shadowspill_client.Resources`. Nonempty, without surrounding spaces,
        at most 256 characters.

    Raises
    ------
    ValueError
        If ``node_id`` is not such an ID, or ``pool`` is empty, has
        surrounding spaces or is longer than 256 characters.
    """

    node_id: str
    pool: str = "default"

    def __post_init__(self) -> None:
        identifier(self.node_id, "authorized node ID")
        if len(text(self.pool, "authorized pool")) > 256:
            raise ValueError("pool name exceeds its bound")


class NodeTokens:
    """Node daemon credentials for a `Server`, each with the node it authorizes.

    A node daemon's request presents one in an ``Authorization: Bearer``
    header. Only the credentials' SHA-256 digests are kept. `Server` refuses
    a credential that is also a client credential.

    Parameters
    ----------
    tokens : mapping of str to NodeAccess
        Each credential, mapped to the node it authorizes; one credential per
        node ID. From 1 to 1024 entries. A credential is 32 to 512 visible
        ASCII characters.

    Raises
    ------
    TypeError
        If ``tokens`` is not a mapping, or a value is not a `NodeAccess`.
    ValueError
        If ``tokens`` has no entries or more than 1024, a credential is
        invalid, or two credentials name the same node ID.
    """

    def __init__(self, tokens: Mapping[str, NodeAccess]) -> None:
        if not isinstance(tokens, Mapping) or any(
            not isinstance(access, NodeAccess) for access in tokens.values()
        ):
            raise TypeError("node credentials require NodeAccess policies")
        self._tokens = BearerTokens(
            {token: value.node_id for token, value in tokens.items()}
        )
        self._policies = {value.node_id: value for value in tokens.values()}
        if len(self._policies) != len(tokens):
            raise ValueError("each node identity requires one credential")

    def authorize(self, headers: Sequence[str]) -> NodeAccess:
        """Return what a node daemon's request is authorized as.

        Parameters
        ----------
        headers : sequence of str
            The request's ``Authorization`` header values.

        Returns
        -------
        NodeAccess
            The policy of the credential presented.

        Raises
        ------
        PermissionError
            If there is not exactly one value, or it is not ``Bearer`` (in any
            case), a space and a configured node credential.
        """
        return self._policies[self._tokens.owner(headers)]

    def require_separate(self, clients: BearerTokens) -> None:
        """Check that no node credential is also a client credential.

        Parameters
        ----------
        clients : BearerTokens
            The client credentials.

        Raises
        ------
        ValueError
            If a credential is in both.
        """
        clients.require_separate(self._tokens)
