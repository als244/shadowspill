"""The choices a program offers: one option per way of running a group."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.schema import artifact_schema
from shadowspill.serialization.sharing import shared_writer

from ..serialization import (
    JsonValue,
    string_list,
)
from ..validation import (
    expect_list,
    expect_mapping,
    expect_string,
    field,
    index_unique,
    require,
    require_identifier,
    require_tuple,
)

PROGRAM_SCHEMA = artifact_schema("program")


@dataclass(frozen=True, slots=True)
class TaskAlternativeOption:
    """One way of running a group: the tasks it activates and the storage it keeps.

    Parameters
    ----------
    option_id : str
        The option's identity, unique in its group.
    active_task_ids : tuple of str
        Tasks that run when the option is taken.
    retained_alias_group_ids : tuple of str, default ()
        Alias groups the option keeps across its alternative group's boundary,
        none of them shared; their sizes are its memory cost.

    Raises
    ------
    ValidationError
        If an identity is empty or padded with whitespace, or a collection is
        not a tuple or repeats an entry.
    """

    option_id: str
    active_task_ids: tuple[str, ...]
    retained_alias_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.option_id, "task_alternative_option.option_id")
        require_tuple(self.active_task_ids, "task_alternative_option.active_task_ids")
        require_tuple(
            self.retained_alias_group_ids,
            "task_alternative_option.retained_alias_group_ids",
        )
        index_unique(self.active_task_ids, "task_alternative_option.active_task_ids")
        index_unique(
            self.retained_alias_group_ids,
            "task_alternative_option.retained_alias_group_ids",
        )

    @shared_writer
    def to_dict(self) -> dict[str, JsonValue]:
        """Return the record as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "active_task_ids": string_list(self.active_task_ids),
            "option_id": self.option_id,
            "retained_alias_group_ids": string_list(self.retained_alias_group_ids),
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskAlternativeOption:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"program.task_alternative_groups[0].options[1]"``.
            Error paths start with it.

        Returns
        -------
        TaskAlternativeOption
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
        data = expect_mapping(value, path)
        active = expect_list(
            field(data, "active_task_ids", path), f"{path}.active_task_ids"
        )
        retained = expect_list(
            field(data, "retained_alias_group_ids", path),
            f"{path}.retained_alias_group_ids",
        )
        return cls(
            option_id=expect_string(
                field(data, "option_id", path), f"{path}.option_id"
            ),
            active_task_ids=tuple(
                expect_string(item, f"{path}.active_task_ids[{index}]")
                for index, item in enumerate(active)
            ),
            retained_alias_group_ids=tuple(
                expect_string(item, f"{path}.retained_alias_group_ids[{index}]")
                for index, item in enumerate(retained)
            ),
        )


@dataclass(frozen=True, slots=True)
class TaskAlternativeGroup:
    """A choice between options, exactly one of which each plan takes.

    A task belongs to at most one group, and a task in no group always runs.

    Parameters
    ----------
    group_id : str
        The group's identity, unique in the program.
    options : tuple of TaskAlternativeOption
        The options; at least one, with distinct IDs.

    Raises
    ------
    ValidationError
        If ``group_id`` is empty or padded with whitespace, or ``options`` is not
        a tuple, is empty or repeats an ID.
    """

    group_id: str
    options: tuple[TaskAlternativeOption, ...]

    def __post_init__(self) -> None:
        require_identifier(self.group_id, "task_alternative_group.group_id")
        require_tuple(self.options, "task_alternative_group.options")
        require(
            bool(self.options), "task_alternative_group.options", "must not be empty"
        )
        index_unique(
            (option.option_id for option in self.options),
            "task_alternative_group.options",
        )

    @shared_writer
    def to_dict(self) -> dict[str, JsonValue]:
        """Return the record as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "group_id": self.group_id,
            "options": [option.to_dict() for option in self.options],
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskAlternativeGroup:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"program.task_alternative_groups[0]"``.
            Error paths start with it.

        Returns
        -------
        TaskAlternativeGroup
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
        data = expect_mapping(value, path)
        options = expect_list(field(data, "options", path), f"{path}.options")
        return cls(
            group_id=expect_string(field(data, "group_id", path), f"{path}.group_id"),
            options=tuple(
                TaskAlternativeOption.from_value(item, f"{path}.options[{index}]")
                for index, item in enumerate(options)
            ),
        )


@dataclass(frozen=True, slots=True)
class TaskAlternativeChoice:
    """The option one plan takes for one group.

    Parameters
    ----------
    group_id : str
        The group, by `TaskAlternativeGroup.group_id`.
    option_id : str
        The option taken, by `TaskAlternativeOption.option_id`.

    Raises
    ------
    ValidationError
        If an identity is empty or padded with whitespace.
    """

    group_id: str
    option_id: str

    def __post_init__(self) -> None:
        require_identifier(self.group_id, "selection.group_id")
        require_identifier(self.option_id, "selection.option_id")

    @shared_writer
    def to_dict(self) -> dict[str, JsonValue]:
        """Return the record as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {"group_id": self.group_id, "option_id": self.option_id}

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskAlternativeChoice:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"plan.selections[0]"``.
            Error paths start with it.

        Returns
        -------
        TaskAlternativeChoice
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
        data = expect_mapping(value, path)
        return cls(
            group_id=expect_string(field(data, "group_id", path), f"{path}.group_id"),
            option_id=expect_string(
                field(data, "option_id", path), f"{path}.option_id"
            ),
        )
