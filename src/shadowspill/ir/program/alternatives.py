"""The choices a program offers: one option per way of running a group."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.schema import artifact_schema

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

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "active_task_ids": string_list(self.active_task_ids),
            "option_id": self.option_id,
            "retained_alias_group_ids": string_list(self.retained_alias_group_ids),
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskAlternativeOption:
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

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "group_id": self.group_id,
            "options": [option.to_dict() for option in self.options],
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskAlternativeGroup:
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
    group_id: str
    option_id: str

    def __post_init__(self) -> None:
        require_identifier(self.group_id, "selection.group_id")
        require_identifier(self.option_id, "selection.option_id")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"group_id": self.group_id, "option_id": self.option_id}

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskAlternativeChoice:
        data = expect_mapping(value, path)
        return cls(
            group_id=expect_string(field(data, "group_id", path), f"{path}.group_id"),
            option_id=expect_string(
                field(data, "option_id", path), f"{path}.option_id"
            ),
        )
