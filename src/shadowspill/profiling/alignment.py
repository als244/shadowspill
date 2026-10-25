"""Bounded sequence alignment for observed task allocation lifetimes."""

from collections.abc import Sequence


def _myers_edit_distance(
    left: Sequence[tuple[object, ...]],
    right: Sequence[tuple[object, ...]],
) -> int:
    """Return insertion/deletion distance in O((N + M)D) time."""

    if left == right:
        return 0
    prefix = 0
    shared = min(len(left), len(right))
    while prefix < shared and left[prefix] == right[prefix]:
        prefix += 1
    left_end = len(left)
    right_end = len(right)
    while (
        left_end > prefix
        and right_end > prefix
        and left[left_end - 1] == right[right_end - 1]
    ):
        left_end -= 1
        right_end -= 1
    left = left[prefix:left_end]
    right = right[prefix:right_end]
    left_count = len(left)
    right_count = len(right)
    if left_count == 0 or right_count == 0:
        return left_count + right_count
    # Profiling paths are normally identical or differ by one optional
    # allocation lifetime.  Recognize that overwhelmingly common case with
    # one linear scan instead of constructing a Myers frontier.
    if abs(left_count - right_count) == 1 and _is_single_insertion(
        left,
        right,
    ):
        return 1
    frontier: dict[int, int] = {1: 0}
    for edits in range(left_count + right_count + 1):
        next_frontier: dict[int, int] = {}
        for diagonal in range(-edits, edits + 1, 2):
            if diagonal == -edits or (
                diagonal != edits
                and frontier.get(diagonal - 1, -1) < frontier.get(diagonal + 1, -1)
            ):
                x = frontier.get(diagonal + 1, 0)
            else:
                x = frontier.get(diagonal - 1, 0) + 1
            y = x - diagonal
            while x < left_count and y < right_count and left[x] == right[y]:
                x += 1
                y += 1
            next_frontier[diagonal] = x
            if x >= left_count and y >= right_count:
                return edits
        frontier = next_frontier
    raise AssertionError("edit-distance frontier did not terminate")


def _is_single_insertion(
    left: Sequence[tuple[object, ...]],
    right: Sequence[tuple[object, ...]],
) -> bool:
    """Return whether the longer sequence inserts exactly one token."""

    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    if len(longer) != len(shorter) + 1:
        return False
    mismatch = 0
    while mismatch < len(shorter) and shorter[mismatch] == longer[mismatch]:
        mismatch += 1
    return shorter[mismatch:] == longer[mismatch + 1 :]


def _unique_minimum_alignment(
    left: Sequence[tuple[object, ...]],
    right: Sequence[tuple[object, ...]],
    distance: int,
) -> tuple[tuple[int, int], ...] | None:
    """Return the unique minimum-edit token mapping, if one exists.

    Forward and backward banded edit costs identify every equal-token pair
    that can participate in a minimum insertion/deletion alignment. If one
    token can pair with two positions, the physical invariant interpretation is
    ambiguous even when either choice would happen to execute successfully.
    """

    if distance == 0:
        return tuple((index, index) for index in range(len(left)))
    left_count = len(left)
    right_count = len(right)
    cells = (left_count + right_count + 2) * (2 * distance + 1)
    if cells > 20_000_000:
        raise ValueError(
            "allocation paths diverge too far for bounded invariant alignment: "
            f"left={left_count}, right={right_count}, distance={distance}"
        )
    forward: list[dict[int, int]] = [dict() for _ in range(left_count + 1)]
    for j in range(min(right_count, distance) + 1):
        forward[0][j] = j
    for i in range(1, left_count + 1):
        row = forward[i]
        previous = forward[i - 1]
        for j in range(max(0, i - distance), min(right_count, i + distance) + 1):
            candidates: list[int] = []
            if j in previous:
                candidates.append(previous[j] + 1)
            if j > 0 and j - 1 in row:
                candidates.append(row[j - 1] + 1)
            if j > 0 and left[i - 1] == right[j - 1] and j - 1 in previous:
                candidates.append(previous[j - 1])
            if candidates:
                value = min(candidates)
                if value <= distance:
                    row[j] = value

    backward: list[dict[int, int]] = [dict() for _ in range(left_count + 1)]
    for j in range(max(0, right_count - distance), right_count + 1):
        backward[left_count][j] = right_count - j
    for i in range(left_count - 1, -1, -1):
        row = backward[i]
        following = backward[i + 1]
        remaining_left = left_count - i
        center = right_count - remaining_left
        for j in range(
            min(right_count, center + distance),
            max(0, center - distance) - 1,
            -1,
        ):
            candidates = []
            if j in following:
                candidates.append(following[j] + 1)
            if j < right_count and j + 1 in row:
                candidates.append(row[j + 1] + 1)
            if j < right_count and left[i] == right[j] and j + 1 in following:
                candidates.append(following[j + 1])
            if candidates:
                value = min(candidates)
                if value <= distance:
                    row[j] = value

    left_matches: dict[int, int] = {}
    right_matches: dict[int, int] = {}
    possible_pairs: list[tuple[int, int]] = []
    for i, token in enumerate(left):
        for j in range(max(0, i - distance), min(right_count, i + distance + 1)):
            if token != right[j]:
                continue
            prefix = forward[i].get(j)
            suffix = backward[i + 1].get(j + 1)
            if prefix is None or suffix is None or prefix + suffix != distance:
                continue
            prior_right = left_matches.setdefault(i, j)
            prior_left = right_matches.setdefault(j, i)
            if prior_right != j or prior_left != i:
                return None
            possible_pairs.append((i, j))
    expected_matches_numerator = left_count + right_count - distance
    if expected_matches_numerator < 0 or expected_matches_numerator % 2 != 0:
        return None
    expected_matches = expected_matches_numerator // 2
    if len(possible_pairs) != expected_matches:
        return None
    previous_right = -1
    for _left_index, right_index in possible_pairs:
        if right_index <= previous_right:
            return None
        previous_right = right_index
    return tuple(possible_pairs)
