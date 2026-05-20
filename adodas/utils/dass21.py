"""DASS-21 subscale grouping. Mirrors backup/common/models/heads.py:8-16."""
from __future__ import annotations

# Index lists are 0-indexed into d01..d21 (so 2 means "d03").
# Standard DASS-21 subscale assignment:
#   Depression: items 3, 5, 10, 13, 16, 17, 21
#   Anxiety:    items 2, 4, 7,  9, 15, 19, 20
#   Stress:     items 1, 6, 8, 11, 12, 14, 18
DASS21_GROUP_ITEMS: dict[str, list[int]] = {
    "D": [2, 4, 9, 12, 15, 16, 20],
    "A": [1, 3, 6, 8, 14, 18, 19],
    "S": [0, 5, 7, 10, 11, 13, 17],
}

# Standard cutoffs applied to 2 * sum(items in subscale) to flag the group
# as elevated. These are the official DASS-21 binary cutoffs used by A1.
DASS21_GROUP_CUTOFFS: dict[str, float] = {"D": 10.0, "A": 8.0, "S": 15.0}
DASS21_GROUP_ORDER: tuple[str, ...] = ("D", "A", "S")

ITEM_COLS: list[str] = [f"d{i:02d}" for i in range(1, 22)]
A1_COLS: list[str] = ["y_D", "y_A", "y_S"]
SESSIONS: tuple[str, ...] = ("A01", "B01", "B02", "B03")


def group_of_item(item_idx: int) -> str:
    """Return 'D' / 'A' / 'S' for a 0-indexed item index in [0, 21)."""
    for g, indices in DASS21_GROUP_ITEMS.items():
        if item_idx in indices:
            return g
    raise ValueError(f"Item index {item_idx} not in any DASS-21 subscale")


def item_to_group_vector() -> list[int]:
    """Map each of 21 items to its subscale index (0=D, 1=A, 2=S)."""
    order = {g: i for i, g in enumerate(DASS21_GROUP_ORDER)}
    return [order[group_of_item(j)] for j in range(21)]
