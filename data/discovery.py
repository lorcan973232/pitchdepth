"""Locate SoccerNet-Depth frames and pair each colour image with its depth label.

Two folder layouts are accepted:

    <root>/{train,valid,test,challenge}/game_*/video_*/{color,depth_r}/
    <root>/depth-football/{Train,Validation,Test,Challenge}/game_*/video_*/{color,depth_r}/
"""

from __future__ import annotations

import os

# Folder names that can hold each split. Both layouts are searched.
SPLIT_FOLDER_NAMES = {
    "train": ("Train", "train"),
    "validation": ("Validation", "valid", "val"),
    "test": ("Test", "test"),
    "challenge": ("Challenge", "challenge"),
    "extra": ("Extra", "extra"),
}
SPLIT_SYNONYMS = {"valid": "validation", "val": "validation"}
# "Complete" means every labelled split, in this order.
LABELLED_SPLITS = ("train", "validation", "test", "extra")
SPORT_FOLDERS = ("depth-football", "depth-basketball")


def _folder_identity(path: str) -> str:
    # "Train" and "train" are the same folder on case-insensitive file systems.
    return os.path.normcase(os.path.realpath(path))


def resolve_split_dirs(base_dir: str, split: str) -> list[str]:
    """Return every existing folder that holds ``split``, each folder once."""
    if split.lower() == "complete":
        folders: list[str] = []
        for name in LABELLED_SPLITS:
            folders += [f for f in resolve_split_dirs(base_dir, name) if f not in folders]
        return folders

    key = SPLIT_SYNONYMS.get(split.lower(), split.lower())
    names = SPLIT_FOLDER_NAMES.get(key, (split,))
    folders, identities = [], set()
    for name in names:
        for parent in ("", *SPORT_FOLDERS):
            folder = os.path.join(base_dir, parent, name) if parent else os.path.join(base_dir, name)
            if os.path.isdir(folder) and _folder_identity(folder) not in identities:
                identities.add(_folder_identity(folder))
                folders.append(folder)
    return folders


def matches_sport(path: str, sport: str | None) -> bool:
    """Tell football and basketball frames apart by their path.

    The flat SN-Depth-2025 layout is football-only and has no sport folder, so
    anything outside a basketball folder counts as football.
    """
    if not sport:
        return True
    basketball = "basket" in str(path).replace("\\", "/").lower()
    return basketball if sport == "basket" else not basketball


def find_rgb_depth(base_dir: str, split: str = "Test", sport: str | None = None) -> list[tuple[str, str]]:
    """List (rgb_path, depth_path) for every frame of ``split`` that has a label.

    Labels live in a ``depth_r`` folder beside each ``color`` folder, under the
    same file name. Folders and files are visited in sorted order, so the list
    is the same on every platform.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for split_dir in resolve_split_dirs(base_dir, split):
        for folder, subfolders, files in os.walk(split_dir):
            subfolders.sort()
            if os.path.basename(folder) != "color":
                continue
            label_folder = os.path.join(os.path.dirname(folder), "depth_r")
            if not os.path.isdir(label_folder):
                continue
            for name in sorted(files):
                # "._" files are macOS metadata, not frames.
                if name.startswith("._") or not name.lower().endswith(".png"):
                    continue
                label = os.path.join(label_folder, name)
                pair = (os.path.join(folder, name), label)
                if pair not in seen and os.path.exists(label) and matches_sport(label, sport):
                    seen.add(pair)
                    pairs.append(pair)
    return pairs
