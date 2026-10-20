"""Validate a release tag against package metadata and the mainline history."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path


class ReleaseTagError(ValueError):
    """Report why a tag is not eligible for container publication."""


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], capture_output=True, check=False, text=True
    )
    if result.returncode != 0:
        raise ReleaseTagError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def verify_release_tag(tag_name: str) -> None:
    """Require an annotated version tag on the checked-out mainline commit.

    Reads ``pyproject.toml`` and local Git refs; it never fetches or modifies
    them. CI must check out full history, including ``origin/main`` and tags.
    Raises ``ReleaseTagError`` before image publication on any mismatch.
    """
    if re.fullmatch(r"v\d+\.\d+\.\d+", tag_name) is None:
        raise ReleaseTagError("Release tags must have the form vMAJOR.MINOR.PATCH.")

    with Path("pyproject.toml").open("rb") as metadata_file:
        version = tomllib.load(metadata_file)["project"]["version"]
    if tag_name != f"v{version}":
        raise ReleaseTagError(
            f"Tag {tag_name} does not match project.version {version}."
        )

    tag_ref = f"refs/tags/{tag_name}"
    if _git("cat-file", "-t", tag_ref) != "tag":
        raise ReleaseTagError("The release tag must be annotated.")

    tagged_commit = _git("rev-parse", f"{tag_ref}^{{commit}}")
    if tagged_commit != _git("rev-parse", "HEAD"):
        raise ReleaseTagError("The checkout is not the tagged commit.")

    mainline = _git("rev-parse", "--verify", "refs/remotes/origin/main")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", tagged_commit, mainline],
        capture_output=True,
        check=False,
        text=True,
    )
    if ancestor.returncode != 0:
        raise ReleaseTagError("The tagged commit is not reachable from origin/main.")


def main() -> int:
    """Return a nonzero status with an actionable release validation error."""
    if len(sys.argv) != 2:
        print("Usage: verify_release_tag.py vMAJOR.MINOR.PATCH", file=sys.stderr)
        return 2
    try:
        verify_release_tag(sys.argv[1])
    except ReleaseTagError as exc:
        print(f"Release tag verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified release tag {sys.argv[1]} on origin/main.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
