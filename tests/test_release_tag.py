"""Exercise release publication guards with isolated, offline Git histories."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

VERIFY_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_release_tag.py"


class ReleaseTagTests(unittest.TestCase):
    """Keep registry publication tied to a reviewed mainline version commit."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repository = Path(self.directory.name)
        self.git("init", "-b", "main")
        self.git("config", "user.name", "CI Test")
        self.git("config", "user.email", "ci-test@example.invalid")
        (self.repository / "pyproject.toml").write_text(
            '[project]\nname = "rag-pipeline"\nversion = "0.4.0"\n',
            encoding="utf-8",
        )
        self.git("add", "pyproject.toml")
        self.git("commit", "-m", "release")
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")
        self.git("tag", "-a", "v0.4.0", "-m", "Release 0.4.0")

    def git(self, *arguments: str) -> None:
        """Run a local Git command in the temporary release history."""
        subprocess.run(
            ["git", *arguments],
            cwd=self.repository,
            check=True,
            capture_output=True,
            text=True,
        )

    def verify(self, tag_name: str = "v0.4.0") -> subprocess.CompletedProcess[str]:
        """Execute the same guard used by CI without a remote service."""
        return subprocess.run(
            [sys.executable, str(VERIFY_SCRIPT), tag_name],
            cwd=self.repository,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_accepts_annotated_matching_mainline_tag(self) -> None:
        self.assertEqual(self.verify().returncode, 0)

    def test_accepts_older_mainline_commit(self) -> None:
        (self.repository / "later.txt").write_text("later", encoding="utf-8")
        self.git("add", "later.txt")
        self.git("commit", "-m", "later main commit")
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")
        self.git("switch", "--detach", "v0.4.0")
        self.assertEqual(self.verify().returncode, 0)

    def test_rejects_invalid_tag_name(self) -> None:
        self.assertIn("vMAJOR.MINOR.PATCH", self.verify("release-0.4.0").stderr)

    def test_rejects_lightweight_tag(self) -> None:
        self.git("tag", "-d", "v0.4.0")
        self.git("tag", "v0.4.0")
        self.assertIn("must be annotated", self.verify().stderr)

    def test_rejects_version_mismatch(self) -> None:
        self.assertIn("does not match project.version", self.verify("v0.4.1").stderr)

    def test_rejects_non_mainline_commit(self) -> None:
        self.git("tag", "-d", "v0.4.0")
        (self.repository / "feature.txt").write_text("feature", encoding="utf-8")
        self.git("add", "feature.txt")
        self.git("commit", "-m", "unmerged feature")
        self.git("tag", "-a", "v0.4.0", "-m", "Release 0.4.0")
        self.assertIn("not reachable from origin/main", self.verify().stderr)

    def test_rejects_checkout_different_from_tagged_commit(self) -> None:
        (self.repository / "later.txt").write_text("later", encoding="utf-8")
        self.git("add", "later.txt")
        self.git("commit", "-m", "later main commit")
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")
        self.assertIn("not the tagged commit", self.verify().stderr)

    def test_rejects_missing_mainline_ref(self) -> None:
        self.git("update-ref", "-d", "refs/remotes/origin/main")
        self.assertIn("refs/remotes/origin/main", self.verify().stderr)


if __name__ == "__main__":
    unittest.main()
