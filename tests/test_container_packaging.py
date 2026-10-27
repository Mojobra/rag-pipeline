"""Guard the image build boundary against accidental local-data inclusion."""

from __future__ import annotations

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ContainerPackagingTests(unittest.TestCase):
    """Keep the local image's source context and runtime inputs explicit."""

    def test_build_context_allows_only_package_inputs(self) -> None:
        """Require an allowlist so private root files cannot enter the builder."""
        rules = [
            line.strip()
            for line in (PROJECT_ROOT / ".dockerignore")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.startswith("#")
        ]

        self.assertEqual(
            rules[:6],
            ["**", "!pyproject.toml", "!uv.lock", "!README.md", "!src", "!src/**"],
        )
        self.assertFalse(any(rule.startswith("!") for rule in rules[6:]))
        self.assertIn("**/.env*", rules)
        self.assertIn("**/*.egg-info", rules)
        self.assertIn("**/*.pem", rules)
        self.assertIn("**/*.key", rules)

    def test_runtime_copies_only_installed_environment(self) -> None:
        """Prevent a broad source copy or embedded credential in the image."""
        instructions = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        context_copies = [
            line.strip()
            for line in instructions.splitlines()
            if line.startswith(("COPY ", "ADD ")) and "--from=" not in line
        ]

        self.assertEqual(
            context_copies,
            ["COPY pyproject.toml uv.lock README.md ./", "COPY src/ ./src/"],
        )
        self.assertIn("COPY --from=builder /app/.venv /app/.venv", instructions)
        self.assertIn("USER rag", instructions)
        self.assertIn("--no-access-log", instructions)
        self.assertNotIn("RAG_API_BEARER_TOKEN", instructions)
