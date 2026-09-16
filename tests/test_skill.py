"""Keep the agent skill (.agents/skills/cuban, mirrored into .claude/skills)
in sync with the CLI: the frontmatter must parse and every --flag the skill
mentions must exist in the argparse parser, so the skill cannot silently go
stale when a flag is renamed or removed."""

import re
from pathlib import Path

from cuban.cli import _build_parser

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = REPO_ROOT / ".agents" / "skills" / "cuban"
SKILL_MD = SKILL_DIR / "SKILL.md"
CLAUDE_LINK = REPO_ROOT / ".claude" / "skills" / "cuban"


def _frontmatter(text):
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert match, "SKILL.md must start with a YAML frontmatter block"
    fields = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        assert sep, f"malformed frontmatter line: {line!r}"
        fields[key.strip()] = value.strip()
    return fields


def test_frontmatter_has_name_and_description():
    fields = _frontmatter(SKILL_MD.read_text())
    assert fields["name"] == "cuban"
    # Both Claude Code and Codex cap the description at 1024 characters.
    assert 0 < len(fields["description"]) <= 1024


def test_every_flag_in_skill_exists_in_cli():
    parser_flags = {opt for action in _build_parser()._actions for opt in action.option_strings}
    text = "\n".join(p.read_text() for p in SKILL_DIR.rglob("*.md"))
    # Flags from bcftools examples are not cuban's; ignore lines mentioning it.
    lines = [line for line in text.splitlines() if "bcftools" not in line]
    mentioned = set(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", "\n".join(lines)))
    unknown = sorted(mentioned - parser_flags)
    assert not unknown, f"SKILL.md mentions flags the CLI does not have: {unknown}"


def test_claude_skill_mirrors_agents_skill():
    assert CLAUDE_LINK.is_symlink(), ".claude/skills/cuban must be a symlink"
    assert CLAUDE_LINK.resolve() == SKILL_DIR.resolve()
    assert (CLAUDE_LINK / "SKILL.md").is_file()
