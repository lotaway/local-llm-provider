import os
from skills import init_skills


def test_skills_env_loading(monkeypatch, tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    skill_dir = skills_dir / "demo-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("""---
name: demo-skill
description: Demo skill for env loading test
---
# Demo
""", encoding="utf-8")

    monkeypatch.setenv("LLP_SKILLS_DIRS", str(skills_dir))
    monkeypatch.setenv("LLP_ENABLE_CLAUDE_GLOBAL", "0")

    init_skills(force=True)
    from skills import registry

    skills = registry.list_skills()
    names = {s.name for s in skills}
    assert "demo-skill" in names
