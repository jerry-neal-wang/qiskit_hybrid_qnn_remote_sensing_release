from src.models.qnn_head import get_qnn_dependency_report


def test_qnn_dependency_report_points_to_unified_requirements() -> None:
    report = get_qnn_dependency_report()
    commands = report.get("install_commands", [])
    assert isinstance(commands, list)
    if commands:
        assert any("requirements.txt" in command for command in commands)
