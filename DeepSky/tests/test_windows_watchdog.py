from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_windows_watchdog_is_installed_by_production_deploy() -> None:
    workflow = (REPO_ROOT / ".github/workflows/deploy-vps.yml").read_text(encoding="utf-8")
    installer = (REPO_ROOT / "deploy/windows/install_watchdog.ps1").read_text(encoding="utf-8")
    watchdog = (REPO_ROOT / "deploy/windows/watchdog.ps1").read_text(encoding="utf-8")

    assert "Install DeepSky self-healing watchdog" in workflow
    assert "install_watchdog.ps1" in workflow
    assert 'AppTaskName = "DeepSky Web App"' in installer
    assert "New-ScheduledTaskTrigger -AtStartup" in installer
    assert "New-TimeSpan -Minutes 2" in installer
    assert "Invoke-WebRequest -Uri $HealthUrl" in watchdog
    assert "Start-ScheduledTask -TaskName $AppTaskName" in watchdog
    assert "Recovered DeepSky successfully" in watchdog


def test_web_task_has_native_restart_policy() -> None:
    installer = (REPO_ROOT / "deploy/windows/install_web_task.ps1").read_text(encoding="utf-8")

    assert 'TaskName = "DeepSky Web App"' in installer
    assert "-StartWhenAvailable" in installer
    assert "-RestartCount 999" in installer
    assert "-RestartInterval (New-TimeSpan -Minutes 1)" in installer
