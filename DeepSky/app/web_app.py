from __future__ import annotations

import os
import json
import logging
import urllib.error
import urllib.request
import urllib.parse
import shutil
import tempfile
import time
import uuid
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .input_analysis import analyze_input_stretch
from .image_io import SUPPORTED_INPUTS, make_preview
from .pipeline import PipelineMode, run_pipeline
from .settings import APP_ROOT, PROJECT_ROOT, default_settings, load_settings


logger = logging.getLogger(__name__)


WEB_WORK_ROOT = Path(tempfile.gettempdir()) / "deepsky_web"
UPLOAD_ROOT = WEB_WORK_ROOT / "uploads"
PREVIEW_ROOT = WEB_WORK_ROOT / "previews"
STAGED_UPLOAD_ROOT = WEB_WORK_ROOT / "staged_uploads"
JOB_OUTPUT_ROOT = WEB_WORK_ROOT / "jobs"
LEGACY_WEB_UPLOAD_ROOT = PROJECT_ROOT / "outputs" / "web_uploads"
TEMP_FILE_TTL_SECONDS = 6 * 60 * 60
MAX_WORKERS = 1
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
MAX_UPLOAD_MB = MAX_UPLOAD_BYTES // (1024 * 1024)
CHUNK_UPLOAD_BYTES = 8 * 1024 * 1024
FREE_IMAGE_CREDITS = 5
PAID_PLAN_LABEL = "$15/month"
PAID_SUBSCRIPTION_STATUSES = {"active", "trialing"}


@dataclass
class WebJob:
    id: str
    user_id: str
    user_email: str | None = None
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    log: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    result: dict[str, Path] | None = None
    error: str | None = None
    stage: str = "Queued"
    progress: int = 0
    credit_consumed: bool = False


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str | None = None


@dataclass
class StagedUpload:
    id: str
    user_id: str
    filename: str
    size: int
    path: Path
    created_at: float = field(default_factory=time.time)


PROGRESS_LINE_RE = re.compile(r"progress:\s*(?P<label>.*?),\s*(?P<percent>\d+(?:\.\d+)?)%")


def _mapped_stage_progress(stage: str, stage_percent: float) -> int:
    spans = {
        "Siril Color Calibration": (22, 42),
        "Siril Deconvolution": (42, 47),
        "DeepSNR Denoising": (48, 68),
        "StarNet Star Separation": (72, 90),
    }
    start, end = spans.get(stage, (0, 100))
    return round(start + ((end - start) * max(0.0, min(100.0, stage_percent)) / 100.0))


def _progress_from_log_message(job: WebJob, message: str) -> tuple[str, int]:
    stage = job.stage
    progress = job.progress
    progress_match = PROGRESS_LINE_RE.search(message)
    if progress_match:
        label = progress_match.group("label").strip() or stage
        stage_percent = float(progress_match.group("percent"))
        if "NL-Bayes" in label:
            stage = "Siril Color Calibration"
        progress = max(progress, _mapped_stage_progress(stage, stage_percent))
        return stage, progress

    markers: tuple[tuple[str, str, int], ...] = (
        ("DeepSky job:", "Preparing Input", 3),
        ("Copied original:", "Preparing Input", 5),
        ("Input stretch analysis", "Analyzing Input", 8),
        ("Creating 16-bit working TIFF.", "Creating Working TIFF", 12),
        ("Pre-stretched input mode enabled", "Pre-Stretched Finish", 20),
        ("Applying pre-stretched object finish", "Pre-Stretched Finish", 28),
        ("Applying local astrophotography stretch", "Stretching Image", 24),
        ("Applying gentle-stretch", "Applying Object Finish", 32),
        ("Siril executable:", "Siril Color Calibration", 22),
        ("Running Siril:", "Siril Color Calibration", 26),
        ("Siril color calibration succeeded.", "Siril Color Calibration", 42),
        ("Siril Richardson-Lucy deconvolution test enabled", "Siril Deconvolution", 42),
        ("Siril deconvolution layer script:", "Siril Deconvolution", 43),
        ("Running command: rl", "Siril Deconvolution", 43),
        ("Richardson-Lucy deconvolution", "Siril Deconvolution", 44),
        ("Blended Siril deconvolution as galaxy-only detail layer", "Siril Deconvolution", 47),
        ("Star reduction enabled", "Star Reduction", 86),
        ("Slight Star Reduction enabled", "Star Reduction", 86),
        ("starless_test.tif:", "Star Reduction", 90),
        ("starless_test_stars.tif:", "Star Reduction", 92),
        ("DeepSNR executable:", "DeepSNR Denoising", 48),
        ("DeepSNR background cleanup executable:", "DeepSNR Background Cleanup", 36),
        ("denoised.tif:", "DeepSNR Denoising", 68),
        ("StarNet executable:", "StarNet Star Separation", 72),
        ("starless.tif:", "StarNet Star Separation", 90),
        ("stars.tif:", "Recombining Stars", 93),
        ("final.tif:", "Creating Final Image", 95),
        ("Final image:", "Creating Preview", 98),
        ("Done.", "Complete", 100),
    )
    for needle, next_stage, next_progress in markers:
        if needle in message:
            stage = next_stage
            progress = max(progress, next_progress)
            break
    return stage, progress


app = FastAPI(title="DeepSky", docs_url=None, redoc_url=None)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
jobs: dict[str, WebJob] = {}
previews: dict[str, str] = {}
staged_uploads: dict[str, StagedUpload] = {}
jobs_lock = Lock()
app.mount("/static", StaticFiles(directory=APP_ROOT / "app" / "static"), name="static")


def _supabase_url() -> str:
    return os.getenv("SUPABASE_URL", "").strip().rstrip("/")


def _supabase_anon_key() -> str:
    return os.getenv("SUPABASE_ANON_KEY", "").strip()


def _supabase_service_role_key() -> str:
    return os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()


def _auth_configured() -> bool:
    return bool(_supabase_url() and _supabase_anon_key())


def _stripe_secret_key() -> str:
    return os.getenv("STRIPE_SECRET_KEY", "").strip()


def _stripe_webhook_secret() -> str:
    return os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()


def _stripe_price_id() -> str:
    return os.getenv("STRIPE_PRICE_ID", "").strip()


def _billing_configured() -> bool:
    return bool(_supabase_url() and _supabase_service_role_key() and _stripe_secret_key() and _stripe_price_id())


def _stripe_module():
    try:
        import stripe  # type: ignore
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Stripe dependency is not installed.") from exc
    stripe.api_key = _stripe_secret_key()
    return stripe


def _supabase_rest_request(
    path: str,
    *,
    method: str = "GET",
    payload: Any | None = None,
    prefer: str | None = None,
) -> Any:
    if not _supabase_url() or not _supabase_service_role_key():
        raise HTTPException(status_code=503, detail="Supabase service role is not configured.")
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "apikey": _supabase_service_role_key(),
        "Authorization": f"Bearer {_supabase_service_role_key()}",
        "Accept": "application/json",
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if prefer:
        headers["Prefer"] = prefer
    request = urllib.request.Request(
        f"{_supabase_url()}/rest/v1/{path.lstrip('/')}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            content = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(status_code=502, detail=f"Supabase billing request failed: {detail or exc.reason}") from exc
    except OSError as exc:
        raise HTTPException(status_code=502, detail="Could not reach Supabase billing store.") from exc
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Supabase billing response was not JSON.") from exc


def _profile_select_filter(column: str, value: str) -> str:
    return f"profiles?{column}=eq.{urllib.parse.quote(value, safe='')}&select=*"


def _profile_period_is_current(profile: dict[str, Any] | None) -> bool:
    period_end = (profile or {}).get("current_period_end")
    if not period_end:
        return True
    try:
        if isinstance(period_end, str):
            parsed = datetime.fromisoformat(period_end.replace("Z", "+00:00"))
        elif isinstance(period_end, datetime):
            parsed = period_end
        else:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed > datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return False


def _is_paid_profile(profile: dict[str, Any] | None) -> bool:
    status = str((profile or {}).get("subscription_status") or "").lower()
    return status in PAID_SUBSCRIPTION_STATUSES and _profile_period_is_current(profile)


def _object_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _get_profile(user_id: str) -> dict[str, Any] | None:
    rows = _supabase_rest_request(_profile_select_filter("user_id", user_id))
    if isinstance(rows, list) and rows:
        return rows[0]
    return None


def _get_profile_by_customer(customer_id: str) -> dict[str, Any] | None:
    rows = _supabase_rest_request(_profile_select_filter("stripe_customer_id", customer_id))
    if isinstance(rows, list) and rows:
        return rows[0]
    return None


def _upsert_profile(user: AuthUser, **updates: Any) -> dict[str, Any]:
    payload = {
        "user_id": user.id,
        "email": user.email,
        **{key: value for key, value in updates.items() if value is not None},
    }
    rows = _supabase_rest_request(
        "profiles?on_conflict=user_id",
        method="POST",
        payload=payload,
        prefer="resolution=merge-duplicates,return=representation",
    )
    if isinstance(rows, list) and rows:
        return rows[0]
    profile = _get_profile(user.id)
    if not profile:
        raise HTTPException(status_code=502, detail="Could not create billing profile.")
    return profile


def _update_profile(user_id: str, updates: dict[str, Any]) -> None:
    _supabase_rest_request(
        f"profiles?user_id=eq.{urllib.parse.quote(user_id, safe='')}",
        method="PATCH",
        payload=updates,
        prefer="return=minimal",
    )


def _billing_profile_for(user: AuthUser) -> dict[str, Any]:
    if not _billing_configured():
        raise HTTPException(status_code=503, detail="Billing is not configured.")
    return _upsert_profile(user)


def _consume_credit_or_require_subscription(user: AuthUser) -> tuple[dict[str, Any], bool]:
    profile = _billing_profile_for(user)
    profile = _reconcile_stripe_subscription(user, profile)
    if _is_paid_profile(profile):
        return profile, False
    remaining = int(profile.get("free_credits_remaining") or 0)
    if remaining <= 0:
        raise HTTPException(
            status_code=402,
            detail="Free image credits used. Upgrade to the $15/month plan for unlimited processing.",
        )
    consumed = _supabase_rest_request(
        "rpc/consume_free_credit",
        method="POST",
        payload={"target_user_id": user.id},
    )
    if consumed is not True:
        raise HTTPException(
            status_code=402,
            detail="Free image credits used. Upgrade to the $15/month plan for unlimited processing.",
        )
    return _get_profile(user.id) or profile, True


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Invalid authorization header.")
    return token.strip()


def require_user(authorization: str | None = Header(default=None)) -> AuthUser:
    if not _auth_configured():
        raise HTTPException(status_code=503, detail="Supabase auth is not configured.")

    token = _extract_bearer_token(authorization)
    request = urllib.request.Request(
        f"{_supabase_url()}/auth/v1/user",
        headers={
            "apikey": _supabase_anon_key(),
            "Authorization": f"Bearer {token}",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise HTTPException(status_code=401, detail="Sign in to continue.") from exc
        raise HTTPException(status_code=502, detail="Could not verify Supabase session.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail="Could not verify Supabase session.") from exc

    user_id = payload.get("id")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    email = payload.get("email") if isinstance(payload.get("email"), str) else None
    return AuthUser(id=user_id, email=email)


def _cleanup_old_temp_files() -> None:
    cutoff = time.time() - TEMP_FILE_TTL_SECONDS
    if LEGACY_WEB_UPLOAD_ROOT.exists():
        shutil.rmtree(LEGACY_WEB_UPLOAD_ROOT, ignore_errors=True)
    for root in (UPLOAD_ROOT, PREVIEW_ROOT, STAGED_UPLOAD_ROOT, JOB_OUTPUT_ROOT):
        if not root.exists():
            continue
        for child in root.iterdir():
            try:
                if child.stat().st_mtime < cutoff:
                    if root == PREVIEW_ROOT:
                        with jobs_lock:
                            previews.pop(child.name, None)
                    if root == STAGED_UPLOAD_ROOT:
                        with jobs_lock:
                            staged_uploads.pop(child.name, None)
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
            except OSError:
                continue


def _delete_job_files(job: WebJob) -> None:
    paths: list[Path] = []
    if job.result and "job_folder" in job.result:
        paths.append(Path(job.result["job_folder"]))
    paths.append(UPLOAD_ROOT / job.id)
    for path in paths:
        try:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
        except OSError:
            continue


def _get_staged_upload(upload_id: str, user: AuthUser) -> StagedUpload:
    if not upload_id or any(ch not in "0123456789abcdef" for ch in upload_id):
        raise HTTPException(status_code=404, detail="Upload not found.")
    with jobs_lock:
        staged = staged_uploads.get(upload_id)
    if not staged or staged.user_id != user.id:
        raise HTTPException(status_code=404, detail="Upload not found.")
    if not staged.path.exists():
        with jobs_lock:
            staged_uploads.pop(upload_id, None)
        raise HTTPException(status_code=404, detail="Upload not found.")
    return staged


def _require_completed_staged_upload(upload_id: str, user: AuthUser) -> StagedUpload:
    staged = _get_staged_upload(upload_id, user)
    if staged.path.stat().st_size != staged.size:
        raise HTTPException(status_code=409, detail="Upload is not complete.")
    return staged


def _discard_staged_upload(upload_id: str) -> None:
    with jobs_lock:
        staged = staged_uploads.pop(upload_id, None)
    if staged:
        shutil.rmtree(staged.path.parent, ignore_errors=True)


def _docs_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DeepSky Docs</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #060a12;
      --panel: #0b121f;
      --line: #20304c;
      --text: #f7fbff;
      --muted: #91a6ca;
      --blue: #5c8dff;
      --green: #67e8c9;
      --amber: #ffd166;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 50% -12%, rgba(92, 141, 255, .22), transparent 380px),
        linear-gradient(180deg, rgba(11,18,31,.55), rgba(6,10,18,0) 280px),
        var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    a { color: inherit; }
    .wrap { width: min(1120px, calc(100vw - 40px)); margin: 0 auto; }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(6, 10, 18, .82);
      border-bottom: 1px solid rgba(32, 48, 76, .65);
      backdrop-filter: blur(14px);
    }
    .nav { min-height: 72px; display: flex; align-items: center; justify-content: space-between; gap: 18px; }
    .brand { font-weight: 900; font-size: 20px; text-decoration: none; }
    .brand span {
      background: linear-gradient(90deg, var(--blue), #a98cff, #ff806d);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }
    .navlinks { display: flex; gap: 22px; align-items: center; color: var(--muted); font-weight: 800; font-size: 14px; }
    .navlinks a { text-decoration: none; }
    .button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
      padding: 0 20px;
      border-radius: 10px;
      background: #2f6fe5;
      color: white;
      text-decoration: none;
      font-weight: 900;
      border: 1px solid rgba(126, 164, 255, .38);
    }
    main { padding: 58px 0 78px; }
    .hero { max-width: 820px; margin-bottom: 34px; }
    .eyebrow {
      color: #9fc0ff;
      font: 13px Consolas, ui-monospace, monospace;
      margin-bottom: 14px;
    }
    h1 { margin: 0; font-size: clamp(42px, 5.4vw, 74px); line-height: .98; letter-spacing: 0; }
    .lead { margin: 20px 0 0; color: var(--muted); font-size: 19px; line-height: 1.6; }
    .grid { display: grid; grid-template-columns: 280px 1fr; gap: 26px; align-items: start; }
    .toc {
      position: sticky;
      top: 96px;
      border: 1px solid var(--line);
      background: rgba(11,18,31,.78);
      border-radius: 8px;
      padding: 18px;
    }
    .toc strong { display: block; margin-bottom: 12px; }
    .toc a { display: block; color: var(--muted); text-decoration: none; padding: 8px 0; font-weight: 750; }
    .content { display: grid; gap: 18px; }
    section {
      border: 1px solid var(--line);
      background: rgba(11,18,31,.78);
      border-radius: 8px;
      padding: 24px;
    }
    h2 { margin: 0 0 14px; font-size: 28px; letter-spacing: 0; }
    h3 { margin: 22px 0 8px; font-size: 19px; }
    p { color: var(--muted); line-height: 1.62; margin: 0 0 12px; }
    ul { margin: 10px 0 0; padding-left: 20px; color: var(--muted); line-height: 1.65; }
    li { margin: 7px 0; }
    .callout {
      border: 1px solid rgba(255, 209, 102, .38);
      background: rgba(255, 209, 102, .08);
      color: #ffe3a0;
      border-radius: 8px;
      padding: 14px 16px;
      margin-top: 14px;
      line-height: 1.55;
    }
    .setting {
      border-top: 1px solid rgba(32,48,76,.72);
      padding-top: 16px;
      margin-top: 16px;
    }
    .setting:first-of-type { border-top: 0; padding-top: 0; }
    .label { color: var(--green); font-weight: 900; }
    .trouble { display: grid; gap: 12px; }
    details {
      border: 1px solid rgba(32,48,76,.8);
      border-radius: 8px;
      background: rgba(6,10,18,.42);
      padding: 16px 18px;
    }
    summary { cursor: pointer; font-weight: 900; font-size: 17px; }
    footer { border-top: 1px solid rgba(32,48,76,.65); color: #6f86aa; padding: 26px 0; text-align: center; }
    footer a { color: #9cbcff; text-decoration: none; font-weight: 900; }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
      .toc { position: static; }
      .navlinks a:not(.button) { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap nav">
      <a class="brand" href="/">DeepSky <span>Processor</span></a>
      <nav class="navlinks">
        <a href="/process">Process</a>
        <a href="/docs">Docs</a>
        <a class="button" href="/process">Process An Image</a>
      </nav>
    </div>
  </header>
  <main class="wrap">
    <div class="hero">
      <div class="eyebrow">DeepSky processing guide</div>
      <h1>Settings, fixes, and what to try when an image looks off.</h1>
      <p class="lead">Use this page as a quick field guide. Start with the object type, leave Input on Auto for most files, then adjust only when the result is too bright, too dark, too noisy, too soft, or missing detail.</p>
    </div>
    <div class="grid">
      <aside class="toc">
        <strong>On this page</strong>
        <a href="#quick-start">Quick Start</a>
        <a href="#settings">Settings</a>
        <a href="#troubleshooting">Troubleshooting</a>
        <a href="#downloads">Downloads</a>
        <a href="#support">Support</a>
      </aside>
      <div class="content">
        <section id="quick-start">
          <h2>Quick Start</h2>
          <p>For most uploads, use these defaults first:</p>
          <ul>
            <li><span class="label">Object:</span> choose Galaxy, Nebula, or Star Cluster based on the main target.</li>
            <li><span class="label">Input:</span> leave on Auto unless you know your file is already stretched.</li>
            <li><span class="label">Stretch:</span> leave on Standard for the first run.</li>
            <li><span class="label">Deconvolution:</span> leave it off first, then test it on galaxies if you want sharper arms and dust lanes.</li>
            <li><span class="label">Star Reduction:</span> DeepSky automatically reduces stars so gas, dust, and galaxy structure stand out.</li>
          </ul>
          <div class="callout">If your first result looks wrong, do not keep changing every setting at once. Change one setting, rerun, and compare.</div>
        </section>

        <section id="settings">
          <h2>What The Settings Mean</h2>
          <div class="setting">
            <h3>Object</h3>
            <p>This chooses the finishing style. Galaxy processing protects broadband color, dust lanes, cores, and spiral texture. Nebula processing focuses on gas and dust. Star Cluster processing keeps stars more natural and avoids heavy star removal.</p>
          </div>
          <div class="setting">
            <h3>Input</h3>
            <p><span class="label">Auto</span> lets DeepSky inspect the file histogram and choose the safest path. Use this first.</p>
            <p><span class="label">Linear</span> tells DeepSky the file is raw or mostly unstretched. Use this if Auto is too gentle or the preview looks very dark before processing.</p>
            <p><span class="label">Pre-stretched</span> tells DeepSky the image already has visible brightness/contrast. Use this if the result looks overexposed, washed out, or aggressively stretched.</p>
          </div>
          <div class="setting">
            <h3>Stretch</h3>
            <p><span class="label">Subtle</span> is safer for bright targets, already-bright files, star clusters, and cores that blow out easily.</p>
            <p><span class="label">Standard</span> is the normal first try.</p>
            <p><span class="label">Aggressive</span> is for very faint targets where the normal result is too dark or does not reveal enough gas or galaxy arms.</p>
          </div>
          <div class="setting">
            <h3>Deconvolution</h3>
            <p>This optional galaxy-focused detail pass can sharpen spiral structure, dust lanes, and galaxy texture. It can also exaggerate grain on noisy data, so compare an unchecked run against a checked run before deciding.</p>
          </div>
          <div class="setting">
            <h3>Star Reduction</h3>
            <p>DeepSky automatically applies slight star reduction. It keeps the brighter stars, reduces the busy faint star field, and lets gas, dust, and galaxy structure stand out without making the image feel empty.</p>
          </div>
        </section>

        <section id="troubleshooting">
          <h2>If The Image Looks Off</h2>
          <div class="trouble">
            <details open>
              <summary>Too bright, blown out, or over-stretched</summary>
              <ul>
                <li>Set <span class="label">Stretch</span> to Subtle.</li>
                <li>Try <span class="label">Input</span> as Pre-stretched if the file already looked bright before processing.</li>
                <li>For galaxies, keep the object set to Galaxy so the core is protected better.</li>
              </ul>
            </details>
            <details>
              <summary>Too dark or not enough faint detail</summary>
              <ul>
                <li>Try <span class="label">Stretch</span> as Aggressive.</li>
                <li>Make sure <span class="label">Input</span> is Auto or Linear, not Pre-stretched.</li>
                <li>For faint nebulae, use Object: Nebula.</li>
              </ul>
            </details>
            <details>
              <summary>Galaxy looks soft or like a blob</summary>
              <ul>
                <li>Use Object: Galaxy.</li>
                <li>Try deconvolution after you have a clean baseline, but turn it back off if the background gets grainy.</li>
                <li>Try Standard first, then Subtle if the core gets too bright.</li>
              </ul>
            </details>
            <details>
              <summary>Stars look weird, missing, or too reduced</summary>
              <ul>
                <li>For star clusters, use Object: Star Cluster to preserve natural stars.</li>
                <li>Use Object: Galaxy for broadband galaxy fields and Object: Nebula for gas/dust targets.</li>
                <li>If stars still look too strong, try a more subtle stretch so the field does not get over-amplified.</li>
              </ul>
            </details>
            <details>
              <summary>Background is green, noisy, or muddy</summary>
              <ul>
                <li>Leave Input on Auto first.</li>
                <li>Try Object: Nebula for emission targets and Object: Galaxy for broadband galaxy fields.</li>
                <li>If the image is already edited or exported from another app, try Pre-stretched.</li>
              </ul>
            </details>
            <details>
              <summary>Nebula gas disappears or looks too gray</summary>
              <ul>
                <li>Use Object: Nebula.</li>
                <li>Try Stretch: Aggressive if the target is faint.</li>
                <li>Try Stretch: Subtle if bright areas are washing out the dust structure.</li>
              </ul>
            </details>
            <details>
              <summary>Star cluster looks unnatural</summary>
              <ul>
                <li>Use Object: Star Cluster.</li>
                <li>Use Stretch: Subtle or Standard.</li>
                <li>Avoid galaxy settings unless the cluster is not the main target.</li>
              </ul>
            </details>
          </div>
        </section>

        <section id="downloads">
          <h2>Downloads</h2>
          <p><span class="label">PNG</span> is best for sharing online and checking the result quickly.</p>
          <p><span class="label">TIFF</span> is best if you want to continue editing in another program. It preserves more image data than PNG.</p>
        </section>

        <section id="support">
          <h2>When You Need Help</h2>
          <p>If the output still looks wrong, send us the processed image and the original file. The fastest way for us to improve DeepSky is to see the exact file and the exact result that failed.</p>
          <p><a class="button" href="https://www.facebook.com/deepskyprocessor/" target="_blank" rel="noreferrer">Message DeepSky Processor</a></p>
        </section>
      </div>
    </div>
  </main>
  <footer>
    <p><a href="/process">Process an image</a> &nbsp;|&nbsp; <a href="/">Home</a></p>
    <p>DeepSky Built By <a href="https://www.linkedin.com/in/diego-perry-64a609240/" target="_blank" rel="noreferrer">Diego Perry</a></p>
  </footer>
</body>
</html>"""


def _landing_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DeepSky Processor</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #060a12;
      --panel: #0b121f;
      --panel2: #101929;
      --line: #20304c;
      --text: #f7fbff;
      --muted: #91a6ca;
      --blue: #5c8dff;
      --violet: #a98cff;
      --coral: #ff806d;
      --green: #67e8c9;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 50% -10%, rgba(75, 117, 255, .24), transparent 360px),
        radial-gradient(circle at 14% 18%, rgba(255, 128, 109, .12), transparent 340px),
        var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    a { color: inherit; }
    .wrap { width: min(1160px, calc(100vw - 40px)); margin: 0 auto; }
    header {
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(6, 10, 18, .78);
      border-bottom: 1px solid rgba(32, 48, 76, .65);
      backdrop-filter: blur(14px);
    }
    .nav { min-height: 72px; display: flex; align-items: center; justify-content: space-between; gap: 18px; }
    .brand { font-weight: 900; letter-spacing: 0; font-size: 20px; }
    .brand span {
      background: linear-gradient(90deg, var(--blue), var(--violet), var(--coral));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }
    .navlinks { display: flex; gap: 22px; align-items: center; color: var(--muted); font-weight: 700; font-size: 14px; }
    .navlinks a { text-decoration: none; }
    .button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 48px;
      padding: 0 24px;
      border-radius: 10px;
      background: #2f6fe5;
      color: white;
      text-decoration: none;
      font-weight: 900;
      border: 1px solid rgba(126, 164, 255, .38);
      box-shadow: 0 14px 36px rgba(47, 111, 229, .24);
    }
    .hero {
      min-height: calc(100vh - 72px);
      display: grid;
      align-items: center;
      padding: 64px 0 42px;
    }
    .hero-grid { display: grid; grid-template-columns: 1.02fr .98fr; gap: 46px; align-items: center; }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 9px;
      color: #9fc0ff;
      border: 1px solid rgba(92, 141, 255, .42);
      background: rgba(17, 29, 54, .72);
      padding: 7px 13px;
      border-radius: 999px;
      font: 13px Consolas, ui-monospace, monospace;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--blue); box-shadow: 0 0 18px var(--blue); }
    h1 { margin: 18px 0 0; font-size: clamp(48px, 6.2vw, 86px); line-height: .95; letter-spacing: 0; }
    .gradient {
      background: linear-gradient(90deg, var(--blue), var(--violet), var(--coral));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }
    .lead { margin: 22px 0 0; color: var(--muted); font-size: clamp(18px, 2vw, 22px); line-height: 1.55; max-width: 650px; }
    .hero-actions { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 32px; align-items: center; }
    .secondary { color: #bfd2f5; text-decoration: none; font-weight: 800; }
    .trust { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 32px; max-width: 650px; }
    .trust div { border: 1px solid var(--line); border-radius: 12px; background: rgba(11, 18, 31, .72); padding: 14px; }
    .trust strong { display: block; font-size: 21px; }
    .trust span { color: var(--muted); font-size: 13px; }
    .hero-showcase { border: 1px solid var(--line); border-radius: 18px; background: rgba(10, 16, 28, .78); padding: 14px; box-shadow: 0 30px 80px rgba(0,0,0,.36); }
    .hero-showcase img { width: 100%; display: block; border-radius: 10px; border: 1px solid #1d2a42; background: #020409; }
    section { padding: 82px 0; }
    .section-head { display: grid; gap: 12px; max-width: 760px; margin-bottom: 28px; }
    h2 { margin: 0; font-size: clamp(34px, 4vw, 54px); line-height: 1.02; letter-spacing: 0; }
    .section-head p, .copy { color: var(--muted); font-size: 18px; line-height: 1.62; margin: 0; }
    .features { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    .feature { border: 1px solid var(--line); background: rgba(11, 18, 31, .78); border-radius: 8px; padding: 22px; }
    .feature h3 { margin: 0 0 10px; font-size: 20px; }
    .feature p { margin: 0; color: var(--muted); line-height: 1.55; }
    .comparison-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
    .slider {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
      padding: 14px;
    }
    .compare {
      position: relative;
      aspect-ratio: 4 / 3;
      overflow: hidden;
      border: 1px solid #1f2d43;
      border-radius: 8px;
      background: #020409;
      cursor: ew-resize;
      user-select: none;
      touch-action: none;
    }
    .compare img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; background: #020409; }
    .compare .after { clip-path: inset(0 0 0 var(--pos, 50%)); }
    .handle { position: absolute; top: 0; bottom: 0; left: var(--pos, 50%); width: 2px; background: white; box-shadow: 0 0 18px rgba(255,255,255,.65); }
    .handle::after {
      content: "";
      position: absolute;
      top: 50%;
      left: 50%;
      width: 34px;
      height: 34px;
      border-radius: 50%;
      transform: translate(-50%, -50%);
      background: #f7fbff;
      border: 4px solid #2f6fe5;
    }
    .slider input { width: 100%; margin-top: 12px; accent-color: var(--blue); }
    .steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; counter-reset: step; }
    .step { counter-increment: step; border: 1px solid var(--line); border-radius: 8px; padding: 20px; background: rgba(11,18,31,.75); }
    .step::before { content: counter(step); display: inline-grid; place-items: center; width: 30px; height: 30px; border-radius: 50%; background: #2f6fe5; font-weight: 900; margin-bottom: 14px; }
    .step h3 { margin: 0 0 8px; }
    .step p { margin: 0; color: var(--muted); line-height: 1.5; }
    .faq { display: grid; gap: 12px; }
    details { border: 1px solid var(--line); border-radius: 8px; background: rgba(11, 18, 31, .78); padding: 18px 20px; }
    summary { cursor: pointer; font-weight: 900; font-size: 18px; }
    details p { color: var(--muted); line-height: 1.6; margin: 12px 0 0; }
    .final-cta { text-align: center; border: 1px solid var(--line); border-radius: 18px; background: linear-gradient(135deg, rgba(47,111,229,.20), rgba(255,128,109,.12)); padding: 46px 20px; }
    .final-cta p { margin: 12px auto 26px; max-width: 680px; }
    footer { border-top: 1px solid rgba(32,48,76,.65); color: #6f86aa; padding: 26px 0; text-align: center; }
    footer p { margin: 0 0 10px; }
    footer a { color: #9cbcff; text-decoration: none; font-weight: 900; }
    @media (max-width: 900px) {
      .hero-grid, .comparison-grid, .features, .steps { grid-template-columns: 1fr; }
      .hero { min-height: auto; }
      .trust { grid-template-columns: 1fr; }
      .navlinks a:not(.button) { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap nav">
      <div class="brand">DeepSky <span>Processor</span></div>
      <nav class="navlinks">
        <a href="#results">Results</a>
        <a href="#how">How it works</a>
        <a href="/docs">Docs</a>
        <a href="#faq">FAQ</a>
        <a class="button" href="/process">Process An Image</a>
      </nav>
    </div>
  </header>
  <main>
    <section class="hero">
      <div class="wrap hero-grid">
        <div>
          <div class="eyebrow"><span class="dot"></span> Deep-sky processing in your browser</div>
          <h1>Turn noisy captures into <span class="gradient">finished space images</span>.</h1>
          <p class="lead">DeepSky processes FITS and TIFF files with a dedicated astrophotography pipeline for galaxies, nebulae, and star clusters. Upload, run, compare, and download the result.</p>
          <div class="hero-actions">
            <a class="button" href="/process">Process An Image</a>
            <a class="secondary" href="/docs">Read the docs</a>
            <a class="secondary" href="#results">See before and after</a>
          </div>
          <div class="trust">
            <div><strong>300 MB</strong><span>FITS/TIFF uploads</span></div>
            <div><strong>1 click</strong><span>full pipeline</span></div>
            <div><strong>RGB</strong><span>color-preserving output</span></div>
          </div>
        </div>
        <div class="hero-showcase">
          <img src="/static/landing/heart_after.png" alt="Processed emission nebula example">
        </div>
      </div>
    </section>
    <section id="results">
      <div class="wrap">
        <div class="section-head">
          <h2>Built for the messy files real imagers upload.</h2>
          <p>DeepSky is designed around common SeeStar, FITS, and TIFF workflows: green casts, lifted backgrounds, noisy skies, faint arms, soft nebula structure, and files that may already be stretched.</p>
        </div>
        <div class="comparison-grid">
          <article class="slider">
            <div class="compare" style="--pos: 50%">
              <img src="/static/landing/heart_before.png" alt="Before nebula processing">
              <img class="after" src="/static/landing/heart_after.png" alt="After nebula processing">
              <div class="handle"></div>
            </div>
            <input type="range" min="0" max="100" value="50" aria-label="Compare nebula before and after">
          </article>
          <article class="slider">
            <div class="compare" style="--pos: 50%">
              <img src="/static/landing/galaxy_before.png" alt="Before galaxy processing">
              <img class="after" src="/static/landing/galaxy_after.jpg" alt="After galaxy processing">
              <div class="handle"></div>
            </div>
            <input type="range" min="0" max="100" value="50" aria-label="Compare galaxy before and after">
          </article>
          <article class="slider">
            <div class="compare" style="--pos: 50%">
              <img src="/static/landing/spiral_before.png" alt="Before spiral galaxy processing">
              <img class="after" src="/static/landing/spiral_after.png" alt="After spiral galaxy processing">
              <div class="handle"></div>
            </div>
            <input type="range" min="0" max="100" value="50" aria-label="Compare spiral galaxy before and after">
          </article>
          <article class="slider">
            <div class="compare" style="--pos: 50%">
              <img src="/static/landing/galaxies_before.png" alt="Before galaxy pair processing">
              <img class="after" src="/static/landing/galaxies_after.png" alt="After galaxy pair processing">
              <div class="handle"></div>
            </div>
            <input type="range" min="0" max="100" value="50" aria-label="Compare galaxy pair before and after">
          </article>
          <article class="slider">
            <div class="compare" style="--pos: 50%">
              <img src="/static/landing/cluster_before.png" alt="Before Orion Nebula processing">
              <img class="after" src="/static/landing/cluster_after.png" alt="After Orion Nebula processing">
              <div class="handle"></div>
            </div>
            <input type="range" min="0" max="100" value="50" aria-label="Compare Orion Nebula before and after">
          </article>
          <article class="slider">
            <div class="compare" style="--pos: 50%">
              <img src="/static/landing/horsehead_before.jpg" alt="Before Horsehead Nebula processing">
              <img class="after" src="/static/landing/horsehead_after.jpg" alt="After Horsehead Nebula processing">
              <div class="handle"></div>
            </div>
            <input type="range" min="0" max="100" value="50" aria-label="Compare Horsehead Nebula before and after">
          </article>
        </div>
      </div>
    </section>
    <section>
      <div class="wrap">
        <div class="section-head">
          <h2>Why use DeepSky?</h2>
          <p>Astrophotography processing usually means juggling multiple tools, settings, and guesses. DeepSky gives beginners and serious hobbyists a faster first result while keeping the workflow simple.</p>
        </div>
        <div class="features">
          <article class="feature"><h3>Cleaner backgrounds</h3><p>Reduce color speckling and noisy sky while protecting stars and the main target.</p></article>
          <article class="feature"><h3>Object-aware processing</h3><p>Choose nebula, galaxy, or star cluster so the pipeline uses the right finishing style.</p></article>
          <article class="feature"><h3>Pre-stretched friendly</h3><p>DeepSky can skip the heavy stretch when an image already has a visible histogram.</p></article>
          <article class="feature"><h3>Color preservation</h3><p>RGB files stay RGB through preview, processing, and output.</p></article>
          <article class="feature"><h3>FITS and TIFF support</h3><p>Upload common astronomy formats without converting files by hand first.</p></article>
          <article class="feature"><h3>Downloadable results</h3><p>Review the processed image, then save the final TIFF or PNG.</p></article>
        </div>
      </div>
    </section>
    <section id="how">
      <div class="wrap">
        <div class="section-head">
          <h2>From upload to result in minutes.</h2>
        </div>
        <div class="steps">
          <article class="step"><h3>Upload</h3><p>Drop a FITS or TIFF file and preview it immediately.</p></article>
          <article class="step"><h3>Choose target</h3><p>Select nebula, galaxy, or star cluster for the right processing profile.</p></article>
          <article class="step"><h3>Run pipeline</h3><p>DeepSky stretches, calibrates color, denoises, and prepares the final image.</p></article>
          <article class="step"><h3>Compare</h3><p>Review before and after side by side, then download the finished TIFF.</p></article>
        </div>
      </div>
    </section>
    <section id="faq">
      <div class="wrap">
        <div class="section-head">
          <h2>FAQ</h2>
        </div>
        <div class="faq">
          <details open><summary>What files can I upload?</summary><p>DeepSky currently accepts FITS, FIT, FTS, TIF, and TIFF files up to 300 MB.</p></details>
          <details><summary>Does it work with SeeStar files?</summary><p>Yes. DeepSky is being tuned around real SeeStar-style FITS and TIFF uploads, including both linear and already-stretched files.</p></details>
          <details><summary>Is this catalog photometric color calibration?</summary><p>When Siril PCC is available and the file has enough metadata, catalog-based star color calibration can be used. Otherwise DeepSky uses pixel-based background and star balancing.</p></details>
          <details><summary>Will it replace manual processing?</summary><p>No. It is meant to produce a strong first processed result quickly, especially for users who do not want to spend hours tuning multiple astronomy tools.</p></details>
          <details><summary>Do I need an account?</summary><p>Yes. Create an account or sign in to process images. Your first 5 images are free.</p></details>
        </div>
      </div>
    </section>
    <section>
      <div class="wrap final-cta">
        <h2>Ready to process your next capture?</h2>
        <p class="copy">Upload a deep-sky image, run the pipeline, and compare the result side by side.</p>
        <a class="button" href="/process">Process An Image</a>
      </div>
    </section>
  </main>
  <footer>
    <p><a href="https://www.facebook.com/deepskyprocessor/" target="_blank" rel="noreferrer">Don't like your image output? Message us a picture of your processed image and the file, we will fix any issues.</a></p>
    <p>DeepSky Built By
    <a href="https://www.linkedin.com/in/diego-perry-64a609240/" target="_blank" rel="noreferrer">Diego Perry</a></p>
  </footer>
  <script>
    document.querySelectorAll(".slider").forEach((slider) => {
      const compare = slider.querySelector(".compare");
      const input = slider.querySelector("input");
      const setPosition = (value) => {
        const next = Math.max(0, Math.min(100, value));
        input.value = String(next);
        compare.style.setProperty("--pos", `${next}%`);
      };
      const setFromEvent = (event) => {
        const rect = compare.getBoundingClientRect();
        const x = event.clientX - rect.left;
        setPosition((x / rect.width) * 100);
      };
      let dragging = false;
      compare.addEventListener("pointerdown", (event) => {
        dragging = true;
        compare.setPointerCapture(event.pointerId);
        setFromEvent(event);
      });
      compare.addEventListener("pointermove", (event) => {
        if (dragging) setFromEvent(event);
      });
      compare.addEventListener("pointerup", (event) => {
        dragging = false;
        compare.releasePointerCapture(event.pointerId);
      });
      compare.addEventListener("pointercancel", () => {
        dragging = false;
      });
      input.addEventListener("input", () => setPosition(Number(input.value)));
      setPosition(Number(input.value));
    });
  </script>
</body>
</html>"""


def _html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DeepSky</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #070b13;
      --panel: #0d1420;
      --line: #243450;
      --muted: #8198bd;
      --text: #f7fbff;
      --blue: #5b8cff;
      --violet: #a88cff;
      --coral: #ff806d;
      --warning: #f6c453;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: radial-gradient(circle at top, #0c1527 0, var(--bg) 420px);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1180px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 44px 0 28px;
    }
    .hero { text-align: center; display: grid; gap: 18px; place-items: center; }
    .pill {
      border: 1px solid #234ea8;
      border-radius: 999px;
      color: #8db5ff;
      background: #101a32;
      padding: 6px 16px;
      font: 12px Consolas, ui-monospace, monospace;
    }
    h1 { margin: 0; font-size: clamp(52px, 7vw, 86px); line-height: .95; letter-spacing: 0; }
    .cosmos {
      background: linear-gradient(90deg, var(--blue), var(--violet), var(--coral));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }
    .subtitle {
      max-width: 760px;
      margin: 0;
      color: var(--muted);
      font-size: clamp(16px, 2vw, 22px);
      line-height: 1.55;
    }
    .account-bar {
      min-height: 38px;
      display: flex;
      justify-content: flex-end;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      color: #9fb5d7;
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 18px;
    }
    .account-bar[hidden], .auth-panel[hidden], .app-shell[hidden] { display: none; }
    .link-button {
      appearance: none;
      border: 1px solid #2c4773;
      border-radius: 8px;
      background: #0b1628;
      color: #cfe0ff;
      font-weight: 800;
      padding: 9px 13px;
      cursor: pointer;
    }
    .billing-status {
      border: 1px solid #263957;
      border-radius: 999px;
      background: #0b1628;
      color: #cfe0ff;
      padding: 8px 12px;
    }
    .auth-panel {
      width: min(520px, 100%);
      margin: 54px auto 72px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(13, 20, 32, .86);
      padding: 24px;
      display: grid;
      gap: 14px;
    }
    .auth-panel h2 { margin: 0; font-size: 24px; letter-spacing: 0; }
    .auth-panel p { margin: 0; color: var(--muted); line-height: 1.5; }
    .auth-field {
      display: grid;
      gap: 7px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 800;
    }
    .auth-field input {
      min-height: 44px;
      border: 1px solid #2c4773;
      border-radius: 8px;
      background: #070b12;
      color: var(--text);
      padding: 0 12px;
      font-size: 15px;
    }
    .auth-actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 4px; }
    .auth-message { min-height: 22px; color: #9fb5d7; font-size: 14px; }
    .forgot-button {
      appearance: none;
      border: 0;
      background: transparent;
      color: #9cbcff;
      font-weight: 800;
      padding: 0;
      width: fit-content;
      cursor: pointer;
    }
    .reset-password-panel {
      display: grid;
      gap: 14px;
    }
    .reset-password-panel[hidden] { display: none; }
    .drop {
      width: min(720px, 100%);
      min-height: 184px;
      border: 2px dashed var(--line);
      border-radius: 18px;
      background: rgba(13, 20, 32, .82);
      display: grid;
      place-items: center;
      padding: 28px;
      cursor: pointer;
      transition: border-color .18s, background .18s, transform .18s;
    }
    .drop:hover, .drop.drag {
      border-color: var(--blue);
      background: #101a2a;
      transform: translateY(-1px);
    }
    .drop strong { display: block; font-size: clamp(24px, 3vw, 34px); line-height: 1.1; }
    .drop span { display: block; margin-top: 12px; color: var(--muted); font: 14px Consolas, ui-monospace, monospace; }
    input[type=file] { display: none; }
    .cta {
      margin-top: 22px;
      appearance: none;
      border: 0;
      border-radius: 10px;
      background: #2f6fe5;
      color: white;
      font-weight: 800;
      font-size: 16px;
      padding: 15px 34px;
      min-width: 260px;
      cursor: pointer;
    }
    .cta:disabled { opacity: .55; cursor: not-allowed; }
    .actions {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: center;
      margin-top: 22px;
    }
    .mode {
      border: 1px solid #2c4773;
      border-radius: 10px;
      background: #0b1628;
      color: #cfe0ff;
      font-weight: 800;
      font-size: 15px;
      padding: 14px 22px;
      min-width: 170px;
      cursor: pointer;
    }
    .mode.active {
      border-color: rgba(246, 196, 83, .8);
      background: rgba(246, 196, 83, .14);
      color: #ffe3a0;
    }
    .select {
      display: grid;
      gap: 7px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-align: left;
    }
    .select select {
      min-width: 190px;
      border: 1px solid #2c4773;
      border-radius: 10px;
      background: #0b1628;
      color: #f7fbff;
      font-weight: 800;
      font-size: 15px;
      padding: 13px 14px;
    }
    .toggle {
      min-height: 48px;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      border: 1px solid #2c4773;
      border-radius: 10px;
      background: #0b1628;
      color: #f7fbff;
      font-size: 14px;
      font-weight: 800;
      padding: 12px 14px;
      cursor: pointer;
    }
    .toggle input {
      width: 18px;
      height: 18px;
      accent-color: var(--blue);
    }
    .toggle.disabled {
      opacity: .52;
      cursor: not-allowed;
    }
    .toggle.disabled input {
      cursor: not-allowed;
    }
    .detail-option {
      width: min(720px, 100%);
      margin: 16px auto 0;
      border: 1px solid rgba(80, 125, 190, .42);
      border-radius: 12px;
      background: rgba(11, 22, 40, .72);
      padding: 16px 18px;
      text-align: left;
    }
    .detail-option h3 {
      margin: 0 0 7px;
      color: #f7fbff;
      font-size: 16px;
      letter-spacing: 0;
    }
    .detail-option p {
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .status { min-height: 24px; color: var(--muted); margin-top: 12px; }
    .warning {
      display: none;
      width: min(720px, 100%);
      border: 1px solid rgba(246, 196, 83, .55);
      background: rgba(246, 196, 83, .10);
      color: #ffe3a0;
      border-radius: 12px;
      padding: 12px 14px;
      line-height: 1.45;
      text-align: left;
    }
    .progress-panel {
      display: grid;
      gap: 10px;
      width: min(720px, 100%);
      margin: 16px auto 0;
    }
    .progress-panel[hidden] { display: none; }
    .progress-row {
      display: grid;
      grid-template-columns: minmax(160px, 220px) minmax(0, 1fr) 48px;
      align-items: center;
      gap: 12px;
      color: #8fb0df;
      font-size: 13px;
      font-weight: 800;
      text-align: left;
    }
    .progress-track {
      height: 9px;
      overflow: hidden;
      border-radius: 999px;
      background: #0b1524;
      border: 1px solid #203552;
    }
    .progress-fill {
      display: block;
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #2f6fe5, #7aa7ff);
      transition: width .22s ease;
    }
    .progress-fill.indeterminate {
      width: 38%;
      background: linear-gradient(90deg, transparent, #2f6fe5, #7aa7ff, transparent);
      animation: progressSlide 1.05s ease-in-out infinite;
    }
    @keyframes progressSlide {
      0% { transform: translateX(-120%); }
      100% { transform: translateX(285%); }
    }
    .previews {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 34px;
    }
    .preview {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #090f18;
      padding: 14px;
      min-height: 360px;
    }
    .preview h2 { margin: 0 0 12px; font-size: 15px; color: #dce6f6; }
    .frame {
      min-height: 310px;
      border: 1px solid #1f2d43;
      background: #05080d;
      display: grid;
      place-items: center;
      overflow: hidden;
    }
    .frame img { max-width: 100%; max-height: 520px; object-fit: contain; display: block; }
    .empty { color: #52667f; }
    .downloads { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; margin-top: 18px; }
    .downloads a {
      color: #cfe0ff;
      border: 1px solid #2c4773;
      background: #0b1628;
      border-radius: 999px;
      padding: 8px 14px;
      text-decoration: none;
      font-weight: 700;
    }
    .footer {
      margin-top: 24px;
      text-align: center;
      color: #6f86aa;
      font-size: 14px;
    }
    .footer p { margin: 0 0 10px; }
    .footer a {
      color: #9cbcff;
      text-decoration: none;
      font-weight: 800;
    }
    .footer a:hover { text-decoration: underline; }
    .processing-indicator {
      display: none;
      align-items: center;
      justify-content: center;
      gap: 12px;
      margin-top: 18px;
      color: #9db9ec;
      font-weight: 800;
    }
    .processing-indicator.active { display: flex; }
    .spinner {
      width: 22px;
      height: 22px;
      border-radius: 999px;
      border: 3px solid #1e355a;
      border-top-color: #6ea1ff;
      animation: spin .8s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    @media (max-width: 780px) {
      main { width: min(100vw - 24px, 1180px); padding-top: 26px; }
      .previews { grid-template-columns: 1fr; }
      .preview { min-height: 280px; }
    }
  </style>
</head>
<body>
  <main>
    <div id="accountBar" class="account-bar" hidden>
      <span id="accountEmail"></span>
      <span id="billingStatus" class="billing-status"></span>
      <button id="upgradePlan" class="link-button" type="button" hidden>Upgrade $15/mo</button>
      <button id="signOut" class="link-button" type="button">Sign out</button>
    </div>
    <section id="authPanel" class="auth-panel" hidden>
      <h2>Sign in to process images</h2>
      <p>Create an account or sign in before running the DeepSky pipeline.</p>
      <label class="auth-field">
        Email
        <input id="authEmail" type="email" autocomplete="email" required />
      </label>
      <label class="auth-field">
        Password
        <input id="authPassword" type="password" autocomplete="current-password" minlength="6" required />
      </label>
      <button id="forgotPassword" class="forgot-button" type="button">Forgot password?</button>
      <div class="auth-actions">
        <button id="signIn" class="cta" type="button">Sign in</button>
        <button id="signUp" class="link-button" type="button">Create account</button>
      </div>
      <div id="authMessage" class="auth-message"></div>
      <div id="resetPasswordPanel" class="reset-password-panel" hidden>
        <h2>Reset password</h2>
        <p>Enter a new password for your DeepSky account.</p>
        <label class="auth-field">
          New password
          <input id="newPassword" type="password" autocomplete="new-password" minlength="8" />
        </label>
        <label class="auth-field">
          Confirm new password
          <input id="confirmPassword" type="password" autocomplete="new-password" minlength="8" />
        </label>
        <button id="updatePassword" class="cta" type="button">Update password</button>
      </div>
    </section>
    <div id="appShell" class="app-shell" hidden>
    <section class="hero">
      <div class="pill">● DeepSky Astrophotography Pipeline v1.0</div>
      <h1>Process the <span class="cosmos">Cosmos</span></h1>
      <p class="subtitle">Turn raw deep-sky captures into processed results in minutes. Upload a FITS or TIFF file, run the full pipeline, and compare the before and after side by side.</p>
      <label id="drop" class="drop">
        <input id="file" type="file" accept=".fits,.fit,.fts,.tif,.tiff" />
        <div>
          <strong>Drag & drop your astrophotography file</strong>
          <span id="fileName">Supports FITS and TIFF formats up to 300 MB</span>
        </div>
      </label>
      <div class="actions">
        <label class="select">
          Object
          <select id="objectType">
            <option value="Nebula" selected>Nebula</option>
            <option value="Galaxy">Galaxy</option>
            <option value="Star Cluster">Star Cluster</option>
          </select>
        </label>
        <label class="select">
          Input
          <select id="inputMode">
            <option value="Auto" selected>Auto</option>
            <option value="Linear">Linear</option>
            <option value="Pre-stretched">Pre-stretched</option>
          </select>
        </label>
        <label class="select">
          Stretch
          <select id="stretchLevel">
            <option value="Subtle">Subtle</option>
            <option value="Standard" selected>Standard</option>
            <option value="Aggressive">Aggressive</option>
          </select>
        </label>
        <button id="run" class="cta" disabled>Run Full Pipeline</button>
      </div>
      <div class="detail-option">
        <h3>Want to add more detail? Test out deconvolution</h3>
        <p>Deconvolution can sharpen galaxy arms and dust lanes when the data is clean. On noisy or faint images it can also make the background look grainy, so compare both versions.</p>
        <label class="toggle" title="Optional: applies Siril Richardson-Lucy deconvolution only for galaxy processing.">
          <input id="sirilDeconvolution" type="checkbox" />
          Use deconvolution
        </label>
      </div>
      <div id="warning" class="warning"></div>
      <div id="progressPanel" class="progress-panel" hidden>
        <div class="progress-row" id="uploadProgressRow">
          <span id="uploadProgressLabel">Uploading file</span>
          <div class="progress-track"><span id="uploadProgressFill" class="progress-fill"></span></div>
          <span id="uploadProgressValue">0%</span>
        </div>
        <div class="progress-row" id="previewProgressRow">
          <span id="previewProgressLabel">Loading preview</span>
          <div class="progress-track"><span id="previewProgressFill" class="progress-fill"></span></div>
          <span id="previewProgressValue">0%</span>
        </div>
      </div>
      <div id="status" class="status">Choose a file to begin.</div>
    </section>
    <section class="previews">
      <article class="preview">
        <h2>Before</h2>
        <div class="frame" id="beforeFrame"><span class="empty">No image selected</span></div>
      </article>
      <article class="preview">
        <h2>After</h2>
        <div class="frame" id="afterFrame"><span class="empty">Waiting for processing</span></div>
      </article>
    </section>
    <nav class="downloads" id="downloads"></nav>
    <div class="processing-indicator" id="processingIndicator">
      <span class="spinner" aria-hidden="true"></span>
      <span>Processing image...</span>
    </div>
    </div>
    <footer class="footer">
      <p><a href="/docs">Processing docs and troubleshooting guide</a></p>
      <p><a href="https://www.facebook.com/deepskyprocessor/" target="_blank" rel="noreferrer">Don't like your image output? Message us a picture of your processed image and the file, we will fix any issues.</a></p>
      <p>DeepSky Built By
      <a href="https://www.linkedin.com/in/diego-perry-64a609240/" target="_blank" rel="noreferrer">Diego Perry</a></p>
    </footer>
  </main>
  <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
  <script>
    const drop = document.getElementById("drop");
    const fileInput = document.getElementById("file");
    const fileName = document.getElementById("fileName");
    const run = document.getElementById("run");
    const objectType = document.getElementById("objectType");
    const inputMode = document.getElementById("inputMode");
    const stretchLevel = document.getElementById("stretchLevel");
    const sirilDeconvolution = document.getElementById("sirilDeconvolution");
    const statusEl = document.getElementById("status");
    const warningEl = document.getElementById("warning");
    const progressPanel = document.getElementById("progressPanel");
    const uploadProgressRow = document.getElementById("uploadProgressRow");
    const uploadProgressLabel = document.getElementById("uploadProgressLabel");
    const uploadProgressFill = document.getElementById("uploadProgressFill");
    const uploadProgressValue = document.getElementById("uploadProgressValue");
    const previewProgressRow = document.getElementById("previewProgressRow");
    const previewProgressLabel = document.getElementById("previewProgressLabel");
    const previewProgressFill = document.getElementById("previewProgressFill");
    const previewProgressValue = document.getElementById("previewProgressValue");
    const processingIndicator = document.getElementById("processingIndicator");
    const beforeFrame = document.getElementById("beforeFrame");
    const afterFrame = document.getElementById("afterFrame");
    const downloads = document.getElementById("downloads");
    const accountBar = document.getElementById("accountBar");
    const accountEmail = document.getElementById("accountEmail");
    const billingStatus = document.getElementById("billingStatus");
    const upgradePlan = document.getElementById("upgradePlan");
    const signOut = document.getElementById("signOut");
    const authPanel = document.getElementById("authPanel");
    const authIntroTitle = authPanel.querySelector("h2");
    const authIntroCopy = authPanel.querySelector("p");
    const appShell = document.getElementById("appShell");
    const authEmail = document.getElementById("authEmail");
    const authPassword = document.getElementById("authPassword");
    const signIn = document.getElementById("signIn");
    const signUp = document.getElementById("signUp");
    const forgotPassword = document.getElementById("forgotPassword");
    const authActions = authPanel.querySelector(".auth-actions");
    const authMessage = document.getElementById("authMessage");
    const resetPasswordPanel = document.getElementById("resetPasswordPanel");
    const newPassword = document.getElementById("newPassword");
    const confirmPassword = document.getElementById("confirmPassword");
    const updatePassword = document.getElementById("updatePassword");
    const AUTH_REDIRECT_URL = "https://app.deepskyprocessor.com/process";
    let authClient = null;
    let session = null;
    let recoveryMode = false;
    let billingStatusPromise = null;
    let selectedFile = null;
    let stagedUpload = null;
    let activeJob = null;
    let previewRequest = 0;
    const MAX_UPLOAD_BYTES_CLIENT = 300 * 1024 * 1024;
    const CHUNKED_UPLOAD_THRESHOLD = 45 * 1024 * 1024;

    function setAuthMessage(message) {
      authMessage.textContent = message || "";
    }

    function authCredentials() {
      const email = authEmail.value.trim();
      const password = authPassword.value;
      if (!email) {
        throw new Error("Enter your email address.");
      }
      if (!authEmail.checkValidity()) {
        throw new Error("Enter a valid email address.");
      }
      if (!password) {
        throw new Error("Enter your password.");
      }
      if (password.length < 6) {
        throw new Error("Password must be at least 6 characters.");
      }
      return { email, password };
    }

    function authEmailValue() {
      const email = authEmail.value.trim();
      if (!email) {
        throw new Error("Enter your email address.");
      }
      if (!authEmail.checkValidity()) {
        throw new Error("Enter a valid email address.");
      }
      return email;
    }

    function resetPasswordValues() {
      const password = newPassword.value;
      const confirmation = confirmPassword.value;
      if (password.length < 8) {
        throw new Error("Password must be at least 8 characters.");
      }
      if (password !== confirmation) {
        throw new Error("New password and confirmation must match.");
      }
      return password;
    }

    function authErrorMessage(error) {
      const message = error && error.message ? error.message : String(error || "Authentication failed.");
      const normalized = message.toLowerCase();
      if (normalized.includes("invalid login credentials")) {
        return "Invalid email or password.";
      }
      if (normalized.includes("email") && normalized.includes("invalid")) {
        return "Enter a valid email address.";
      }
      if (normalized.includes("password")) {
        return message;
      }
      if (normalized.includes("anonymous")) {
        return "Email and password are required. Guest sign-in is not enabled.";
      }
      return message;
    }

    function setAuthBusy(isBusy) {
      signIn.disabled = isBusy;
      signUp.disabled = isBusy;
      forgotPassword.disabled = isBusy;
      updatePassword.disabled = isBusy;
    }

    function cleanAuthUrl() {
      if (window.location.pathname === "/process" && (window.location.hash || window.location.search)) {
        window.history.replaceState({}, document.title, "/process");
      }
    }

    function setRecoveryMode(isActive, nextSession = session) {
      recoveryMode = isActive;
      session = nextSession;
      authPanel.hidden = false;
      appShell.hidden = true;
      accountBar.hidden = true;
      resetPasswordPanel.hidden = !isActive;
      authIntroTitle.hidden = isActive;
      authIntroCopy.hidden = isActive;
      authEmail.closest("label").hidden = isActive;
      authPassword.closest("label").hidden = isActive;
      forgotPassword.hidden = isActive;
      authActions.hidden = isActive;
      if (isActive) {
        newPassword.focus();
      }
    }

    function setSignedIn(nextSession) {
      if (recoveryMode) {
        setRecoveryMode(true, nextSession);
        return;
      }
      session = nextSession;
      const user = session && session.user;
      appShell.hidden = !user;
      accountBar.hidden = !user;
      authPanel.hidden = !!user;
      resetPasswordPanel.hidden = true;
      authIntroTitle.hidden = false;
      authIntroCopy.hidden = false;
      authEmail.closest("label").hidden = false;
      authPassword.closest("label").hidden = false;
      forgotPassword.hidden = false;
      authActions.hidden = false;
      accountEmail.textContent = user ? (user.email || "Signed in") : "";
      if (user) {
        void loadBillingStatus();
      } else {
        billingStatus.textContent = "";
        upgradePlan.hidden = true;
      }
      if (!user) {
        selectedFile = null;
        activeJob = null;
        resetProgress();
      }
    }

    async function getAccessToken() {
      if (!authClient) return null;
      const { data } = await authClient.auth.getSession();
      session = data.session;
      return session && session.access_token ? session.access_token : null;
    }

    async function authHeaders() {
      const token = await getAccessToken();
      if (!token) throw new Error("Sign in to continue.");
      return { Authorization: `Bearer ${token}` };
    }

    function setProgress(fill, value, percent) {
      fill.classList.remove("indeterminate");
      fill.style.transform = "";
      const bounded = Math.max(0, Math.min(100, Math.round(percent)));
      fill.style.width = `${bounded}%`;
      value.textContent = `${bounded}%`;
    }

    function setIndeterminateProgress(fill, value, text) {
      fill.style.width = "";
      fill.style.transform = "";
      fill.classList.add("indeterminate");
      value.textContent = text;
    }

    function resetProgress() {
      progressPanel.hidden = true;
      uploadProgressRow.hidden = false;
      previewProgressRow.hidden = false;
      uploadProgressFill.classList.remove("indeterminate");
      previewProgressFill.classList.remove("indeterminate");
      setProgress(uploadProgressFill, uploadProgressValue, 0);
      setProgress(previewProgressFill, previewProgressValue, 0);
    }

    function showUploadProgress(label) {
      progressPanel.hidden = false;
      uploadProgressRow.hidden = false;
      previewProgressRow.hidden = true;
      uploadProgressLabel.textContent = label;
      setIndeterminateProgress(uploadProgressFill, uploadProgressValue, "Uploading");
    }

    function showPreviewProgress() {
      progressPanel.hidden = false;
      uploadProgressRow.hidden = false;
      previewProgressRow.hidden = false;
      previewProgressLabel.textContent = "Loading preview";
      setIndeterminateProgress(previewProgressFill, previewProgressValue, "Working");
    }

    function showPipelineProgress(stage, percent) {
      progressPanel.hidden = false;
      uploadProgressRow.hidden = true;
      previewProgressRow.hidden = false;
      previewProgressLabel.textContent = stage || "Processing";
      setProgress(previewProgressFill, previewProgressValue, percent || 0);
    }

    async function fetchAuthed(url) {
      return fetch(url, { headers: await authHeaders() });
    }

    async function readJsonResponse(response, fallbackMessage) {
      const contentType = response.headers.get("content-type") || "";
      if (!contentType.toLowerCase().includes("application/json")) {
        const text = await response.text().catch(() => "");
        const snippet = text.replace(/\s+/g, " ").trim().slice(0, 120);
        const status = response.status ? `HTTP ${response.status}` : "no HTTP status";
        const type = contentType || "no content-type";
        throw new Error(`${fallbackMessage} (${status}, ${type}${snippet ? `: ${snippet}` : ""})`);
      }
      return response.json();
    }

    async function postJsonAuthed(url, payload = {}, fallbackMessage = "Service temporarily unavailable. Please refresh.") {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          ...(await authHeaders()),
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      const data = await readJsonResponse(response, fallbackMessage);
      if (!response.ok) {
        throw new Error(data.detail || data.error || "Request failed.");
      }
      return data;
    }

    async function loadBillingStatus() {
      if (!session || !session.user) return;
      if (billingStatusPromise) return billingStatusPromise;
      billingStatusPromise = (async () => {
      try {
        const response = await fetchAuthed("/api/billing/status");
        const data = await readJsonResponse(response, "Billing service temporarily unavailable. Please refresh.");
        if (!response.ok) {
          throw new Error(data.detail || data.error || "Billing status unavailable.");
        }
        if (data.is_paid) {
          billingStatus.textContent = "Paid plan active";
          upgradePlan.textContent = "Manage plan";
          upgradePlan.dataset.billingAction = "portal";
          upgradePlan.hidden = false;
        } else {
          const credits = Number(data.free_credits_remaining || 0);
          billingStatus.textContent = `${credits} free image${credits === 1 ? "" : "s"} left`;
          upgradePlan.textContent = "Upgrade $15/mo";
          upgradePlan.dataset.billingAction = "checkout";
          upgradePlan.hidden = false;
        }
      } catch (error) {
        billingStatus.textContent = error.message || "Billing status unavailable.";
        upgradePlan.hidden = true;
      } finally {
        billingStatusPromise = null;
      }
      })();
      return billingStatusPromise;
    }

    async function loadImageIntoFrame(url, frame, alt) {
      const response = await fetchAuthed(url);
      if (!response.ok) throw new Error("Image is not available yet.");
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      frame.innerHTML = `<img src="${objectUrl}" alt="${alt}">`;
    }

    async function downloadFile(url, filename) {
      const response = await fetchAuthed(url);
      if (!response.ok) throw new Error("Download is not available yet.");
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    }

    function renderAcceptedDownloads(job) {
      downloads.innerHTML = `
        <button class="link-button" type="button" data-download-url="${job.final}" data-download-name="deepsky-final.tif">Download TIFF</button>
        <button class="link-button" type="button" data-download-url="${job.png}" data-download-name="deepsky-final.png">Download PNG</button>
      `;
    }

    async function postFormJson(url, data, { onUploadProgress, onServerWait } = {}) {
      const headers = await authHeaders();
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        let lastPercent = 0;
        xhr.open("POST", url);
        Object.entries(headers).forEach(([key, value]) => xhr.setRequestHeader(key, value));
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable && onUploadProgress) {
            const percent = (event.loaded / event.total) * 100;
            if (percent >= 3 && percent > lastPercent + 1) {
              lastPercent = percent;
              onUploadProgress(percent);
            }
          }
        };
        xhr.upload.onload = () => {
          if (onUploadProgress) onUploadProgress(100);
          if (onServerWait) onServerWait();
        };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              resolve(JSON.parse(xhr.responseText));
            } catch (error) {
              reject(error);
            }
          } else {
            let message = xhr.responseText || `Request failed with status ${xhr.status}`;
            const contentType = xhr.getResponseHeader("content-type") || "";
            if (contentType.toLowerCase().includes("application/json")) {
              try {
                const parsed = JSON.parse(xhr.responseText);
                message = parsed.detail || parsed.error || message;
              } catch (_error) {}
            } else {
              message = "Processing service temporarily unavailable. Please refresh.";
            }
            const error = new Error(message);
            error.status = xhr.status;
            reject(error);
          }
        };
        xhr.onerror = () => reject(new Error("Network error while uploading file."));
        xhr.send(data);
      });
    }

    async function uploadFileInChunks(file, { onUploadProgress, onServerWait } = {}) {
      const initResponse = await fetch("/api/uploads/init", {
        method: "POST",
        headers: {
          ...(await authHeaders()),
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ filename: file.name, size: file.size }),
      });
      const initData = await readJsonResponse(initResponse, "Upload service temporarily unavailable. Please refresh.");
      if (!initResponse.ok) {
        throw new Error(initData.detail || initData.error || "Could not start upload.");
      }
      const chunkSize = Number(initData.chunk_size || (8 * 1024 * 1024));
      let offset = 0;
      while (offset < file.size) {
        const chunk = file.slice(offset, Math.min(offset + chunkSize, file.size));
        const chunkResponse = await fetch(`/api/uploads/${initData.upload_id}/chunk`, {
          method: "POST",
          headers: {
            ...(await authHeaders()),
            "Content-Type": "application/octet-stream",
            "X-Upload-Offset": String(offset),
          },
          body: chunk,
        });
        const chunkData = await readJsonResponse(chunkResponse, "Upload service temporarily unavailable. Please refresh.");
        if (!chunkResponse.ok) {
          throw new Error(chunkData.detail || chunkData.error || "Could not upload file chunk.");
        }
        offset = Number(chunkData.offset || (offset + chunk.size));
        if (onUploadProgress) onUploadProgress((offset / file.size) * 100);
      }
      if (onUploadProgress) onUploadProgress(100);
      if (onServerWait) onServerWait();
      return initData.upload_id;
    }

    async function setFile(file) {
      selectedFile = file;
      const requestId = ++previewRequest;
      resetProgress();
      stagedUpload = null;
      fileName.textContent = file ? file.name : "Supports FITS and TIFF formats up to 300 MB";
      const tooLarge = file && file.size > MAX_UPLOAD_BYTES_CLIENT;
      run.disabled = true;
      beforeFrame.innerHTML = file ? '<span class="empty">Loading preview</span>' : '<span class="empty">No image selected</span>';
      afterFrame.innerHTML = '<span class="empty">Waiting for processing</span>';
      downloads.innerHTML = "";
      processingIndicator.classList.remove("active");
      statusEl.textContent = tooLarge ? "File is too large. Maximum upload size is 300 MB." : file ? "Preparing preview..." : "Choose a file to begin.";
      warningEl.style.display = "none";
      warningEl.textContent = "";
      if (inputMode.value === "Pre-stretched") {
        warningEl.style.display = "block";
        warningEl.textContent = "Pre-stretched mode is on. DeepSky will skip its stretch/color-stretch stage and process the image as already stretched.";
      } else if (inputMode.value === "Auto") {
        warningEl.style.display = "block";
        warningEl.textContent = "Auto input mode is on. DeepSky will choose linear, gentle stretch, or pre-stretched processing from the file histogram.";
      }
      if (!file || tooLarge) return;
      try {
        let preview;
        if (file.size > CHUNKED_UPLOAD_THRESHOLD) {
          showUploadProgress("Uploading large file");
          const uploadId = await uploadFileInChunks(file, {
            onUploadProgress: (percent) => setProgress(uploadProgressFill, uploadProgressValue, percent),
            onServerWait: () => {
              uploadProgressValue.textContent = "Done";
              showPreviewProgress();
            },
          });
          if (requestId !== previewRequest) return;
          stagedUpload = { id: uploadId, name: file.name, size: file.size, lastModified: file.lastModified };
          preview = await postJsonAuthed(
            `/api/uploads/${uploadId}/preview`,
            {},
            "Preview service temporarily unavailable. Please refresh."
          );
        } else {
          const data = new FormData();
          data.append("file", file);
          showUploadProgress("Uploading preview");
          preview = await postFormJson("/api/preview", data, {
            onUploadProgress: (percent) => setProgress(uploadProgressFill, uploadProgressValue, percent),
            onServerWait: () => {
              uploadProgressValue.textContent = "Done";
              showPreviewProgress();
            },
          });
        }
        if (requestId !== previewRequest) return;
        setProgress(uploadProgressFill, uploadProgressValue, 100);
        setProgress(previewProgressFill, previewProgressValue, 100);
        await loadImageIntoFrame(`${preview.preview_url}&t=${Date.now()}`, beforeFrame, "Before preview");
        run.disabled = false;
        statusEl.textContent = "Ready to run full pipeline.";
        setTimeout(() => {
          if (requestId === previewRequest) progressPanel.hidden = true;
        }, 700);
      } catch (error) {
        if (requestId !== previewRequest) return;
        progressPanel.hidden = true;
        beforeFrame.innerHTML = '<span class="empty">Preview unavailable</span>';
        if (file.size > CHUNKED_UPLOAD_THRESHOLD && !stagedUpload) {
          run.disabled = true;
          statusEl.textContent = "Large file upload failed. Try selecting the file again.";
        } else {
          run.disabled = false;
          statusEl.textContent = "Ready to run full pipeline.";
        }
        warningEl.style.display = "block";
        warningEl.textContent = `Preview could not be generated yet. ${error.message || error}`;
      }
    }

    inputMode.addEventListener("change", () => {
      if (inputMode.value === "Pre-stretched") {
        warningEl.style.display = "block";
        warningEl.textContent = "Pre-stretched mode is on. DeepSky will skip its stretch/color-stretch stage and process the image as already stretched.";
      } else if (inputMode.value === "Auto") {
        warningEl.style.display = "block";
        warningEl.textContent = "Auto input mode is on. DeepSky will choose linear, gentle stretch, or pre-stretched processing from the file histogram.";
      } else {
        warningEl.style.display = "none";
        warningEl.textContent = "";
      }
    });

    signIn.addEventListener("click", async () => {
      if (!authClient) return;
      try {
        const { email, password } = authCredentials();
        setAuthBusy(true);
        setAuthMessage("Signing in...");
        const { data, error } = await authClient.auth.signInWithPassword({ email, password });
        if (error) {
          setAuthMessage(authErrorMessage(error));
          return;
        }
        setAuthMessage("");
        setSignedIn(data.session);
      } catch (error) {
        setAuthMessage(authErrorMessage(error));
      } finally {
        setAuthBusy(false);
      }
    });

    signUp.addEventListener("click", async () => {
      if (!authClient) return;
      try {
        const { email, password } = authCredentials();
        setAuthBusy(true);
        setAuthMessage("Creating account...");
        const { data, error } = await authClient.auth.signUp({
          email,
          password,
          options: {
            emailRedirectTo: AUTH_REDIRECT_URL,
          },
        });
        if (error) {
          setAuthMessage(authErrorMessage(error));
          return;
        }
        setSignedIn(data.session);
        setAuthMessage(data.session ? "" : "Check your email to confirm your account.");
      } catch (error) {
        setAuthMessage(authErrorMessage(error));
      } finally {
        setAuthBusy(false);
      }
    });

    forgotPassword.addEventListener("click", async () => {
      if (!authClient) return;
      try {
        const email = authEmailValue();
        setAuthBusy(true);
        setAuthMessage("Sending password reset email...");
        const { error } = await authClient.auth.resetPasswordForEmail(email, {
          redirectTo: AUTH_REDIRECT_URL,
        });
        if (error) {
          setAuthMessage(authErrorMessage(error));
          return;
        }
        setAuthMessage("Password reset email sent. Check your inbox.");
      } catch (error) {
        setAuthMessage(authErrorMessage(error));
      } finally {
        setAuthBusy(false);
      }
    });

    updatePassword.addEventListener("click", async () => {
      if (!authClient) return;
      try {
        const password = resetPasswordValues();
        setAuthBusy(true);
        setAuthMessage("Updating password...");
        const { error } = await authClient.auth.updateUser({ password });
        if (error) {
          setAuthMessage(authErrorMessage(error));
          return;
        }
        const { data } = await authClient.auth.getSession();
        newPassword.value = "";
        confirmPassword.value = "";
        recoveryMode = false;
        setAuthMessage("Password updated. You can now continue.");
        statusEl.textContent = "Password updated. You can now continue.";
        setSignedIn(data.session);
      } catch (error) {
        setAuthMessage(authErrorMessage(error));
      } finally {
        setAuthBusy(false);
      }
    });

    signOut.addEventListener("click", async () => {
      if (!authClient) return;
      recoveryMode = false;
      await authClient.auth.signOut();
      setSignedIn(null);
    });

    upgradePlan.addEventListener("click", async () => {
      try {
        upgradePlan.disabled = true;
        const action = upgradePlan.dataset.billingAction || "checkout";
        billingStatus.textContent = action === "portal" ? "Opening billing portal..." : "Opening checkout...";
        const response = await postJsonAuthed(action === "portal" ? "/api/billing/portal" : "/api/billing/checkout");
        window.location.href = response.url;
      } catch (error) {
        billingStatus.textContent = error.message || String(error);
        upgradePlan.disabled = false;
      }
    });

    downloads.addEventListener("click", async (event) => {
      const button = event.target.closest("button[data-download-url]");
      if (!button) return;
      try {
        await downloadFile(button.dataset.downloadUrl, button.dataset.downloadName);
      } catch (error) {
        statusEl.textContent = error.message || String(error);
      }
    });

    fileInput.addEventListener("change", () => { void setFile(fileInput.files[0]); });
    drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("drag"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      drop.classList.remove("drag");
      if (event.dataTransfer.files.length) void setFile(event.dataTransfer.files[0]);
    });

    async function poll(jobId) {
      let job;
      let res;
      try {
        res = await fetchAuthed(`/api/jobs/${jobId}`);
        job = await readJsonResponse(res, "Processing service temporarily unavailable. Please refresh.");
      } catch (error) {
        statusEl.textContent = error.message || String(error);
        processingIndicator.classList.remove("active");
        run.disabled = false;
        return;
      }
      if (!res.ok) {
        statusEl.textContent = job.detail || job.error || "Sign in to continue.";
        processingIndicator.classList.remove("active");
        run.disabled = false;
        return;
      }
      if (job.status === "queued" || job.status === "running") {
        showPipelineProgress(job.stage, job.progress);
        statusEl.textContent = `${job.stage || "Processing"} - ${Math.round(job.progress || 0)}%`;
      } else {
        statusEl.textContent = job.status;
      }
      processingIndicator.classList.toggle("active", job.status === "queued" || job.status === "running");
      if (job.warnings && job.warnings.length) {
        warningEl.style.display = "block";
        warningEl.textContent = job.warnings.join(" ");
      }
      if (job.before_preview && !beforeFrame.querySelector("img")) {
        await loadImageIntoFrame(`${job.before_preview}&t=${Date.now()}`, beforeFrame, "Before preview");
      }
      if (job.after_preview) {
        await loadImageIntoFrame(`${job.after_preview}&t=${Date.now()}`, afterFrame, "After preview");
      }
      if (job.status === "finished") {
        showPipelineProgress("Complete", 100);
        processingIndicator.classList.remove("active");
        statusEl.textContent = "Processing complete. Downloads are ready.";
        renderAcceptedDownloads(job);
        void loadBillingStatus();
        run.disabled = false;
        return;
      }
      if (job.status === "failed") {
        statusEl.textContent = "Processing failed.";
        progressPanel.hidden = true;
        processingIndicator.classList.remove("active");
        run.disabled = false;
        return;
      }
      setTimeout(() => poll(jobId), 1800);
    }

    run.addEventListener("click", async () => {
      if (!selectedFile) return;
      run.disabled = true;
      statusEl.textContent = "Uploading...";
      processingIndicator.classList.add("active");
      downloads.innerHTML = "";
      afterFrame.innerHTML = '<span class="empty">Processing</span>';
      showUploadProgress("Uploading job");
      const data = new FormData();
      const canUseStagedUpload =
        stagedUpload &&
        stagedUpload.name === selectedFile.name &&
        stagedUpload.size === selectedFile.size &&
        stagedUpload.lastModified === selectedFile.lastModified;
      if (canUseStagedUpload) {
        data.append("upload_id", stagedUpload.id);
        setProgress(uploadProgressFill, uploadProgressValue, 100);
        uploadProgressValue.textContent = "Done";
        previewProgressRow.hidden = true;
        statusEl.textContent = "Starting pipeline...";
      } else {
        data.append("file", selectedFile);
      }
      data.append("object_type", objectType.value);
      data.append("input_mode", inputMode.value);
      data.append("stretch_level", stretchLevel.value);
      data.append("siril_deconvolution", sirilDeconvolution.checked ? "true" : "false");
      data.append("star_setting", "Slight Star Reduction");
      data.append("starless_test", "true");
      data.append("pre_stretched", inputMode.value === "Pre-stretched" ? "true" : "false");
      let job;
      try {
        job = await postFormJson("/api/jobs", data, {
          onUploadProgress: (percent) => {
            if (!canUseStagedUpload) setProgress(uploadProgressFill, uploadProgressValue, percent);
          },
          onServerWait: () => {
            if (canUseStagedUpload) return;
            setProgress(uploadProgressFill, uploadProgressValue, 100);
            uploadProgressValue.textContent = "Done";
            previewProgressRow.hidden = true;
            statusEl.textContent = "Starting pipeline...";
          },
        });
      } catch (error) {
        statusEl.textContent = error.message || String(error);
        progressPanel.hidden = true;
        run.disabled = false;
        return;
      }
      progressPanel.hidden = true;
      activeJob = job.id;
      if (canUseStagedUpload) stagedUpload = null;
      void loadBillingStatus();
      showPipelineProgress("Starting Pipeline", 1);
      poll(activeJob);
    });

    async function initAuth() {
      authPanel.hidden = false;
      appShell.hidden = true;
      accountBar.hidden = true;
      try {
        const response = await fetch("/api/auth/config");
        const config = await readJsonResponse(response, "Service temporarily unavailable. Please refresh.");
        if (!config.configured) {
          setAuthMessage("Supabase auth is not configured on this server.");
          signIn.disabled = true;
          signUp.disabled = true;
          return;
        }
        if (!window.supabase) {
          setAuthMessage("Supabase client could not be loaded.");
          signIn.disabled = true;
          signUp.disabled = true;
          return;
        }
        authClient = window.supabase.createClient(config.supabase_url, config.supabase_anon_key);
        const hashParams = new URLSearchParams(window.location.hash.replace(/^#/, ""));
        const queryParams = new URLSearchParams(window.location.search);
        const isRecoveryUrl = hashParams.get("type") === "recovery" || queryParams.get("type") === "recovery";
        const isEmailConfirmationUrl =
          ["signup", "email_change", "magiclink"].includes(hashParams.get("type")) ||
          ["signup", "email_change", "magiclink"].includes(queryParams.get("type")) ||
          hashParams.has("access_token") ||
          queryParams.has("code");
        let authCallbackHandled = false;
        authClient.auth.onAuthStateChange((event, nextSession) => {
          if (event === "PASSWORD_RECOVERY") {
            authCallbackHandled = true;
            setRecoveryMode(true, nextSession);
            cleanAuthUrl();
            return;
          }
          if (nextSession) {
            authCallbackHandled = true;
            if (!recoveryMode) {
              setSignedIn(nextSession);
            }
            if (isEmailConfirmationUrl) {
              cleanAuthUrl();
            }
            return;
          }
          if (!recoveryMode) {
            setSignedIn(nextSession);
          }
        });

        const { data } = await authClient.auth.getSession();
        if (isRecoveryUrl && data.session) {
          setRecoveryMode(true, data.session);
          cleanAuthUrl();
        } else if (data.session) {
          setSignedIn(data.session);
          if (isEmailConfirmationUrl) {
            cleanAuthUrl();
          }
        } else if (isEmailConfirmationUrl) {
          setSignedIn(null);
          window.setTimeout(async () => {
            const { data: refreshed } = await authClient.auth.getSession();
            if (refreshed.session) {
              setSignedIn(refreshed.session);
              cleanAuthUrl();
              return;
            }
            if (!authCallbackHandled) {
              setAuthMessage("Email confirmed. Please sign in to continue.");
              cleanAuthUrl();
            }
          }, 800);
        } else {
          setSignedIn(data.session);
        }
      } catch (error) {
        setAuthMessage(error.message || String(error));
        signIn.disabled = true;
        signUp.disabled = true;
      }
    }

    void initAuth();
  </script>
</body>
</html>"""


def _job_response(job: WebJob) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": job.id,
        "status": job.status,
        "log": job.log[-300:],
        "warnings": job.warnings[-10:],
        "error": job.error,
        "stage": job.stage,
        "progress": job.progress,
        "credit_consumed": job.credit_consumed,
        "can_download": job.status == "finished" and bool(job.result),
    }
    if job.result:
        payload.update(
            {
                "before_preview": f"/api/jobs/{job.id}/file/before_preview?inline=1",
                "after_preview": f"/api/jobs/{job.id}/file/after_preview?inline=1",
            }
        )
        payload.update(
            {
                "final": f"/api/jobs/{job.id}/file/final",
                "png": f"/api/jobs/{job.id}/file/png",
            }
        )
    return payload


def _subscription_payload(subscription: Any, *, status_override: str | None = None) -> dict[str, Any]:
    status = status_override or _object_get(subscription, "status")
    current_period_end = _object_get(subscription, "current_period_end")
    return {
        "subscription_status": status or "free",
        "stripe_subscription_id": _object_get(subscription, "id"),
        "current_period_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_period_end))
        if current_period_end
        else None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _apply_subscription_update(subscription: Any, *, status_override: str | None = None) -> None:
    metadata = _object_get(subscription, "metadata") or {}
    user_id = metadata.get("user_id")
    customer_id = _object_get(subscription, "customer")
    updates = _subscription_payload(subscription, status_override=status_override)
    if customer_id:
        updates["stripe_customer_id"] = customer_id
    if user_id:
        _update_profile(user_id, updates)
        return
    if customer_id:
        profile = _get_profile_by_customer(str(customer_id))
        if profile:
            _update_profile(profile["user_id"], updates)


def _subscription_matches_price(subscription: Any) -> bool:
    expected_price_id = _stripe_price_id()
    if not expected_price_id:
        return True
    items = _object_get(_object_get(subscription, "items") or {}, "data") or []
    for item in items:
        price = _object_get(item, "price") or {}
        if _object_get(price, "id") == expected_price_id:
            return True
    return False


def _reconcile_stripe_subscription(user: AuthUser, profile: dict[str, Any]) -> dict[str, Any]:
    customer_id = profile.get("stripe_customer_id")
    if not customer_id:
        return profile
    if _is_paid_profile(profile):
        return profile
    stripe = _stripe_module()
    try:
        subscriptions = stripe.Subscription.list(customer=customer_id, status="all", limit=10)
    except Exception as exc:
        logger.warning("Stripe subscription reconciliation failed for user_id=%s: %s", user.id, exc)
        return profile
    subscription_rows = _object_get(subscriptions, "data") or []
    matching = [subscription for subscription in subscription_rows if _subscription_matches_price(subscription)]
    paid_subscription = next(
        (
            subscription
            for subscription in matching
            if str(_object_get(subscription, "status") or "").lower() in PAID_SUBSCRIPTION_STATUSES
        ),
        None,
    )
    if not paid_subscription:
        status = str(profile.get("subscription_status") or "").lower()
        if status in PAID_SUBSCRIPTION_STATUSES and not _profile_period_is_current(profile):
            updates = {
                "subscription_status": "canceled",
                "current_period_end": profile.get("current_period_end"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            _update_profile(user.id, updates)
            return {**profile, **updates}
        return profile
    updates = _subscription_payload(paid_subscription)
    updates["stripe_customer_id"] = customer_id
    _update_profile(user.id, updates)
    return _get_profile(user.id) or {**profile, **updates}


def _run_job(
    job_id: str,
    input_path: Path,
    pre_stretched: bool = False,
    object_type: str = "Nebula",
    input_mode: str = "Auto",
    stretch_level: str = "Standard",
    siril_deconvolution: bool = False,
    starless_test: bool = False,
    star_setting: str = "Slight Star Reduction",
) -> None:
    with jobs_lock:
        job = jobs[job_id]
        job.status = "running"
        job.stage = "Starting Pipeline"
        job.progress = 1

    def write_log(message: str) -> None:
        with jobs_lock:
            job = jobs[job_id]
            job.log.append(message)
            job.stage, job.progress = _progress_from_log_message(job, message)

    try:
        try:
            analysis = analyze_input_stretch(input_path)
            metrics = ", ".join(f"{key}={value:.4f}" for key, value in analysis.metrics.items())
            if analysis.likely_stretched:
                warning = f"Pre-stretched input warning ({analysis.confidence} confidence): {analysis.message}"
                with jobs_lock:
                    jobs[job_id].warnings.append(warning)
                    jobs[job_id].log.append(warning)
                    jobs[job_id].log.append(f"Input stretch analysis: {metrics}")
            else:
                write_log(f"Input stretch analysis: {analysis.message} ({metrics})")
            write_log(f"Auto input recommendation: {analysis.recommended_mode} ({analysis.recommended_reason}).")
        except Exception as exc:
            write_log(f"Input stretch analysis skipped: {exc}")

        settings = load_settings()
        defaults = default_settings()
        output_root = JOB_OUTPUT_ROOT / job_id
        output_root.mkdir(parents=True, exist_ok=True)
        settings.output_folder = str(output_root)
        mode = input_mode if input_mode in {"Auto", "Linear", "Pre-stretched"} else "Auto"
        if pre_stretched and mode == "Auto":
            mode = "Pre-stretched"
        settings.input_processing_mode = mode
        settings.prestretched_input = mode == "Pre-stretched"
        settings.object_type = object_type if object_type in {"Nebula", "Galaxy", "Star Cluster"} else "Nebula"
        settings.stretch_level = stretch_level if stretch_level in {"Subtle", "Standard", "Aggressive"} else "Standard"
        settings.siril_deconvolution_enabled = bool(siril_deconvolution)
        settings.star_handling_mode = "Slight Star Reduction"
        settings.starless_test_enabled = True
        write_log(f"Selected object type: {settings.object_type}")
        write_log(f"Selected input mode: {settings.input_processing_mode}")
        write_log(f"Selected stretch level: {settings.stretch_level}")
        write_log(f"Siril deconvolution test: {'enabled' if settings.siril_deconvolution_enabled else 'disabled'}")
        write_log(f"Star settings: {settings.star_handling_mode}")
        for attr in ("siril_folder", "deepsnr_folder", "starnet_folder"):
            if not Path(getattr(settings, attr)).exists():
                setattr(settings, attr, getattr(defaults, attr))
        result = run_pipeline(input_path, settings, PipelineMode.FULL, write_log)
        shutil.rmtree(input_path.parent, ignore_errors=True)
    except Exception as exc:
        shutil.rmtree(input_path.parent, ignore_errors=True)
        with jobs_lock:
            jobs[job_id].status = "failed"
            jobs[job_id].error = str(exc)
            jobs[job_id].stage = "Failed"
            jobs[job_id].log.append(f"ERROR: {exc}")
    else:
        with jobs_lock:
            jobs[job_id].status = "finished"
            jobs[job_id].result = result
            jobs[job_id].stage = "Complete"
            jobs[job_id].progress = 100


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _landing_html()


@app.get("/docs", response_class=HTMLResponse)
def docs_page() -> str:
    return _docs_html()


@app.get("/process", response_class=HTMLResponse)
def process_page() -> str:
    return _html()


@app.get("/api/auth/config")
def auth_config() -> dict[str, str | bool]:
    return {
        "configured": _auth_configured(),
        "supabase_url": _supabase_url(),
        "supabase_anon_key": _supabase_anon_key(),
    }


@app.get("/api/billing/status", response_model=None)
def billing_status(user: AuthUser = Depends(require_user)) -> Any:
    try:
        profile = _billing_profile_for(user)
        profile = _reconcile_stripe_subscription(user, profile)
        is_paid = _is_paid_profile(profile)
        return {
            "configured": _billing_configured(),
            "plan": PAID_PLAN_LABEL,
            "is_paid": is_paid,
            "subscription_status": profile.get("subscription_status") or "free",
            "free_credits_remaining": int(profile.get("free_credits_remaining") or 0),
        }
    except Exception as exc:
        if isinstance(exc, HTTPException):
            logger.warning("Billing status unavailable for user_id=%s: %s", user.id, exc.detail)
            detail = exc.detail if isinstance(exc.detail, str) else "Billing status unavailable"
            return JSONResponse({"error": detail}, status_code=exc.status_code)
        logger.exception("Billing status lookup failed for user_id=%s", user.id)
        return JSONResponse({"error": "Billing status unavailable"}, status_code=500)


@app.post("/api/billing/checkout")
def create_billing_checkout(user: AuthUser = Depends(require_user)) -> dict[str, str]:
    profile = _billing_profile_for(user)
    stripe = _stripe_module()
    customer_id = profile.get("stripe_customer_id")
    if not customer_id:
        customer = stripe.Customer.create(
            email=user.email,
            metadata={"user_id": user.id},
        )
        customer_id = customer.id
        _update_profile(user.id, {"stripe_customer_id": customer_id, "email": user.email})

    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=customer_id,
        client_reference_id=user.id,
        line_items=[{"price": _stripe_price_id(), "quantity": 1}],
        allow_promotion_codes=True,
        success_url=os.getenv("STRIPE_SUCCESS_URL", "https://app.deepskyprocessor.com/process?billing=success"),
        cancel_url=os.getenv("STRIPE_CANCEL_URL", "https://app.deepskyprocessor.com/process?billing=cancel"),
        metadata={"user_id": user.id},
        subscription_data={"metadata": {"user_id": user.id}},
    )
    return {"url": session.url}


@app.post("/api/billing/portal")
def create_billing_portal(user: AuthUser = Depends(require_user)) -> dict[str, str]:
    profile = _billing_profile_for(user)
    profile = _reconcile_stripe_subscription(user, profile)
    customer_id = profile.get("stripe_customer_id")
    if not customer_id:
        raise HTTPException(status_code=404, detail="No paid plan found for this account.")
    stripe = _stripe_module()
    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=os.getenv("STRIPE_PORTAL_RETURN_URL", "https://app.deepskyprocessor.com/process"),
    )
    return {"url": session.url}


@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request) -> dict[str, bool]:
    webhook_secret = _stripe_webhook_secret()
    if not webhook_secret:
        raise HTTPException(status_code=503, detail="Stripe webhook secret is not configured.")
    stripe = _stripe_module()
    payload = await request.body()
    signature = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, signature, webhook_secret)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid Stripe webhook signature.") from exc

    event_type = event["type"]
    event_object = event["data"]["object"]
    if event_type == "checkout.session.completed":
        user_id = event_object.get("client_reference_id") or (event_object.get("metadata") or {}).get("user_id")
        customer_id = event_object.get("customer")
        subscription_id = event_object.get("subscription")
        if user_id and customer_id:
            updates: dict[str, Any] = {"stripe_customer_id": customer_id, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if subscription_id:
                subscription = stripe.Subscription.retrieve(subscription_id)
                updates.update(_subscription_payload(subscription))
            _update_profile(user_id, updates)
    elif event_type in {"customer.subscription.created", "customer.subscription.updated"}:
        _apply_subscription_update(event_object)
    elif event_type == "customer.subscription.deleted":
        _apply_subscription_update(event_object, status_override="canceled")
    return {"received": True}


@app.post("/api/uploads/init")
async def init_upload(request: Request, user: AuthUser = Depends(require_user)) -> dict[str, Any]:
    _cleanup_old_temp_files()
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        payload = {}
    filename = Path(str(payload.get("filename") or "upload")).name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_INPUTS:
        raise HTTPException(status_code=400, detail="Upload a FITS or TIFF file.")
    try:
        size = int(payload.get("size") or 0)
    except (TypeError, ValueError):
        size = 0
    if size <= 0:
        raise HTTPException(status_code=400, detail="Upload is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.")

    STAGED_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    upload_id = uuid.uuid4().hex
    upload_dir = STAGED_UPLOAD_ROOT / upload_id
    upload_dir.mkdir(parents=True, exist_ok=False)
    staged_path = upload_dir / filename
    staged_path.touch()
    with jobs_lock:
        staged_uploads[upload_id] = StagedUpload(
            id=upload_id,
            user_id=user.id,
            filename=filename,
            size=size,
            path=staged_path,
        )
    return {"upload_id": upload_id, "chunk_size": CHUNK_UPLOAD_BYTES}


@app.post("/api/uploads/{upload_id}/chunk")
async def upload_chunk(
    upload_id: str,
    request: Request,
    x_upload_offset: int = Header(default=0, alias="X-Upload-Offset"),
    user: AuthUser = Depends(require_user),
) -> dict[str, int | bool]:
    staged = _get_staged_upload(upload_id, user)
    chunk = await request.body()
    if not chunk:
        raise HTTPException(status_code=400, detail="Chunk is empty.")
    if len(chunk) > CHUNK_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload chunk is too large.")
    current_size = staged.path.stat().st_size
    if x_upload_offset != current_size:
        raise HTTPException(status_code=409, detail="Upload offset mismatch. Please retry the upload.")
    if current_size + len(chunk) > staged.size:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.")
    with staged.path.open("ab") as handle:
        handle.write(chunk)
    next_offset = current_size + len(chunk)
    return {"offset": next_offset, "complete": next_offset == staged.size}


@app.post("/api/uploads/{upload_id}/preview")
def create_staged_preview(upload_id: str, user: AuthUser = Depends(require_user)) -> dict[str, str]:
    _cleanup_old_temp_files()
    staged = _require_completed_staged_upload(upload_id, user)
    preview_id = uuid.uuid4().hex
    preview_dir = PREVIEW_ROOT / preview_id
    preview_dir.mkdir(parents=True, exist_ok=False)
    preview_path = preview_dir / "before_preview.png"
    try:
        make_preview(staged.path, preview_path)
    except Exception as exc:
        shutil.rmtree(preview_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Could not create preview: {exc}") from exc
    with jobs_lock:
        previews[preview_id] = user.id
    return {"preview_url": f"/api/previews/{preview_id}?inline=1"}


@app.post("/api/preview")
async def create_preview(
    file: UploadFile = File(...),
    user: AuthUser = Depends(require_user),
) -> dict[str, str]:
    _cleanup_old_temp_files()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_INPUTS:
        raise HTTPException(status_code=400, detail="Upload a FITS or TIFF file.")
    content_length = file.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.")

    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    preview_id = uuid.uuid4().hex
    preview_dir = PREVIEW_ROOT / preview_id
    preview_dir.mkdir(parents=True, exist_ok=False)
    input_path = preview_dir / Path(file.filename or f"preview{suffix}").name
    total = 0
    with input_path.open("wb") as handle:
        while chunk := file.file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                shutil.rmtree(preview_dir, ignore_errors=True)
                raise HTTPException(status_code=413, detail=f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.")
            handle.write(chunk)

    preview_path = preview_dir / "before_preview.png"
    try:
        make_preview(input_path, preview_path)
        input_path.unlink(missing_ok=True)
    except Exception as exc:
        shutil.rmtree(preview_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Could not create preview: {exc}") from exc
    with jobs_lock:
        previews[preview_id] = user.id
    return {"preview_url": f"/api/previews/{preview_id}?inline=1"}


@app.get("/api/previews/{preview_id}")
def get_preview(
    preview_id: str,
    inline: int = 1,
    user: AuthUser = Depends(require_user),
) -> FileResponse:
    if not preview_id or any(ch not in "0123456789abcdef" for ch in preview_id):
        raise HTTPException(status_code=404, detail="Preview not found.")
    with jobs_lock:
        owner_id = previews.get(preview_id)
    if owner_id != user.id:
        raise HTTPException(status_code=404, detail="Preview not found.")
    preview_path = PREVIEW_ROOT / preview_id / "before_preview.png"
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found.")
    disposition = "inline" if inline else "attachment"
    return FileResponse(preview_path, media_type="image/png", filename="before_preview.png", content_disposition_type=disposition)


@app.post("/api/jobs")
async def create_job(
    file: UploadFile | None = File(None),
    upload_id: str = Form(""),
    object_type: str = Form("Nebula"),
    pre_stretched: bool = Form(False),
    input_mode: str = Form("Auto"),
    stretch_level: str = Form("Standard"),
    siril_deconvolution: bool = Form(False),
    starless_test: bool = Form(True),
    star_setting: str = Form(""),
    user: AuthUser = Depends(require_user),
) -> dict[str, str]:
    _cleanup_old_temp_files()
    staged = _require_completed_staged_upload(upload_id, user) if upload_id else None
    filename = staged.filename if staged else Path((file.filename if file else "") or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_INPUTS:
        raise HTTPException(status_code=400, detail="Upload a FITS or TIFF file.")
    if not staged:
        if not file:
            raise HTTPException(status_code=400, detail="Upload a FITS or TIFF file.")
        content_length = file.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.")

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    upload_dir = UPLOAD_ROOT / job_id
    upload_dir.mkdir(parents=True, exist_ok=False)
    input_path = upload_dir / (filename or f"upload{suffix}")

    if staged:
        try:
            shutil.copy2(staged.path, input_path)
        except OSError as exc:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail="Could not prepare uploaded file for processing.") from exc
    else:
        total = 0
        with input_path.open("wb") as handle:
            while chunk := file.file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    handle.close()
                    shutil.rmtree(upload_dir, ignore_errors=True)
                    raise HTTPException(status_code=413, detail=f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.")
                handle.write(chunk)

    try:
        _, credit_consumed = _consume_credit_or_require_subscription(user)
    except HTTPException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    if staged:
        _discard_staged_upload(staged.id)

    with jobs_lock:
        jobs[job_id] = WebJob(id=job_id, user_id=user.id, user_email=user.email, credit_consumed=credit_consumed)
        if pre_stretched:
            jobs[job_id].warnings.append(
                "Pre-stretched mode enabled. DeepSky will skip its stretch/color-stretch stage for this upload."
            )
        if siril_deconvolution:
            jobs[job_id].warnings.append(
                "Experimental Siril deconvolution is enabled for this run. Compare against unchecked results."
            )
        jobs[job_id].warnings.append(
            "Slight Star Reduction is enabled for this run. DeepSky will reduce the star layer while preserving the target."
        )
    executor.submit(
        _run_job,
        job_id,
        input_path,
        pre_stretched,
        object_type,
        input_mode,
        stretch_level,
        siril_deconvolution,
        starless_test,
        star_setting,
    )
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, user: AuthUser = Depends(require_user)) -> dict[str, Any]:
    _cleanup_old_temp_files()
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or job.user_id != user.id:
            raise HTTPException(status_code=404, detail="Job not found.")
        return _job_response(job)


@app.get("/api/jobs/{job_id}/file/{kind}")
def get_job_file(
    job_id: str,
    kind: str,
    inline: bool = False,
    user: AuthUser = Depends(require_user),
):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or job.user_id != user.id or not job.result:
            raise HTTPException(status_code=404, detail="File not ready.")
        mapping = {
            "before_preview": job.result["before_preview"],
            "after_preview": job.result["after_preview"],
            "png": job.result.get("png", job.result["after_preview"]),
            "final": job.result["final"],
        }
        path = mapping.get(kind)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="File not found.")
    media_type = "image/png" if Path(path).suffix.lower() == ".png" else "application/octet-stream"
    disposition = "inline" if inline else "attachment"
    return FileResponse(path, media_type=media_type, filename=Path(path).name, content_disposition_type=disposition)


@app.get("/api/jobs/{job_id}/files", response_class=PlainTextResponse)
def list_job_files(job_id: str, user: AuthUser = Depends(require_user)) -> str:
    raise HTTPException(status_code=404, detail="Job file listing is disabled.")
