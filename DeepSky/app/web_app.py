from __future__ import annotations

import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from .input_analysis import analyze_input_stretch
from .image_io import SUPPORTED_INPUTS
from .pipeline import PipelineMode, run_pipeline
from .settings import PROJECT_ROOT, default_settings, load_settings


UPLOAD_ROOT = PROJECT_ROOT / "outputs" / "web_uploads"
MAX_WORKERS = 1
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_MB = MAX_UPLOAD_BYTES // (1024 * 1024)


@dataclass
class WebJob:
    id: str
    status: str = "queued"
    log: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    result: dict[str, Path] | None = None
    error: str | None = None


app = FastAPI(title="DeepSky")
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
jobs: dict[str, WebJob] = {}
jobs_lock = Lock()


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
    .log {
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #05080d;
      color: #93a9c9;
      font: 12px Consolas, ui-monospace, monospace;
      padding: 14px;
      min-height: 90px;
      max-height: 180px;
      overflow: auto;
      white-space: pre-wrap;
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
    <section class="hero">
      <div class="pill">● DeepSky Astrophotography Pipeline v1.0</div>
      <h1>Process the <span class="cosmos">Cosmos</span></h1>
      <p class="subtitle">Automated deep-sky processing. Drop a FITS or TIFF file, run the full pipeline, and compare the before and after.</p>
      <label id="drop" class="drop">
        <input id="file" type="file" accept=".fits,.fit,.fts,.tif,.tiff" />
        <div>
          <strong>Drag & drop your astrophotography file</strong>
          <span id="fileName">Supports FITS and TIFF formats up to 50 MB</span>
        </div>
      </label>
      <button id="run" class="cta" disabled>Run Full Pipeline</button>
      <div id="warning" class="warning"></div>
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
    <pre class="log" id="log">Processing log will appear here.</pre>
  </main>
  <script>
    const drop = document.getElementById("drop");
    const fileInput = document.getElementById("file");
    const fileName = document.getElementById("fileName");
    const run = document.getElementById("run");
    const statusEl = document.getElementById("status");
    const warningEl = document.getElementById("warning");
    const logEl = document.getElementById("log");
    const beforeFrame = document.getElementById("beforeFrame");
    const afterFrame = document.getElementById("afterFrame");
    const downloads = document.getElementById("downloads");
    let selectedFile = null;
    let activeJob = null;

    function setFile(file) {
      selectedFile = file;
      fileName.textContent = file ? file.name : "Supports FITS and TIFF formats up to 50 MB";
      const tooLarge = file && file.size > 50 * 1024 * 1024;
      run.disabled = !file || tooLarge;
      beforeFrame.innerHTML = file ? '<span class="empty">Preview will appear after upload</span>' : '<span class="empty">No image selected</span>';
      afterFrame.innerHTML = '<span class="empty">Waiting for processing</span>';
      downloads.innerHTML = "";
      statusEl.textContent = tooLarge ? "File is too large. Maximum upload size is 50 MB." : file ? "Ready to run full pipeline." : "Choose a file to begin.";
      warningEl.style.display = "none";
      warningEl.textContent = "";
    }

    fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
    drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("drag"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      drop.classList.remove("drag");
      if (event.dataTransfer.files.length) setFile(event.dataTransfer.files[0]);
    });

    async function poll(jobId) {
      const res = await fetch(`/api/jobs/${jobId}`);
      const job = await res.json();
      statusEl.textContent = job.status === "running" ? "Processing..." : job.status;
      if (job.warnings && job.warnings.length) {
        warningEl.style.display = "block";
        warningEl.textContent = job.warnings.join(" ");
      }
      logEl.textContent = job.log?.length ? job.log.join("\\n") : "Processing...";
      logEl.scrollTop = logEl.scrollHeight;
      if (job.before_preview) {
        beforeFrame.innerHTML = `<img src="${job.before_preview}&t=${Date.now()}" alt="Before preview">`;
      }
      if (job.after_preview) {
        afterFrame.innerHTML = `<img src="${job.after_preview}&t=${Date.now()}" alt="After preview">`;
      }
      if (job.status === "finished") {
        statusEl.textContent = "Processing complete.";
        downloads.innerHTML = `
          <a href="${job.final}" download>Download final TIFF</a>
          <a href="${job.log_file}" download>Download log</a>
          <a href="${job.job_folder}" target="_blank" rel="noreferrer">Open job files</a>
        `;
        run.disabled = false;
        return;
      }
      if (job.status === "failed") {
        statusEl.textContent = "Processing failed.";
        run.disabled = false;
        return;
      }
      setTimeout(() => poll(jobId), 1800);
    }

    run.addEventListener("click", async () => {
      if (!selectedFile) return;
      run.disabled = true;
      statusEl.textContent = "Uploading...";
      logEl.textContent = "Uploading file...";
      downloads.innerHTML = "";
      afterFrame.innerHTML = '<span class="empty">Processing</span>';
      const data = new FormData();
      data.append("file", selectedFile);
      const res = await fetch("/api/jobs", { method: "POST", body: data });
      if (!res.ok) {
        statusEl.textContent = await res.text();
        run.disabled = false;
        return;
      }
      const job = await res.json();
      activeJob = job.id;
      poll(activeJob);
    });
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
    }
    if job.result:
        payload.update(
            {
                "before_preview": f"/api/jobs/{job.id}/file/before_preview?inline=1",
                "after_preview": f"/api/jobs/{job.id}/file/after_preview?inline=1",
                "final": f"/api/jobs/{job.id}/file/final",
                "log_file": f"/api/jobs/{job.id}/file/log",
                "job_folder": f"/api/jobs/{job.id}/files",
            }
        )
    return payload


def _run_job(job_id: str, input_path: Path) -> None:
    with jobs_lock:
        job = jobs[job_id]
        job.status = "running"

    def write_log(message: str) -> None:
        with jobs_lock:
            jobs[job_id].log.append(message)

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
        except Exception as exc:
            write_log(f"Input stretch analysis skipped: {exc}")

        settings = load_settings()
        defaults = default_settings()
        settings.output_folder = str(PROJECT_ROOT / "outputs")
        for attr in ("siril_folder", "deepsnr_folder", "starnet_folder"):
            if not Path(getattr(settings, attr)).exists():
                setattr(settings, attr, getattr(defaults, attr))
        result = run_pipeline(input_path, settings, PipelineMode.FULL, write_log)
    except Exception as exc:
        with jobs_lock:
            jobs[job_id].status = "failed"
            jobs[job_id].error = str(exc)
            jobs[job_id].log.append(f"ERROR: {exc}")
    else:
        with jobs_lock:
            jobs[job_id].status = "finished"
            jobs[job_id].result = result


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _html()


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)) -> dict[str, str]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_INPUTS:
        raise HTTPException(status_code=400, detail="Upload a FITS or TIFF file.")
    content_length = file.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.")

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    upload_dir = UPLOAD_ROOT / job_id
    upload_dir.mkdir(parents=True, exist_ok=False)
    input_path = upload_dir / Path(file.filename or f"upload{suffix}").name
    total = 0
    with input_path.open("wb") as handle:
        while chunk := file.file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                handle.close()
                shutil.rmtree(upload_dir, ignore_errors=True)
                raise HTTPException(status_code=413, detail=f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.")
            handle.write(chunk)

    with jobs_lock:
        jobs[job_id] = WebJob(id=job_id)
    executor.submit(_run_job, job_id, input_path)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return _job_response(job)


@app.get("/api/jobs/{job_id}/file/{kind}")
def get_job_file(job_id: str, kind: str, inline: bool = False):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or not job.result:
            raise HTTPException(status_code=404, detail="File not ready.")
        mapping = {
            "before_preview": job.result["before_preview"],
            "after_preview": job.result["after_preview"],
            "final": job.result["final"],
            "log": job.result["log"],
        }
        path = mapping.get(kind)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="File not found.")
    media_type = "image/png" if Path(path).suffix.lower() == ".png" else "application/octet-stream"
    disposition = "inline" if inline else "attachment"
    return FileResponse(path, media_type=media_type, filename=Path(path).name, content_disposition_type=disposition)


@app.get("/api/jobs/{job_id}/files", response_class=PlainTextResponse)
def list_job_files(job_id: str) -> str:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or not job.result:
            raise HTTPException(status_code=404, detail="Job not ready.")
        folder = Path(job.result["job_folder"])
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Job folder not found.")
    return "\n".join(str(path) for path in sorted(folder.iterdir()))
