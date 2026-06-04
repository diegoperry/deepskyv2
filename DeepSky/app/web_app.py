from __future__ import annotations

import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .input_analysis import analyze_input_stretch
from .image_io import SUPPORTED_INPUTS, make_preview
from .pipeline import PipelineMode, run_pipeline
from .settings import APP_ROOT, PROJECT_ROOT, default_settings, load_settings


UPLOAD_ROOT = PROJECT_ROOT / "outputs" / "web_uploads"
PREVIEW_ROOT = UPLOAD_ROOT / "previews"
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
app.mount("/static", StaticFiles(directory=APP_ROOT / "app" / "static"), name="static")


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
            <a class="secondary" href="#results">See before and after</a>
          </div>
          <div class="trust">
            <div><strong>50 MB</strong><span>FITS/TIFF uploads</span></div>
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
              <img class="after" src="/static/landing/galaxy_after.png" alt="After galaxy processing">
              <div class="handle"></div>
            </div>
            <input type="range" min="0" max="100" value="50" aria-label="Compare galaxy before and after">
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
          <article class="feature"><h3>Downloadable results</h3><p>Save the final TIFF and processing log after each run.</p></article>
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
          <details open><summary>What files can I upload?</summary><p>DeepSky currently accepts FITS, FIT, FTS, TIF, and TIFF files up to 50 MB.</p></details>
          <details><summary>Does it work with SeeStar files?</summary><p>Yes. DeepSky is being tuned around real SeeStar-style FITS and TIFF uploads, including both linear and already-stretched files.</p></details>
          <details><summary>Is this catalog photometric color calibration?</summary><p>When Siril PCC is available and the file has enough metadata, catalog-based star color calibration can be used. Otherwise DeepSky uses pixel-based background and star balancing.</p></details>
          <details><summary>Will it replace manual processing?</summary><p>No. It is meant to produce a strong first processed result quickly, especially for users who do not want to spend hours tuning multiple astronomy tools.</p></details>
          <details><summary>Do I need an account?</summary><p>No account or paid tier is required yet. The current version is focused on letting users process an image.</p></details>
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
    DeepSky Built By
    <a href="https://www.linkedin.com/in/diego-perry-64a609240/" target="_blank" rel="noreferrer">Diego Perry</a>
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
      grid-template-columns: 132px minmax(0, 1fr) 48px;
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
    .footer a {
      color: #9cbcff;
      text-decoration: none;
      font-weight: 800;
    }
    .footer a:hover { text-decoration: underline; }
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
      <p class="subtitle">Turn raw deep-sky captures into processed results in minutes. Upload a FITS or TIFF file, run the full pipeline, and compare the before and after side by side.</p>
      <label id="drop" class="drop">
        <input id="file" type="file" accept=".fits,.fit,.fts,.tif,.tiff" />
        <div>
          <strong>Drag & drop your astrophotography file</strong>
          <span id="fileName">Supports FITS and TIFF formats up to 50 MB</span>
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
        <button id="run" class="cta" disabled>Run Full Pipeline</button>
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
    <pre class="log" id="log">Processing log will appear here.</pre>
    <footer class="footer">
      DeepSky Built By
      <a href="https://www.linkedin.com/in/diego-perry-64a609240/" target="_blank" rel="noreferrer">Diego Perry</a>
    </footer>
  </main>
  <script>
    const drop = document.getElementById("drop");
    const fileInput = document.getElementById("file");
    const fileName = document.getElementById("fileName");
    const run = document.getElementById("run");
    const objectType = document.getElementById("objectType");
    const inputMode = document.getElementById("inputMode");
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
    const logEl = document.getElementById("log");
    const beforeFrame = document.getElementById("beforeFrame");
    const afterFrame = document.getElementById("afterFrame");
    const downloads = document.getElementById("downloads");
    let selectedFile = null;
    let activeJob = null;
    let previewRequest = 0;

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

    function postFormJson(url, data, { onUploadProgress, onServerWait } = {}) {
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        let lastPercent = 0;
        xhr.open("POST", url);
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
            reject(new Error(xhr.responseText || `Request failed with status ${xhr.status}`));
          }
        };
        xhr.onerror = () => reject(new Error("Network error while uploading file."));
        xhr.send(data);
      });
    }

    async function setFile(file) {
      selectedFile = file;
      const requestId = ++previewRequest;
      resetProgress();
      fileName.textContent = file ? file.name : "Supports FITS and TIFF formats up to 50 MB";
      const tooLarge = file && file.size > 50 * 1024 * 1024;
      run.disabled = !file || tooLarge;
      beforeFrame.innerHTML = file ? '<span class="empty">Loading preview</span>' : '<span class="empty">No image selected</span>';
      afterFrame.innerHTML = '<span class="empty">Waiting for processing</span>';
      downloads.innerHTML = "";
      statusEl.textContent = tooLarge ? "File is too large. Maximum upload size is 50 MB." : file ? "Preparing preview..." : "Choose a file to begin.";
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
        const data = new FormData();
        data.append("file", file);
        showUploadProgress("Uploading preview");
        const preview = await postFormJson("/api/preview", data, {
          onUploadProgress: (percent) => setProgress(uploadProgressFill, uploadProgressValue, percent),
          onServerWait: () => {
            uploadProgressValue.textContent = "Done";
            showPreviewProgress();
          },
        });
        if (requestId !== previewRequest) return;
        setProgress(uploadProgressFill, uploadProgressValue, 100);
        setProgress(previewProgressFill, previewProgressValue, 100);
        beforeFrame.innerHTML = `<img src="${preview.preview_url}&t=${Date.now()}" alt="Before preview">`;
        statusEl.textContent = "Ready to run full pipeline.";
        setTimeout(() => {
          if (requestId === previewRequest) progressPanel.hidden = true;
        }, 700);
      } catch (error) {
        if (requestId !== previewRequest) return;
        progressPanel.hidden = true;
        beforeFrame.innerHTML = '<span class="empty">Preview unavailable</span>';
        statusEl.textContent = "Ready to run full pipeline.";
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

    fileInput.addEventListener("change", () => { void setFile(fileInput.files[0]); });
    drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("drag"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      drop.classList.remove("drag");
      if (event.dataTransfer.files.length) void setFile(event.dataTransfer.files[0]);
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
      if (job.before_preview && !beforeFrame.querySelector("img")) {
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
      showUploadProgress("Uploading job");
      const data = new FormData();
      data.append("file", selectedFile);
      data.append("object_type", objectType.value);
      data.append("input_mode", inputMode.value);
      data.append("pre_stretched", inputMode.value === "Pre-stretched" ? "true" : "false");
      let job;
      try {
        job = await postFormJson("/api/jobs", data, {
          onUploadProgress: (percent) => setProgress(uploadProgressFill, uploadProgressValue, percent),
          onServerWait: () => {
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


def _run_job(
    job_id: str,
    input_path: Path,
    pre_stretched: bool = False,
    object_type: str = "Nebula",
    input_mode: str = "Auto",
) -> None:
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
            write_log(f"Auto input recommendation: {analysis.recommended_mode} ({analysis.recommended_reason}).")
        except Exception as exc:
            write_log(f"Input stretch analysis skipped: {exc}")

        settings = load_settings()
        defaults = default_settings()
        settings.output_folder = str(PROJECT_ROOT / "outputs")
        mode = input_mode if input_mode in {"Auto", "Linear", "Pre-stretched"} else "Auto"
        if pre_stretched and mode == "Auto":
            mode = "Pre-stretched"
        settings.input_processing_mode = mode
        settings.prestretched_input = mode == "Pre-stretched"
        settings.object_type = object_type if object_type in {"Nebula", "Galaxy", "Star Cluster"} else "Nebula"
        write_log(f"Selected object type: {settings.object_type}")
        write_log(f"Selected input mode: {settings.input_processing_mode}")
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
    return _landing_html()


@app.get("/process", response_class=HTMLResponse)
def process_page() -> str:
    return _html()


@app.post("/api/preview")
async def create_preview(file: UploadFile = File(...)) -> dict[str, str]:
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
    except Exception as exc:
        shutil.rmtree(preview_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Could not create preview: {exc}") from exc
    return {"preview_url": f"/api/previews/{preview_id}?inline=1"}


@app.get("/api/previews/{preview_id}")
def get_preview(preview_id: str, inline: int = 1) -> FileResponse:
    if not preview_id or any(ch not in "0123456789abcdef" for ch in preview_id):
        raise HTTPException(status_code=404, detail="Preview not found.")
    preview_path = PREVIEW_ROOT / preview_id / "before_preview.png"
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found.")
    disposition = "inline" if inline else "attachment"
    return FileResponse(preview_path, media_type="image/png", filename="before_preview.png", content_disposition_type=disposition)


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    object_type: str = Form("Nebula"),
    pre_stretched: bool = Form(False),
    input_mode: str = Form("Auto"),
) -> dict[str, str]:
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
        if pre_stretched:
            jobs[job_id].warnings.append(
                "Pre-stretched mode enabled. DeepSky will skip its stretch/color-stretch stage for this upload."
            )
    executor.submit(_run_job, job_id, input_path, pre_stretched, object_type, input_mode)
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
