# DeepSky V2

DeepSky is an astrophotography processing app for FITS and TIFF files.

It includes:

- A PySide6 desktop app
- A FastAPI web app prototype
- FITS/TIFF loading and 16-bit TIFF output
- Siril, DeepSNR, and StarNet CLI integration
- Before/after previews
- Galaxy background refinement and color-preserving processing

The Windows CLI tool folders are bundled under `tools/` and tracked with Git LFS:

- `tools/siril`
- `tools/deepsnr`
- `tools/starnet`

Clone with Git LFS enabled so the actual binaries download instead of pointer files.

## Local Web Preview

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\windows\install.ps1
powershell -ExecutionPolicy Bypass -File .\deploy\windows\run_web.ps1
```

Then open:

```text
http://127.0.0.1:8000
```

## Web Auth

The web app uses Supabase Auth. Set these environment variables before starting the FastAPI server:

```powershell
$env:SUPABASE_URL="https://your-project.supabase.co"
$env:SUPABASE_ANON_KEY="your-supabase-anon-key"
```

Unsigned users can view the process page, but previews, processing jobs, job status, and downloads require a valid Supabase session.

## Web Billing

Run `deploy/supabase/billing.sql` in the Supabase SQL editor to create the billing profile table and free-credit function.

Stripe billing uses one $15/month subscription Price. Set these server-side environment variables before starting the FastAPI server:

```powershell
$env:SUPABASE_SERVICE_ROLE_KEY="your-supabase-service-role-key"
$env:STRIPE_SECRET_KEY="sk_live_or_test_..."
$env:STRIPE_WEBHOOK_SECRET="whsec_..."
$env:STRIPE_PRICE_ID="price_..."
```

Optional overrides:

```powershell
$env:STRIPE_SUCCESS_URL="https://app.deepskyprocessor.com/process?billing=success"
$env:STRIPE_CANCEL_URL="https://app.deepskyprocessor.com/process?billing=cancel"
```

Configure the Stripe webhook endpoint as:

```text
https://app.deepskyprocessor.com/api/stripe/webhook
```

Each signed-in user starts with 3 free image credits. After those credits are used, `/api/jobs` requires an active Stripe subscription. Paid users have unlimited image processing.

## Desktop App

```powershell
cd DeepSky
python -m app.main
```

## Windows Build

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
```

## Production Deployment

See `DEPLOYMENT.md` for the Windows VPS + Cloudflare Tunnel deployment path.
