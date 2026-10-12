# DataLens

Upload CSV files, explore statistics and charts, find data-quality problems, clean data without losing the original, and **chat with an AI assistant about your own dataset** (powered by the Groq API).

Everything is one small Flask app: `app.py` (all Python), `templates/index.html`, `static/css/style.css`, `static/js/app.js` and `static/image/logo.png`. No front-end build step.

```
DataLens/
├── app.py                 # routes, models, auth, CSV parsing, stats, charts, cleaning, PDF, Groq AI, admin
├── requirements.txt
├── README.md
├── .gitignore
├── templates/index.html   # auth, dashboard, analysis workspace, AI chat, settings, admin
└── static/
    ├── css/style.css
    ├── js/app.js
    └── image/logo.png
```

## Features

- Sign up / log in, password change, server-side sessions, CSRF protection, rate limiting, account lock-out after repeated failures.
- CSV upload (drag and drop or picker) with validation: encoding (UTF-8, UTF-8 BOM, UTF-16, Windows-1252 fallback), delimiter detection (`, ; tab |`), duplicate/empty header repair, row/column/size limits.
- Overview, summary statistics (numeric, categorical, date), missing values, duplicates, constant and mixed-type columns, IQR outliers.
- Charts: histogram, category frequency, group comparison, time series, scatter (with trend line), correlation heatmap, missing-value chart.
- Data explorer: server-side search, filters, sorting, column selection, pagination, row detail.
- Non-destructive cleaning (previewed, undoable, resettable): remove duplicates/missing rows, fill numbers or text, convert types, rename/remove columns, filter rows.
- Insights that separate **observations** (measured) from **interpretations**.
- AI chat grounded in your data: the server calculates answers with pandas first, then Groq explains them.
- Exports: original CSV, cleaned CSV, summary statistics CSV, missing-value CSV, PDF report (spreadsheet-formula-safe).
- Admin dashboard: users, active users, uploads, analyses, AI requests (success/failure/rate-limit), tokens (only when Groq reports them), downloads. All from real database records.

Core analysis works with **no** AI key. If `GROQ_API_KEY` is missing, the chat page explains that AI needs server configuration.

## Requirements

- Python 3.10 or newer (developed and tested on Python 3.12).
- A Groq API key for AI chat (optional): <https://console.groq.com/keys>

## Quick start (Windows, PowerShell)

```powershell
cd DataLens
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Required for production, optional for local dev (a key is generated into instance/ in dev):
$env:SECRET_KEY = "replace-with-a-long-random-string"
# First administrator (created once, only if it does not exist yet):
$env:INITIAL_ADMIN_USERNAME = "admin"
$env:INITIAL_ADMIN_EMAIL = "you@example.com"
$env:INITIAL_ADMIN_PASSWORD = "a-strong-password-of-10+chars"
# AI chat (optional):
$env:GROQ_API_KEY = "gsk_..."

python app.py
```

Open <http://127.0.0.1:5000>. If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process RemoteSigned` first.

macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`, then `export VAR=value` instead of `$env:VAR`.

You can also put the variables in a `.env` file next to `app.py` (it is git-ignored and loaded automatically):

```ini
SECRET_KEY=replace-with-a-long-random-string
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.1-8b-instant
INITIAL_ADMIN_USERNAME=admin
INITIAL_ADMIN_EMAIL=you@example.com
INITIAL_ADMIN_PASSWORD=change-me-please-123
```

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `SECRET_KEY` | Signs sessions and CSRF tokens. **Required when `DATALENS_ENV=production`.** | random key stored in `instance/` (dev only) |
| `DATALENS_ENV` | `production` enables secure cookies and HSTS and refuses to start without `SECRET_KEY`. | `development` |
| `DATABASE_URL` | SQLAlchemy URL. SQLite by default. | `sqlite:///instance/datalens.db` |
| `UPLOAD_FOLDER` | Private directory for uploaded CSVs (outside `static/`). | `./private_uploads` |
| `MAX_CONTENT_LENGTH` | Max upload size in bytes. | `26214400` (25 MB) |
| `MAX_ROWS` / `MAX_COLS` / `MAX_CELLS` | Dataset limits. | `200000` / `200` / `5000000` |
| `MAX_DATASETS_PER_USER` | Datasets per account. | `50` |
| `GROQ_API_KEY` | Groq key. Read only on the server. | unset (AI disabled) |
| `GROQ_MODEL` | Model id to use. | `llama-3.1-8b-instant` |
| `AI_DAILY_LIMIT` | AI questions per user per UTC day. | `50` |
| `AI_PER_MINUTE` | AI questions per user per minute. | `8` |
| `AI_MAX_TOKENS` / `AI_TIMEOUT` | Answer length cap / request timeout (seconds). | `900` / `30` |
| `AI_CONTEXT_CHARS` / `AI_SAMPLE_ROWS` / `AI_HISTORY_MESSAGES` | Size bounds for what is sent to Groq. | `14000` / `8` / `8` |
| `INITIAL_ADMIN_USERNAME`, `INITIAL_ADMIN_EMAIL`, `INITIAL_ADMIN_PASSWORD` | One-time admin bootstrap. | unset |
| `RATELIMIT_STORAGE_URI` | Rate-limit store, for example `redis://host:6379`. | `memory://` |
| `TRUST_PROXY` | Set to `1` behind a reverse proxy (Render, Nginx) so client IPs and HTTPS are detected. | off |
| `HOST` / `PORT` | Used by `python app.py`. | `127.0.0.1` / `5000` |

## Groq setup and model choice

1. Create a key at <https://console.groq.com/keys> and set `GROQ_API_KEY` on the server. It is never sent to the browser, never returned by an API route and never logged.
2. Pick a model your account can use with `GROQ_MODEL`. Model availability and rate limits are **account dependent and change over time**; check <https://console.groq.com/docs/models> and your account's limits page. `llama-3.1-8b-instant` is the default (a faster, cheaper option).
3. If the model is retired or not enabled for your account, chat shows "The configured AI model is unavailable" and the admin should change `GROQ_MODEL`.

How the assistant avoids guessing numbers:

- It never receives your whole CSV. It receives a bounded summary: dimensions, column types, statistics, missing and duplicate counts, correlations, a few evenly spaced sample rows (emails, long numbers and columns named like phone/password/address are redacted) and the currently selected chart.
- For calculation questions ("total units by region") the model first returns a **JSON plan** limited to a fixed list of safe operations (aggregate, value counts, correlation, trend, count-where, top rows, …). The server validates every column name and runs the calculation with pandas. The exact results are then passed back for explanation. **No model-generated code, SQL or shell command is ever executed.**
- AI replies are rendered as text with a small safe Markdown renderer (no raw HTML).
- Rate limits, quotas, invalid keys, unavailable models, timeouts and network errors are each reported clearly; a failed request is never shown or counted as an answer, and 429s are not retried automatically.

## Database and admin bootstrap

Tables are created automatically at start-up (`python app.py` or Gunicorn). To initialise manually: `flask --app app init-db`. There are no migrations; if you change models in a future version, back up first.

Admin options (choose one):

1. **Environment bootstrap.** Set the three `INITIAL_ADMIN_*` variables and start the app once. The admin is created only if that username/email does not already exist, the password is **never reset** on later starts, and the admin **must change the password at first login**. Passwords under 10 characters are rejected. Remove the password variable after the first start.
2. **Interactive CLI.** `flask --app app create-admin` prompts for username, email and password (hidden input).

Public sign-up can never create an administrator. Every `/api/admin/*` route checks the role on the server.

## Production (Gunicorn)

```bash
export DATALENS_ENV=production
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export TRUST_PROXY=1
gunicorn app:app --workers 2 --threads 4 --timeout 60 --bind 0.0.0.0:${PORT:-8000}
```

Notes:

- Terminate HTTPS at your proxy/host; production mode sets `Secure` cookies and HSTS.
- Rate-limit counters live in memory per worker. For strict limits across workers/instances set `RATELIMIT_STORAGE_URI` to Redis (`pip install redis`).
- The first request after a restart re-parses a dataset from disk; parsed data is cached in memory (`CACHE_SIZE`, default 6 datasets per worker).

### Persistent storage (important)

SQLite and uploaded files must live on **persistent disk**. On hosts with an ephemeral filesystem (including Render's free web services) they are **lost on every restart or redeploy**.

- Mount a persistent disk and point at it: `DATABASE_URL=sqlite:////var/data/datalens.db` and `UPLOAD_FOLDER=/var/data/uploads`.
- Or use a managed database (`DATABASE_URL=postgresql+psycopg://…`, add `psycopg[binary]` to requirements) plus a persistent disk for uploads.
- SQLite suits a single instance; use `--workers 2` or fewer.

### Deploy to Render

1. Push this folder to a Git repository (the `.gitignore` keeps secrets, databases and uploads out).
2. New → **Web Service** → connect the repo. Runtime: Python. Build command: `pip install -r requirements.txt`. Start command: `gunicorn app:app --workers 2 --threads 4 --timeout 60`.
3. Add a **Disk** (paid plans) mounted at `/var/data`.
4. Set environment variables: `DATALENS_ENV=production`, `SECRET_KEY`, `TRUST_PROXY=1`, `DATABASE_URL=sqlite:////var/data/datalens.db`, `UPLOAD_FOLDER=/var/data/uploads`, `GROQ_API_KEY`, `GROQ_MODEL`, and the `INITIAL_ADMIN_*` trio for the first deploy.
5. Health check path: `/healthz`.

A Flask backend cannot run on static-only hosting such as Netlify. Deploy it to a Python-capable service (Render, Railway, Fly.io, a VPS, …).

## Data handling, backups and deletion

- Uploaded CSVs are stored unmodified in `UPLOAD_FOLDER/<user id>/<random id>.csv`. Names are generated, so user-supplied filenames never touch the filesystem. Files are served only through authenticated, ownership-checked routes.
- **Retention policy:** data is kept until the user deletes it. Deleting a dataset removes its file, cleaning steps and its AI conversations immediately. Deleting an account removes all its datasets, files and conversations; anonymous analytics events (counts only, no content) are kept with the user detached. Deleting chat history removes the messages permanently.
- **Backups:** stop writes or use SQLite's online backup, then copy the database and the whole upload folder together, for example `sqlite3 datalens.db ".backup 'backup.db'"` and `cp -r uploads uploads-backup`. Restore by putting both back in place.
- Analytics events store only type, time, user id, dataset id and safe counts (rows, token totals, error codes). Never CSV contents or chat text.

## Behaviour notes

- Missing-value tokens: empty cells and `NA`, `N/A`, `NaN`, `null`, `none`, `nil`, `-`, `--`, `?` are treated as missing. Leading/trailing spaces are trimmed.
- A column is numeric only if **every** filled cell is a number (thousands separators and currency symbols `$ € £ ₦ ¥` are understood; codes with leading zeros stay text). A column that is 90-99% numeric is flagged "mixed" and kept as text so nothing is silently dropped; convert it in the Clean tab (non-numbers then become empty, shown in the preview).
- Date columns are detected when all filled values look like dates. Ambiguous `03/04/2024` is read month-first.
- The **original CSV download returns your exact uploaded bytes**. Other exports prefix text starting with `= + - @` with an apostrophe so spreadsheets never run them as formulas.
- Downloads are counted only after the file was generated successfully.
- "Active users": distinct users with a qualifying event (sign-in, upload, analysis, chart, cleaning, AI request, download) in the last 1 / 7 / 30 days. "Rows processed": sum of row counts at upload time.

## Testing

Manual checklist:

1. Start the app; sign up; log out and log in; change the password.
2. Upload a CSV; open the Overview, Statistics, Data quality and Insights tabs; compare a few numbers with your own spreadsheet.
3. Draw each chart type; try a deliberately unsuitable one (a histogram of a text column) and confirm the explanatory message.
4. In Explorer, search, sort, filter and open a row; use "Duplicates only".
5. In Clean, preview and apply a step, then Undo and Reset; download the original and confirm it is unchanged.
6. Without `GROQ_API_KEY`, open AI chat and confirm the explanatory banner. With a key, ask "Compare total units by region" style questions.
7. Log in as admin and confirm the analytics match what you did; confirm a normal user cannot open `/#/admin-analytics` (server returns 403 for the API).

What was verified for this release (automated, using Flask's test client, Chromium via Playwright, Gunicorn and a *simulated* Groq client):

- Authentication, hashing, CSRF, lock-out, rate limiting, admin bootstrap and role checks.
- Upload validation (empty, binary, malformed, oversized, wrong type, BOM, semicolon, Windows-1252, duplicate headers, path-traversal names).
- Statistics, missing values, duplicates, outliers and chart data compared with independent pandas/numpy calculations.
- Ownership isolation on every dataset and conversation route; pagination, search, filters, sorting; cleaning operations, undo/reset and original-file integrity.
- CSV formula neutralisation, PDF generation, download counting (including that a failed generation is not counted).
- AI context contents, redaction, bounded history, safe plan execution (unknown columns and an injected "eval" operation ignored), token accounting, and handling of 429, auth, unavailable-model, timeout, network and 5xx errors with simulated provider exceptions.
- Browser UI at 1440 px and 320/375/390/430/768 px: no horizontal overflow, working drawer and chat layout, no CSP violations, all icons rendered, AI output rendered as inert text.
- Production mode (secure cookies, HSTS, missing `SECRET_KEY` refusal), restart persistence, Gunicorn start-up.

**Not verified:** live calls to the real Groq service (the test environment has no access to it), so real model answer quality, real rate-limit headers and real token counts are untested; the real CDN delivery of Chart.js/Lucide/Google Fonts (local copies were substituted); and deployment on Render itself.

## Troubleshooting

| Problem | Fix |
|---|---|
| `RuntimeError: SECRET_KEY must be set` | Set `SECRET_KEY` when `DATALENS_ENV=production`. |
| "AI chat needs to be configured" | Set `GROQ_API_KEY` on the server and restart. |
| "The configured AI model is unavailable" | Set `GROQ_MODEL` to a model your Groq account can use. |
| "rate limit was reached" | Wait and retry; check your Groq limits or lower `AI_PER_MINUTE`. |
| Charts or icons missing | The page loads Chart.js and Lucide from jsDelivr; check the network/firewall. |
| Logged out on every restart / uploads vanished | Persistent disk missing; see *Persistent storage*. |
| Cannot log in as admin | The bootstrap only creates the account once; use `flask --app app create-admin` with a different username. |
| "Too many requests" | Authentication and AI limits are intentional; wait a minute. |
| Upload says "Most rows have a different number of fields" | Check the delimiter and quoting, or re-export the CSV. |
| Windows: `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process RemoteSigned`. |

## Security summary

Hashed passwords (scrypt), CSRF tokens on every write, `HttpOnly`/`SameSite` (and `Secure` in production) cookies, per-IP and per-user rate limits, server-side authorisation and ownership checks, private upload storage with generated names and path-traversal guards, strict Content-Security-Policy (no inline scripts or styles), no raw HTML from uploads or AI output, spreadsheet-formula-safe exports, sessions invalidated on password change, and logs that avoid keys, datasets and chat text.
