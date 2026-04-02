# EPL Prediction Engine — Deployment Guide

---

## PART 1: Run Locally (Test Before Deploying)

### Prerequisites

Before anything else, install:

1. **Julia 1.10+**
   Download the installer from https://julialang.org/downloads/
   On Windows: run the `.exe`. On Mac: run the `.dmg`. On Linux: run the installer script.
   After install, confirm it works:
   ```
   julia --version
   # Expected: julia version 1.10.x
   ```

2. **A modern browser** (Chrome, Firefox, Edge — any of them)

3. **Git** (optional, only needed for cloud deployment later)

---

### Step 1 — Unzip the project

Unzip `epl-predictor.zip` to a folder. You should see:
```
epl-predictor/
├── backend/
│   ├── server.jl
│   ├── model.jl
│   ├── features.jl
│   ├── monte_carlo.jl
│   ├── betting.jl
│   ├── Project.toml
│   ├── Dockerfile
│   └── data/
│       └── epl_final.csv
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── api.js
│       ├── app.js
│       └── charts.js
├── render.yaml
├── railway.toml
├── netlify.toml
├── vercel.json
└── README.md
```

---

### Step 2 — Install Julia packages

Open a terminal (PowerShell on Windows, Terminal on Mac/Linux).

```bash
# Navigate to the backend folder
cd path/to/epl-predictor/backend

# Start Julia in the project environment and install all packages
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
```

⏳ This takes **5–15 minutes** on first run. It downloads and compiles:
- `HTTP.jl` (the web server)
- `MLJ.jl` + `MLJXGBoostInterface.jl` (the ML framework + XGBoost)
- `CSV.jl`, `DataFrames.jl` (data loading)
- `JSON3.jl` (API responses)
- and others

You'll see a lot of progress bars. That's normal. Once it returns to the prompt, you're ready.

---

### Step 3 — Start the Julia backend server

Still in the `backend/` folder:

```bash
julia --threads=auto --project=. server.jl
```

You will see output like:
```
[ Info: === EPL Prediction Engine — Julia Backend ===
[ Info: Loading/training model from .../data/epl_final.csv ...
[ Info: Loading dataset from .../epl_final.csv...
[ Info: Computing ELO ratings (9380 matches)...
[ Info: Computing rolling averages...
[ Info: Training XGBoostClassifier with grid-search CV (max_depth 3..9)...
[ Info: Test accuracy: 68.XX%
[ Info: Model saved to .../data/footballmodel.jlso
[ Info: Starting HTTP server on http://0.0.0.0:8080
[ Info: Open frontend/index.html in your browser to use the UI.
```

⏳ **First run takes ~5–10 minutes** to train the XGBoost model and save it.
✅ **Subsequent runs take ~30 seconds** because it loads the saved model.

**Keep this terminal open.** The server must be running for the frontend to work.

---

### Step 4 — Test the API directly

Open a new terminal tab and run:

```bash
# Health check
curl http://localhost:8080/api/health

# Expected response:
# {"status":"ok","model":"ready","teams":47,"timestamp":"..."}

# Get teams list
curl http://localhost:8080/api/teams

# Test a prediction
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{"home":"Arsenal","away":"Chelsea","confidence":false}'
```

If you see JSON responses, the backend is working.

---

### Step 5 — Open the frontend

Simply open the file in your browser:

**Option A (easiest):** Double-click `frontend/index.html` in your file explorer.

**Option B (from terminal):**
```bash
# Mac
open path/to/epl-predictor/frontend/index.html

# Windows
start path/to/epl-predictor/frontend/index.html

# Linux
xdg-open path/to/epl-predictor/frontend/index.html
```

You should see:
- The dark dashboard loads
- The status dot turns **green** and shows "Connected · 47 teams · Model Ready"
- All four tabs (Match Predictor, Betting EV, Season Simulator, ELO Rankings) work

---

### Common Local Issues

| Problem | Fix |
|---------|-----|
| `julia: command not found` | Julia isn't in your PATH. On Mac/Linux: add `export PATH="$PATH:/path/to/julia/bin"` to your `~/.bashrc`. On Windows: reinstall Julia and check "Add Julia to PATH" |
| `Error: Package X not found` | Run `julia --project=. -e "using Pkg; Pkg.instantiate()"` again from the `backend/` folder |
| `Port 8080 already in use` | Something else is using port 8080. Kill it: `lsof -ti:8080 | xargs kill` (Mac/Linux) or change `PORT` in `server.jl` |
| Status dot stays **red** | The server isn't running or crashed. Check the terminal running `server.jl` for error messages |
| `CORS error` in browser console | You're on `file://` — this is fine for local use. If you see CORS errors on localhost, make sure the server is running |
| Browser shows blank page | Open DevTools (F12) → Console tab and look for errors. Usually a missing file or JS syntax error |

---

---

## PART 2: Deploy Backend on Render (Recommended — Free Tier)

Render is the easiest option for Julia because it supports Docker natively and the free tier works.

### Why Render over Railway for Julia?
- Render free tier: 750 hours/month (enough for one always-on service)
- Railway free tier: $5 credit/month (gets used up quickly with a heavy Julia image)
- Render has better logs visibility for debugging startup issues

---

### Step 1 — Push your code to GitHub

```bash
# Create a new repo on github.com first, then:

cd path/to/epl-predictor        # The ROOT folder, not backend/
git init
git add .
git commit -m "Initial commit — EPL Prediction Engine"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/epl-predictor.git
git push -u origin main
```

Your repo structure on GitHub should look exactly like the unzipped folder.

---

### Step 2 — Create a Render account

Go to https://render.com and sign up with GitHub (it's free, no credit card needed for the free tier).

---

### Step 3 — Create a new Web Service on Render

1. In the Render dashboard, click **"New +"** → **"Web Service"**
2. Connect your GitHub account if prompted
3. Select your **`epl-predictor`** repository
4. Render will detect the `render.yaml` file automatically

   **If it doesn't auto-configure, set these manually:**
   | Setting | Value |
   |---------|-------|
   | Name | `epl-prediction-api` |
   | Environment | `Docker` |
   | Dockerfile Path | `./backend/Dockerfile` |
   | Docker Context | `./backend` |
   | Instance Type | Free |

5. Click **"Create Web Service"**

---

### Step 4 — Wait for the build

Render will:
1. Pull your GitHub repo
2. Build the Docker image (installs Julia + all packages — **takes 10–20 min on first build**)
3. Train the XGBoost model on startup (**takes another 5–10 min**)
4. Show "Live" in green when done

Watch the logs in the Render dashboard. You should eventually see:
```
Starting HTTP server on http://0.0.0.0:8080
```

Your API will be live at a URL like:
```
https://epl-prediction-api.onrender.com
```

Test it:
```bash
curl https://epl-prediction-api.onrender.com/api/health
```

---

### ⚠️ Important: Free Tier Spin-Down

On Render's free tier, **the service sleeps after 15 minutes of no traffic**. The next request will wake it up but takes ~30 seconds. To avoid this:
- Upgrade to the Starter plan ($7/month)
- Or use a free uptime pinger like https://uptimerobot.com to ping `/api/health` every 10 minutes

---

---

## PART 3: Deploy Backend on Railway (Alternative)

Railway is faster to set up but costs money once your free credit runs out.

### Step 1 — Create a Railway account

Go to https://railway.app and sign up with GitHub.

---

### Step 2 — Create a new project

1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose your `epl-predictor` repository
4. Railway reads `railway.toml` automatically and finds the `backend/Dockerfile`

---

### Step 3 — Set environment variables

In Railway's Variables tab, add:
| Variable | Value |
|----------|-------|
| `PORT` | `8080` |
| `JULIA_NUM_THREADS` | `2` |

Railway also auto-injects its own `PORT` variable — the `server.jl` code reads it with `get(ENV, "PORT", "8080")`.

---

### Step 4 — Get your public URL

In Railway, go to **Settings → Networking → Generate Domain**.
You'll get a URL like:
```
https://epl-predictor-production.up.railway.app
```

Test it:
```bash
curl https://epl-predictor-production.up.railway.app/api/health
```

---

---

## PART 4: Deploy Frontend on Netlify (Recommended)

### Step 1 — Update the API URL in the frontend

Before deploying the frontend, you MUST tell it where the backend lives.

Open `frontend/index.html` and find this line near the top:
```html
<script>window.EPL_API_BASE = "";</script>
```

Replace the empty string with your deployed backend URL:
```html
<!-- If using Render: -->
<script>window.EPL_API_BASE = "https://epl-prediction-api.onrender.com/api";</script>

<!-- If using Railway: -->
<script>window.EPL_API_BASE = "https://epl-predictor-production.up.railway.app/api";</script>
```

Save the file, commit, and push to GitHub:
```bash
git add frontend/index.html
git commit -m "Set production API URL"
git push
```

---

### Step 2 — Create a Netlify account

Go to https://netlify.com and sign up with GitHub (free).

---

### Step 3 — Deploy

**Option A — Drag and drop (fastest, no Git needed):**
1. Go to https://app.netlify.com
2. Drag and drop your entire `frontend/` folder onto the Netlify dashboard
3. Done — it's live in 30 seconds

**Option B — Connect GitHub (auto-deploys on every push):**
1. Click **"Add new site"** → **"Import an existing project"**
2. Connect GitHub and select your `epl-predictor` repo
3. Set these build settings:
   | Setting | Value |
   |---------|-------|
   | Base directory | `frontend` |
   | Build command | *(leave empty)* |
   | Publish directory | `frontend` |
4. Click **"Deploy site"**

Netlify reads `netlify.toml` from the repo root automatically. Your frontend will be live at:
```
https://your-site-name.netlify.app
```

---

---

## PART 5: Deploy Frontend on Vercel (Alternative)

### Step 1 — Update the API URL

Same as Netlify Step 1 above — update `index.html` with your backend URL.

---

### Step 2 — Create a Vercel account

Go to https://vercel.com and sign up with GitHub (free).

---

### Step 3 — Deploy

1. Click **"Add New"** → **"Project"**
2. Import your `epl-predictor` GitHub repo
3. Vercel reads `vercel.json` automatically
4. Set Framework Preset to **"Other"**
5. Set Root Directory to **`frontend`**
6. Click **"Deploy"**

Your frontend will be live at:
```
https://epl-predictor.vercel.app
```

---

---

## PART 6: Final Production Checklist

Work through this in order:

```
[ ] 1. Local: Julia installed and `julia --version` works
[ ] 2. Local: `Pkg.instantiate()` completed without errors
[ ] 3. Local: `server.jl` starts and shows "Model Ready"
[ ] 4. Local: curl http://localhost:8080/api/health returns {"status":"ok"}
[ ] 5. Local: frontend/index.html opens, green dot, prediction works

[ ] 6. GitHub: repo pushed with all files including Dockerfile and data/epl_final.csv
[ ] 7. Backend: Render/Railway build completes (check logs for "Starting HTTP server")
[ ] 8. Backend: curl https://YOUR-BACKEND.onrender.com/api/health returns {"status":"ok"}

[ ] 9. Frontend: index.html updated with production API URL (window.EPL_API_BASE = "...")
[ ] 10. Frontend: committed and pushed to GitHub
[ ] 11. Frontend: Netlify/Vercel deployed, site URL is live
[ ] 12. Production: Open frontend URL, green dot appears, prediction works end-to-end
```

---

## Architecture Reminder (what talks to what)

```
Browser (Netlify / Vercel)
        │
        │  HTTP POST /api/predict  (CORS-enabled)
        │
        ▼
Julia Server (Render / Railway)   ← runs model.jl, features.jl, betting.jl, monte_carlo.jl
        │
        │  reads
        ▼
epl_final.csv  (bundled inside Docker image)
```

The frontend and backend are completely separate deployments. The only connection between them is the URL in `window.EPL_API_BASE`.

---

## Cost Summary

| Service | Free Tier | Paid |
|---------|-----------|------|
| Render (backend) | 750 hrs/month, sleeps after 15 min idle | Starter $7/mo — always on |
| Railway (backend) | $5 credit/month | Usage-based ~$5-15/mo |
| Netlify (frontend) | 100GB bandwidth, unlimited | Pro $19/mo (you won't need this) |
| Vercel (frontend) | 100GB bandwidth, unlimited | Pro $20/mo (you won't need this) |

**Recommended stack:** Render (Starter $7/mo) + Netlify (free) = **$7/month total**
for a fully production-ready, always-on EPL prediction system.
