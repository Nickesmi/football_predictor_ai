# Football Predictor AI Agent

A statistical analysis and probabilistic pattern mining engine designed to predict the outcomes of football (soccer) matches. **This is not an ML-first project.** It focuses on extracting high-confidence deterministic insights from historical performance by analyzing specific home and away context.

## 🎯 Primary Objective

For any given match: **Home Team A vs Away Team B in League L**

The system will:

1. **Analyze Conditional History:**
   - Evaluates *all Home matches* of Team A in League L.
   - Evaluates *all Away matches* of Team B in League L.

2. **Extract High-Confidence Patterns:**
   Identifies recurring events across the following markets:
   - **BTTS** (Both Teams to Score)
   - **Over/Under Goals** (Full Time & Half Time - e.g., O/U 1.5, O/U 2.5)
   - **Corners** (e.g., Over 8.5)
   - **Cards** (Yellow/Red limits)
   - **Win/Draw/Loss**
   - **Team to Score** (Clean sheets, failed to score)
   - **First Half Events** (Goals, corners, cards in 1H)

3. **Compute Probabilities & Confidence:**
   - *Most common factors for the Home Team (Team A)*
   - *Most common factors for the Away Team (Team B)*
   - *Their intersection* (Where patterns from both teams align)
   - *Confidence Percentage* for the predicted outcomes based on historical occurrences.
   - **Crucially:** Predict the **Final Match Result (1X2)** and **Half-Time Result (1X2)** alongside their computed confidence percentages.

## Project Structure

This project uses deterministic, probabilistic pattern mining + statistical filtering to identify the best betting angles or match predictions.
- **Data Fetching:** Fetching league-specific home and away matches from a Football API.
- **Statistical Filtering Engine:** Computing percentages and filtering events that surpass a high-confidence threshold (e.g., > 80% occurrence).
- **Pattern Intersection Finder:** Cross-referencing Team A's home trends with Team B's away trends.
- **LLM/AI Formatter (Optional):** Using a Large Language Model strictly to convert the discovered *statistical* intersections into an easily readable Natural Language report, *not* to guess the outcome.

## 🌐 Global Deployment (GitHub Education)

The repository ships with a fully automated CI/CD pipeline that publishes the frontend to **GitHub Pages** (free, globally available) and the backend to **Render.com** (free tier — included in the GitHub Student Developer Pack).

### Live URLs (after setup)
| Service | URL |
|---------|-----|
| Frontend | `https://nickesmi.github.io/football_predictor_ai/` |
| Backend API | `https://football-predictor-api.onrender.com` |

> **Custom domain:** GitHub Education (via Namecheap) gives you a free `.me` domain for one year. Once you have it, add it under **Settings → Pages → Custom domain** in GitHub and set `VITE_BASE_PATH` to `/` (see step 5 below).

---

### Step-by-step setup

#### 1 — Enable GitHub Pages
1. Go to **Settings → Pages** in this repository.
2. Set **Source** to `GitHub Actions`.
3. Save — no branch selection is needed; the workflow handles everything.

#### 2 — Deploy the backend to Render
1. Sign in to [render.com](https://render.com) with your GitHub account.
2. Click **New + → Blueprint** and select this repository.  
   Render detects `render.yaml` and creates the service automatically.
3. In the Render dashboard, open the service → **Environment** and add:
   ```
   APIFOOTBALL_API_KEY = <your key from api-football.com>
   ```
4. Copy the public URL shown at the top of the Render service page  
   (e.g. `https://football-predictor-api.onrender.com`).

#### 3 — Add GitHub Secrets
In this repository go to **Settings → Secrets and variables → Actions → New repository secret** and add:

| Secret name | Value |
|-------------|-------|
| `VITE_API_URL` | The Render URL from step 2 (e.g. `https://football-predictor-api.onrender.com`) |
| `RENDER_DEPLOY_HOOK_URL` | *(optional)* Render deploy-hook URL so backend re-deploys on every push |
| `VITE_BASE_PATH` | `/football_predictor_ai/` — or `/` if you have a custom domain |

#### 4 — Push to `main`
Any push to the `main` branch automatically:
- Builds the React frontend with the correct API URL and base path.
- Deploys the built site to GitHub Pages.
- Triggers a Render backend re-deploy (if the deploy hook is set).

#### 5 — (Optional) Custom domain
1. Get your free `.me` domain from the [GitHub Student Developer Pack](https://education.github.com/pack) (via Namecheap).
2. Add a `CNAME` DNS record pointing to `nickesmi.github.io`.
3. In **Settings → Pages → Custom domain**, enter your domain and enable **Enforce HTTPS**.
4. Set the `VITE_BASE_PATH` secret to `/` so asset paths are correct.

---

## Local Development

### Backend
```bash
# 1. Clone and enter the repo
git clone https://github.com/Nickesmi/football_predictor_ai.git
cd football_predictor_ai

# 2. Create a virtual environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Copy the example env file and add your API key
cp .env.example .env
# Edit .env and set APIFOOTBALL_API_KEY=<your key>

# 4. Start the API server
uvicorn api.main:app --reload
# API available at http://127.0.0.1:8000
```

### Frontend
```bash
cd frontend
npm install

# Point at the local backend (default – no variable needed)
npm run dev
# UI available at http://localhost:5173
```
