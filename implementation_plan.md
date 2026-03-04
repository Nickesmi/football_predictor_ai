# Goal Description

The user requested a "deluxe and luxury frontend" that displays daily football matches and shows the prediction odds (statistical factor report) when a match is clicked.

To achieve this, we need to:
1.  **Backend API (FastAPI)**: Expose our existing `APIFootballFetcher`, `PatternAnalyzer`, `FactorAnalyzer`, and `ReportFormatter` through a fast, lightweight REST API.
2.  **Frontend Web App (React + Vite)**: Build a premium UI (dark mode, glassmorphism, gold accents, smooth animations) using modern web technologies to consume the API.

## User Review Required

> [!IMPORTANT]
> The Python codebase currently operates purely as a CLI script. To power a frontend, we need to wrap our core logic in an API server. I propose using **FastAPI** because it's asynchronous, very fast, and extremely easy to set up alongside our existing Pydantic-like domain models (dataclasses).
>
> For the frontend, I propose using **React with Vite** and **TailwindCSS** (for rapid, premium styling).
>
> **Do you approve of using FastAPI for the backend API and React/Vite for the frontend?**

## Proposed Changes

### Backend API (`api/`) REST Layer

We will create a new package for the API layer to keep the CLI (`main.py`) intact but allow the app to be served over HTTP.

#### [NEW] `api/main.py`
The FastAPI application entry point.
- GET `/api/leagues`: Returns supported luxury leagues (e.g., Premier League, La Liga, Serie A).
- GET `/api/fixtures/today`: Fetches today's fixtures for the supported leagues.
- GET `/api/analysis/match/{fixture_id}`: Runs the full analysis pipeline (Pattern -> Factor -> Report) for a specific match and returns the JSON dictionary format we built in Issue #4.

#### [MODIFY] `requirements.txt`
Add `fastapi`, `uvicorn` (for the server), and `pydantic` (for API request/response validation).

### Frontend Web App (`frontend/`)

We will initialize a new Vite/React project in a `frontend/` directory.

#### [NEW] `frontend/index.html` & `frontend/src/main.jsx`
Standard Vite initializers.

#### [NEW] `frontend/src/index.css`
Global styles. This is where we will define the "luxury" CSS variables: deep blacks, dark greys, subtle gold gradients (`#D4AF37`), and glassmorphic utility classes (semi-transparent backgrounds with backdrop blur).

#### [NEW] `frontend/src/App.jsx`
The main application shell. Will include a sleek header, date selector (defaulting to today), and a grid of Match Cards.

#### [NEW] `frontend/src/components/MatchCard.jsx`
A luxurious card displaying the home and away teams, match time, and league. Includes hover animations (slight lift, gold glow). Clicking it expands or opens a modal.

#### [NEW] `frontend/src/components/PredictionModal.jsx`
A beautifully styled modal or expanded section that displays the `MatchFactorReport` data we get from the backend.
- Displays the "Combined %" with sleek progress bars.
- Uses color-coded confidence indicators (🟢 Very High, 🟡 Medium, etc.).
- Includes the mandatory disclaimer clearly.

## Verification Plan

### Automated Tests
- We already have 216 passing tests for the core logic.
- We will add `pytest` unit tests for the FastAPI route handlers (`api/main.py`) using `TestClient` to ensure they correctly return HTTP 200 and the expected JSON payload.

### Manual Verification
1.  Run the backend server: `uvicorn api.main:app --reload`
2.  Run the frontend dev server: `npm run dev`
3.  Open the browser to the frontend URL.
4.  Verify the UI looks "deluxe and luxury" (dark theme, gold accents, smooth hover states).
5.  Click on a match and verify the loading state and subsequent display of the full statistical intersection report from the backend.
