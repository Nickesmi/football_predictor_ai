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

## Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nickesmi/football_predictor_ai.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables (e.g., API keys for Football Data and LLM providers).
4. Run the application:
   ```bash
   python main.py
   ```
