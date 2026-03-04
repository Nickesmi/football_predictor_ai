# Football Predictor AI Agent

An AI-powered application that analyzes and predicts the outcomes of football (soccer) matches by analyzing the latest form of both the home and away teams. It generates insights such as the most common factors, scoring percentages, and overall predictions based on historical match data.

## Features

- **Home Team Analysis:** Evaluates the last 10 **home** games played by the home team.
- **Away Team Analysis:** Evaluates the last 10 **away** games played by the away team.
- **Common Factor Extraction:** Identifies the highest probability events based on recent matches:
  - e.g., "Over 8.5 corners"
  - e.g., "Both Teams to Score (BTTS) = Yes"
  - e.g., "Home Win (Home team hasn't lost in 10 home matches)"
  - e.g., "Away team scored in every game"
  - e.g., "Away team conceded in every game"
- **AI-Powered Insights:** Uses Large Language Models (LLMs) to generate human-readable explanations and percentage-based probabilities for predicted outcomes.

## Project Structure

This project is divided into several modules:
- Data fetching (e.g., integrating with a Football API for live and historical match stats).
- Data processing and feature engineering (calculating corners, goals, win/loss streaks).
- AI/LLM Integration (formatting data to prompt an LLM or using an ML model to extract common factors).
- Output representation (CLI or Web UI).

## Setup & Installation

*(Coming soon when the codebase is initialized)*

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

## Development Tasks / Issues

The project development is tracked using GitHub Issues. Key implementation phases include:

1. **Issue 1: Data Collection & API Integration**
2. **Issue 2: Data Processing & Feature Engineering**
3. **Issue 3: Most Common Factor Analysis Engine**
4. **Issue 4: AI/LLM Integration for Natural Language Output**
5. **Issue 5: Testing and Validation**
