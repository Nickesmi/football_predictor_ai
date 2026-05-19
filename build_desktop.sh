#!/bin/bash

# Exit on error
set -e

echo "======================================"
echo "Building Football Predictor AI Desktop"
echo "======================================"

# 1. Install Python build dependencies
echo "=> Installing pywebview and pyinstaller..."
pip install pywebview pyinstaller

# 2. Build the React Frontend
echo "=> Building React frontend..."
cd frontend
npm install
npm run build
cd ..

# 3. Clean previous builds
echo "=> Cleaning up old builds..."
rm -rf build dist FootballPredictor.spec

# 4. Bundle with PyInstaller
echo "=> Packaging as Desktop Application..."
# Note: For MacOS, the separator is colon (:) for --add-data. On Windows it's semicolon (;).
# Since the user is on a Mac, we use colon.
pyinstaller --name "Football Predictor" \
  --windowed \
  --add-data "frontend/dist:frontend/dist" \
  --add-data "api:api" \
  --add-data "src:src" \
  --add-data "models:models" \
  --collect-all uvicorn \
  --collect-all fastapi \
  --collect-all pydantic \
  --collect-all xgboost \
  --icon=NONE \
  desktop.py

echo "======================================"
echo "✅ Build Complete!"
echo "Your application can be found in the 'dist' folder:"
echo "macOS: open dist/Football\ Predictor.app"
echo "======================================"
