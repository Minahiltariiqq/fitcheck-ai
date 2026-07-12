# FitCheck AI

AI Fashion & Outfit Recommendation Engine.

## What it does

Upload a photo of a clothing item and FitCheck AI classifies it, scores its style (0–100) and suggests what to pair it with based on the occasion. It's built around a trained image classification model that recognizes clothing categories and evaluates style compatibility, wrapped in a simple web interface for uploading and viewing results.

## Features

- Clothing item classification from an uploaded photo
- Style scoring (0–100) for the item
- Outfit pairing suggestions based on occasion
- Web based upload and results interface

## Stack

- **Model:** MobileNetV2 (transfer learning, PyTorch)
- **Backend:** Flask + Python
- **Frontend:** React + Vite + Tailwind CSS
- **Trained on:** Fashion Product Images (Small) — Kaggle

## Quick start

See `SETUP_GUIDE.md` for the full step-by-step walkthrough.

## License

MIT
