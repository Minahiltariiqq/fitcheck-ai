# FitCheck AI

AI Fashion & Outfit Recommendation Engine — UCP BSCS Final Year Project, Spring 2026.

**Team:** Minahil Malik (L1F23BSCS0431) · Maham Iqbal (L1F23BSCS0411)

## What it does
Upload a photo of a clothing item → FitCheck AI classifies it, scores its style (0–100), and suggests what to pair it with based on occasion.

## Stack
- **Model:** MobileNetV2 (transfer learning, PyTorch)
- **Backend:** Flask + Python
- **Frontend:** React + Vite + Tailwind CSS
- **Trained on:** Fashion Product Images (Small) — Kaggle

## Quick start
See `SETUP_GUIDE.md` for the full step-by-step walkthrough.

## Live demo
*(Add your Vercel URL here after deployment)*

## Architecture
```
[ React frontend ]  ──HTTPS──►  [ Flask backend ]  ──►  [ MobileNetV2 model ]
   (Vercel)                       (HF Spaces)              (model.pth)
```
