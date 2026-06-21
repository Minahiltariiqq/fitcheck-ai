# Dockerfile for deploying the FitCheck backend to Hugging Face Spaces.
# You only need this file when you deploy. Locally you just run python.

FROM python:3.11-slim

WORKDIR /code

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (including the trained model.pth)
COPY . .

# HF Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

# Use gunicorn in production (not Flask's dev server)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "app.main:app"]
