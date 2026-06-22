"""
backend/app/main.py

Flask server with Claude AI integration for style rationale.
Run locally with: python -m app.main
"""
import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from app.inference import predict
from app.recommender import build_response

app = Flask(__name__)
CORS(app, origins=["*"])
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')


def get_ai_rationale(category, color_name, score, occasion, confidence):
    """Call Claude API to generate a unique, intelligent style comment."""
    if not ANTHROPIC_API_KEY:
        return None

    occasion_str = f" for a {occasion} occasion" if occasion else ""
    confidence_pct = round(confidence * 100)

    prompt = f"""You are a concise, knowledgeable fashion stylist. 
A user uploaded a photo of their {category.lower()}.

Details:
- Item: {category}
- Dominant color: {color_name}
- Occasion context: {occasion or 'General / unspecified'}
- Style score: {score}/100
- Model confidence: {confidence_pct}%

Write ONE sentence (max 20 words) of specific, actionable style advice for this exact item{occasion_str}.
Do NOT start with "I" or mention the score or confidence.
Be direct, specific, and helpful. Sound like a real stylist, not a template."""

    try:
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': ANTHROPIC_API_KEY,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json',
            },
            json={
                'model': 'claude-haiku-4-5-20251001',
                'max_tokens': 80,
                'messages': [{'role': 'user', 'content': prompt}]
            },
            timeout=8
        )
        if response.status_code == 200:
            return response.json()['content'][0]['text'].strip()
    except Exception:
        pass
    return None


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'fitcheck-ai',
        'ai_enabled': bool(ANTHROPIC_API_KEY)
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided. Send as multipart/form-data field "image".'}), 400

    occasion = request.form.get('occasion')
    image_bytes = request.files['image'].read()

    if not image_bytes:
        return jsonify({'error': 'Empty image file.'}), 400

    try:
        predictions = predict(image_bytes, top_k=3)
        top_category = predictions[0][0]
        top_confidence = predictions[0][1]

        # Try Claude AI rationale first, fall back to templates
        from app.recommender import get_color_palette, style_score
        palette = get_color_palette(image_bytes)
        color_name = palette['dominant']['name']
        dom_rgb = tuple(palette['dominant']['rgb'])
        score = style_score(predictions, top_category, dom_rgb)

        ai_rationale = get_ai_rationale(top_category, color_name, score, occasion, top_confidence)

        result = build_response(image_bytes, predictions, occasion=occasion, ai_rationale=ai_rationale)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Image is too large. Max size is 8 MB.'}), 413


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
