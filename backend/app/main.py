"""
backend/app/main.py

Flask server. Run locally with:
    python -m app.main

This is the file that receives requests from the React frontend.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from app.inference import predict
from app.recommender import build_response

app = Flask(__name__)

# Allow the React frontend to call us from a different origin.
# In production, replace "*" with your actual Vercel URL for safety.
CORS(app, origins=["*"])

# Cap upload size at 8 MB (most phone photos are 2-4 MB)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024


@app.route('/health', methods=['GET'])
def health():
    """Simple endpoint to check the server is alive."""
    return jsonify({'status': 'ok', 'service': 'fitcheck-ai'})


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main endpoint. Expects:
      - multipart/form-data
      - 'image' file field (JPEG or PNG)
      - 'occasion' text field (optional: Casual, Formal, Party, Ethnic, Sports)
    """
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image provided. Send as multipart/form-data field "image".'
        }), 400

    occasion = request.form.get('occasion')  # optional
    image_bytes = request.files['image'].read()

    if not image_bytes:
        return jsonify({'error': 'Empty image file.'}), 400

    try:
        predictions = predict(image_bytes, top_k=3)
        result = build_response(image_bytes, predictions, occasion=occasion)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Image is too large. Max size is 8 MB.'}), 413


if __name__ == '__main__':
    # debug=True auto-reloads on code changes — great for development
    # host='0.0.0.0' lets your phone on the same wifi access it too
    app.run(host='0.0.0.0', port=5001, debug=True)
