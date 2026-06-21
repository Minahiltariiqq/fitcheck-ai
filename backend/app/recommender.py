"""
backend/app/recommender.py

Generates a 0-100 style score and outfit pairing suggestions.
Rule-based for now (uses color theory + category pairing).
You can replace this with a learned model later.
"""
from PIL import Image
from io import BytesIO
import colorsys
import numpy as np

# What pairs well with what (basic styling rules)
PAIRS = {
    'Tshirts':       ['Jeans', 'Shorts', 'Trousers', 'Casual Shoes', 'Sneakers'],
    'Shirts':        ['Jeans', 'Trousers', 'Formal Shoes', 'Casual Shoes'],
    'Jeans':         ['Tshirts', 'Shirts', 'Casual Shoes', 'Sneakers'],
    'Trousers':      ['Shirts', 'Formal Shoes'],
    'Shorts':        ['Tshirts', 'Sneakers', 'Casual Shoes'],
    'Dresses':       ['Heels', 'Flats', 'Handbags'],
    'Kurtas':        ['Jeans', 'Trousers', 'Sandals', 'Flats'],
    'Sarees':        ['Heels', 'Sandals', 'Handbags'],
    'Casual Shoes':  ['Tshirts', 'Jeans', 'Shorts'],
    'Formal Shoes':  ['Shirts', 'Trousers'],
    'Sneakers':      ['Tshirts', 'Jeans', 'Shorts'],
    'Heels':         ['Dresses', 'Sarees'],
    'Watches':       ['Shirts', 'Tshirts', 'Trousers'],
    'Handbags':      ['Dresses', 'Kurtas', 'Sarees'],
    'Tops':          ['Jeans', 'Skirts', 'Trousers'],
    'Sandals':       ['Kurtas', 'Sarees', 'Shorts'],
    'Flats':         ['Dresses', 'Kurtas'],
    'Sunglasses':    ['Tshirts', 'Shirts'],
    'Belts':         ['Trousers', 'Jeans', 'Shirts'],
    'Wallets':       ['Trousers', 'Jeans'],
}

# Which items suit which occasion
OCCASION_MAP = {
    'Formal': ['Shirts', 'Trousers', 'Formal Shoes', 'Watches', 'Belts'],
    'Casual': ['Tshirts', 'Jeans', 'Casual Shoes', 'Sneakers', 'Shorts'],
    'Party':  ['Dresses', 'Heels', 'Handbags', 'Watches'],
    'Ethnic': ['Kurtas', 'Sarees', 'Sandals', 'Flats'],
    'Sports': ['Tshirts', 'Shorts', 'Sneakers'],
}


def get_dominant_color(image_bytes: bytes):
    """Return the average color of the image as (R, G, B) and a color name."""
    img = Image.open(BytesIO(image_bytes)).convert('RGB').resize((50, 50))
    pixels = np.array(img).reshape(-1, 3)
    avg = pixels.mean(axis=0).astype(int)
    return tuple(avg.tolist()), _color_name(avg)


def _color_name(rgb):
    """Convert an RGB triple to a human color name (Red, Blue, etc.)."""
    r, g, b = [c / 255 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.2:
        return 'Black'
    if s < 0.15 and v > 0.85:
        return 'White'
    if s < 0.15:
        return 'Grey'
    h_deg = h * 360
    if h_deg < 15 or h_deg >= 345:
        return 'Red'
    if h_deg < 45:
        return 'Orange'
    if h_deg < 65:
        return 'Yellow'
    if h_deg < 170:
        return 'Green'
    if h_deg < 260:
        return 'Blue'
    if h_deg < 290:
        return 'Purple'
    return 'Pink'


def style_score(predictions, dominant_color):
    """
    Compute a 0-100 style score.
    60 points: model's confidence in its top prediction (clearer item = better)
    40 points: color saturation + value (more 'styled' colors score higher)
    """
    top_confidence = predictions[0][1]
    confidence_score = top_confidence * 60

    rgb = dominant_color
    r, g, b = [c / 255 for c in rgb]
    _, s, v = colorsys.rgb_to_hsv(r, g, b)
    color_score = (s * 0.6 + v * 0.4) * 40

    return round(confidence_score + color_score, 1)


def suggest_matches(top_category, occasion=None):
    """Return up to 5 category suggestions that pair with the detected item."""
    matches = PAIRS.get(top_category, [])
    if occasion and occasion in OCCASION_MAP:
        occ_items = OCCASION_MAP[occasion]
        filtered = [m for m in matches if m in occ_items]
        matches = filtered if filtered else occ_items
    return matches[:5]


def _rationale(score, category, color):
    """One-line plain-English explanation of the score."""
    if score >= 80:
        return f"Strong choice — the {color.lower()} {category.lower()} reads as polished and intentional."
    if score >= 60:
        return f"Solid base. The {color.lower()} {category.lower()} works; pair thoughtfully to elevate it."
    return f"Workable item. Pair carefully — the {color.lower()} {category.lower()} needs the right context."


def build_response(image_bytes, predictions, occasion=None):
    """Bundle everything the frontend needs into one dict."""
    dom_rgb, color_name = get_dominant_color(image_bytes)
    top_category = predictions[0][0]
    score = style_score(predictions, dom_rgb)
    matches = suggest_matches(top_category, occasion)

    return {
        'detected_category': top_category,
        'top_predictions': [
            {'label': l, 'confidence': round(c, 3)} for l, c in predictions
        ],
        'dominant_color': {'rgb': list(dom_rgb), 'name': color_name},
        'style_score': score,
        'rationale': _rationale(score, top_category, color_name),
        'suggested_matches': matches,
    }
