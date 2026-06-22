"""
backend/app/recommender.py

Handles:
* Color extraction (center crop, background removal, 3-color palette)
* Style scoring (confidence + category + color)
* Match suggestions (category + occasion aware)
"""

from PIL import Image
from io import BytesIO
import colorsys
import numpy as np

# ---------------------------------------------------------------------------
# Pairing rules
# ---------------------------------------------------------------------------

PAIRS = {
    'Tshirts':       ['Slim Jeans', 'Joggers', 'Shorts', 'Sneakers', 'Casual Shoes'],
    'Shirts':        ['Chinos', 'Dress Trousers', 'Jeans', 'Oxford Shoes', 'Leather Belt'],
    'Tops':          ['High-waist Jeans', 'Midi Skirt', 'Trousers', 'Heels', 'Flats'],
    'Kurtas':        ['Churidar', 'Straight Trousers', 'Kolhapuri Sandals', 'Dupatta'],
    'Casual Shoes':  ['Slim Jeans', 'Chinos', 'Tshirts', 'Shorts', 'Rolled Cuffs'],
    'Sports Shoes':  ['Joggers', 'Dry-fit Shorts', 'Track Pants', 'Sports Socks'],
    'Heels':         ['Slim Trousers', 'Midi Dress', 'Pencil Skirt', 'Clutch Bag'],
    'Handbags':      ['Casual Jeans & Top', 'Midi Dress', 'Kurta Set', 'Blazer'],
    'Watches':       ['Formal Shirt', 'Casual Tshirt', 'Blazer', 'Chinos'],
    'Sunglasses':    ['Casual Tshirt & Jeans', 'Summer Dress', 'Linen Shirt'],
}

OCCASION_PAIRS = {
    'Formal':  ['Dress Trousers', 'Oxford Shoes', 'Leather Belt', 'Formal Watch', 'Blazer'],
    'Casual':  ['Slim Jeans', 'White Sneakers', 'Casual Watch', 'Canvas Tote', 'Sunglasses'],
    'Party':   ['Statement Heels', 'Clutch Bag', 'Bold Earrings', 'Fitted Dress', 'Red Lip'],
    'Ethnic':  ['Churidar', 'Dupatta', 'Kolhapuri Sandals', 'Jhumkas', 'Potli Bag'],
    'Sports':  ['Compression Shorts', 'Sports Socks', 'Cap', 'Water Bottle', 'Gym Bag'],
}

CATEGORY_RELEVANCE = {
    'Watches': 30, 'Heels': 28, 'Handbags': 27,
    'Shirts': 26, 'Tops': 25, 'Kurtas': 25,
    'Sports Shoes': 24, 'Casual Shoes': 23,
    'Tshirts': 22, 'Sunglasses': 22,
}


# ---------------------------------------------------------------------------
# Color extraction
# ---------------------------------------------------------------------------

def _color_name(rgb):
    r, g, b = [c / 255 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.18:        return 'Black'
    if s < 0.12 and v > 0.88: return 'White'
    if s < 0.12:        return 'Grey'
    h_deg = h * 360
    if h_deg < 15 or h_deg >= 345: return 'Red'
    if h_deg < 40:  return 'Orange'
    if h_deg < 65:  return 'Yellow'
    if h_deg < 80:  return 'Yellow-Green'
    if h_deg < 170: return 'Green'
    if h_deg < 195: return 'Teal'
    if h_deg < 260: return 'Blue'
    if h_deg < 290: return 'Purple'
    if h_deg < 345: return 'Pink'
    return 'Red'


def _filter_background(pixels):
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    not_white = ~((r > 220) & (g > 220) & (b > 220))
    not_black = ~((r < 30) & (g < 30) & (b < 30))
    max_c = pixels.max(axis=1).astype(float)
    min_c = pixels.min(axis=1).astype(float)
    sat = np.where(max_c > 0, (max_c - min_c) / max_c, 0)
    not_grey = sat > 0.10
    mask = not_white & not_black & not_grey
    return pixels[mask] if mask.sum() > 20 else pixels


def get_color_palette(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    w, h = img.size
    mw, mh = int(w * 0.20), int(h * 0.20)
    img = img.crop((mw, mh, w - mw, h - mh)).resize((80, 80))
    pixels = np.array(img).reshape(-1, 3).astype(float)
    filtered = _filter_background(pixels)
    if len(filtered) < 10:
        filtered = pixels

    dominant = filtered.mean(axis=0).astype(int)
    half = len(filtered) // 2
    secondary = filtered[:half].mean(axis=0).astype(int) if half > 5 else dominant
    hsv_s = np.array([colorsys.rgb_to_hsv(p[0]/255, p[1]/255, p[2]/255)[1] for p in filtered])
    top_idx = np.argsort(hsv_s)[-max(1, len(filtered)//5):]
    accent = filtered[top_idx].mean(axis=0).astype(int)

    def make(arr):
        rgb = tuple(np.clip(arr, 0, 255).tolist())
        return {'rgb': list(rgb), 'name': _color_name(rgb)}

    return {'dominant': make(dominant), 'secondary': make(secondary), 'accent': make(accent)}


def get_dominant_color(image_bytes):
    p = get_color_palette(image_bytes)
    d = p['dominant']
    return tuple(d['rgb']), d['name']


# ---------------------------------------------------------------------------
# Style scoring
# ---------------------------------------------------------------------------

def style_score(predictions, category, dominant_rgb):
    top_conf = predictions[0][1]
    conf_component = min(top_conf * 55, 40)
    cat_component = CATEGORY_RELEVANCE.get(category, 20)
    r, g, b = [c / 255 for c in dominant_rgb]
    _, s, v = colorsys.rgb_to_hsv(r, g, b)
    color_component = 22 if (s < 0.15 and v > 0.75) else round((s * 0.5 + v * 0.5) * 30)
    return round(min(max(conf_component + cat_component + color_component, 40), 95), 1)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def suggest_matches(top_category, occasion=None):
    base = PAIRS.get(top_category, ['Well-fitted Bottoms', 'Clean Footwear', 'Minimal Accessories'])
    if occasion and occasion in OCCASION_PAIRS:
        combined = list(dict.fromkeys(base + OCCASION_PAIRS[occasion]))
        return combined[:5]
    return base[:5]


# ---------------------------------------------------------------------------
# Fallback rationale (used when Claude API unavailable)
# ---------------------------------------------------------------------------

FALLBACK_RATIONALE = {
    'Tshirts':      ["Clean tee — keep the fit slim and the outfit simple.",
                     "Solid casual pick. The right bottoms elevate this instantly.",
                     "Basic but versatile. Good shoes and clean bottoms do the work."],
    'Shirts':       ["Sharp shirt — works from office to evening with minimal effort.",
                     "Decent shirt. Focus on fit; tuck or untuck to adjust formality.",
                     "The shirt has potential — ironing and the right trouser seal it."],
    'Tops':         ["Stylish top — pairs beautifully with high-waist bottoms.",
                     "Good top. Fitted bottoms keep proportions balanced.",
                     "The top needs the right pairing — avoid anything too loose below."],
    'Kurtas':       ["Elegant ethnic pick — silhouette and fabric look refined.",
                     "Nice kurta. Churidar or straight trousers complete the look.",
                     "Traditional but understated — accessories add the personality."],
    'Casual Shoes': ["Great everyday shoe — versatile and effortless to style.",
                     "Solid casual footwear. Keep the outfit simple to let them lead.",
                     "Functional shoes. Slim or tapered bottoms avoid a heavy silhouette."],
    'Sports Shoes': ["Strong sneaker — on-trend and pairs well with streetwear looks.",
                     "Good athletic shoe. Works for gym or relaxed casual wear.",
                     "Functional sports shoe — best kept to active or casual contexts."],
    'Heels':        ["Elegant heel — adds height and polish effortlessly.",
                     "Nice heels. Tailored clothing gives the most put-together result.",
                     "Statement heels — keep everything else in the outfit minimal."],
    'Handbags':     ["Well-chosen bag — structure and proportion look balanced.",
                     "Good handbag. Match the formality level to your full outfit.",
                     "The bag works functionally — coordinate the colour carefully."],
    'Watches':      ["Great timepiece — reads intentional and elevates any outfit.",
                     "Solid watch. Dress it up or down depending on strap and dial.",
                     "Functional watch — works best with smart-casual or formal looks."],
    'Sunglasses':   ["Strong frame — the shape suits a wide range of face types.",
                     "Good sunglasses. Ensure the frame shape complements your face.",
                     "Functional eyewear — frame size and shape make a real difference."],
}
_DEFAULT_FB = ["Strong item — reads polished and well-considered.",
               "Solid pick. Pair thoughtfully to get the most out of it.",
               "Workable item — the right styling context brings it to life."]


def _fallback_rationale(score, category):
    t = FALLBACK_RATIONALE.get(category, _DEFAULT_FB)
    return t[0] if score >= 75 else (t[1] if score >= 58 else t[2])


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_response(image_bytes, predictions, occasion=None, ai_rationale=None):
    palette = get_color_palette(image_bytes)
    dom_rgb = tuple(palette['dominant']['rgb'])
    top_category = predictions[0][0]
    score = style_score(predictions, top_category, dom_rgb)
    matches = suggest_matches(top_category, occasion)
    rationale = ai_rationale or _fallback_rationale(score, top_category)

    return {
        'detected_category': top_category,
        'top_predictions': [{'label': l, 'confidence': round(c, 3)} for l, c in predictions],
        'color_palette': palette,
        'dominant_color': palette['dominant'],
        'style_score': score,
        'rationale': rationale,
        'suggested_matches': matches,
    }