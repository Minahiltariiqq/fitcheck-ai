"""
ml/scripts/01_prepare_data.py

Prepares the Kaggle Fashion Product Images dataset for training:
- Filters to top N most common article types
- Resizes all images to 224x224 (MobileNetV2 input)
- Splits into train (70%) / val (15%) / test (15%)
- Outputs to ml/data/processed/

Run from inside the ml/scripts folder:
    python 01_prepare_data.py

Works on Windows / Mac / Linux. CPU only — no GPU needed for this step.
"""
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────
# Paths are relative to this script's location, so it works on any OS
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / 'data' / 'raw'
PROC_DIR = SCRIPT_DIR.parent / 'data' / 'processed'
IMG_SIZE = 224

# Keep this manageable for a beginner FYP. Start with 10, can grow later.
TOP_N_TYPES = 10


def main():
    # Verify the dataset was downloaded
    styles_csv = RAW_DIR / 'styles.csv'
    images_dir = RAW_DIR / 'images'

    if not styles_csv.exists():
        print(f"ERROR: Can't find {styles_csv}")
        print("Did you download the Kaggle dataset and unzip it into ml/data/raw/?")
        print("See SETUP_GUIDE.md, Phase 2, for download instructions.")
        return

    if not images_dir.exists():
        print(f"ERROR: Can't find {images_dir}")
        return

    # Load metadata
    print("Loading metadata...")
    df = pd.read_csv(styles_csv, on_bad_lines='skip')
    df = df.dropna(subset=['articleType', 'baseColour', 'usage'])
    print(f"  Total entries: {len(df)}")

    # Keep only the top N most common article types
    top_types = df['articleType'].value_counts().head(TOP_N_TYPES).index.tolist()
    df = df[df['articleType'].isin(top_types)].copy()
    print(f"  After filtering to top {TOP_N_TYPES} types: {len(df)}")
    print(f"  Categories: {top_types}")

    # Filter to only rows where the image file actually exists
    df['img_path'] = df['id'].apply(lambda x: images_dir / f'{x}.jpg')
    df = df[df['img_path'].apply(lambda p: p.exists())].reset_index(drop=True)
    print(f"  Images that exist on disk: {len(df)}")

    # Stratified split 70/15/15
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df['articleType'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['articleType'], random_state=42
    )

    splits = [('train', train_df), ('val', val_df), ('test', test_df)]

    for split_name, split_df in splits:
        # Create subfolder per category
        for article_type in split_df['articleType'].unique():
            (PROC_DIR / split_name / article_type).mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {split_name} ({len(split_df)} images)...")
        skipped = 0
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            try:
                img = Image.open(row['img_path']).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                out_path = PROC_DIR / split_name / row['articleType'] / f"{row['id']}.jpg"
                img.save(out_path, quality=90)
            except Exception as e:
                skipped += 1
        if skipped:
            print(f"  Skipped {skipped} unreadable images")

    # Save metadata for later use (color/usage prediction extensions)
    df.to_csv(PROC_DIR / 'metadata.csv', index=False)

    print("\n" + "=" * 50)
    print("DONE.")
    print(f"Processed data is in: {PROC_DIR}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("=" * 50)
    print("\nNext step: open ml/notebooks/colab_train.ipynb in Google Colab")
    print("to train the model on a free GPU (much faster than your CPU).")


if __name__ == '__main__':
    main()
