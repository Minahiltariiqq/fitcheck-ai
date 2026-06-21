# FitCheck AI — Step-by-Step Setup Guide (Windows + Dell Laptop)

This guide walks you from a fresh Windows install to a deployed web app.
**Total time:** ~6-8 hours of work spread across a few days.

**Your setup:** Dell laptop, Windows 10/11, 16 GB RAM, Intel/AMD CPU (no NVIDIA GPU).

---

## Important upfront: where training happens

Your laptop has no GPU, so training a deep learning model locally would take **8-12 hours**. We're going to be smart:

| Step | Where it runs | Why |
|---|---|---|
| Data preparation | Your laptop | Just resizing images, CPU is fine |
| **Model training** | **Google Colab (free GPU)** | **~30 minutes instead of 10 hours** |
| Backend server | Your laptop | Single image inference is fast on CPU |
| Frontend | Your laptop | Just JavaScript |
| Final deployment | Free cloud (Vercel + Hugging Face) | Free forever |

You'll only need internet for Colab and the final deployment. Everything else works offline.

---

# Phase 0 — Install Everything (1 hour, one-time)

## 0.1 Install Git for Windows

1. Go to https://git-scm.com/download/win
2. Download and run the installer
3. **During installation, accept all defaults except this one:**
   - When asked about "Choose the default editor used by Git" → pick **"Use Visual Studio Code as Git's default editor"** (you'll install VS Code in step 0.4)
4. After install, open **Git Bash** from the Start Menu. Test it:
   ```bash
   git --version
   ```
   You should see something like `git version 2.45.0`.

> **Important:** From now on, use **Git Bash** as your terminal (not Command Prompt or PowerShell). It looks and behaves like a Mac/Linux terminal, which matches all online tutorials.

## 0.2 Install Miniforge (Python for Data Science)

Don't install Anaconda — it's bloated. Miniforge is leaner.

1. Go to https://github.com/conda-forge/miniforge/releases
2. Scroll down and download `Miniforge3-Windows-x86_64.exe`
3. Run the installer:
   - **Install for "Just Me"** (not all users)
   - **Important:** On the "Advanced Installation Options" screen, check ✅ **"Add Miniforge3 to my PATH environment variable"** (it warns you not to — ignore the warning, we want this).
4. After install, **close Git Bash and reopen it.**
5. Test it:
   ```bash
   conda --version
   ```
   Should print `conda 24.x.x`.

## 0.3 Create your Python environment

In Git Bash:

```bash
conda create -n fitcheck python=3.11 -y
conda activate fitcheck
```

You should now see `(fitcheck)` at the start of your prompt.

> **Every time you open Git Bash to work on this project, run `conda activate fitcheck` first.** If you forget, packages will install into the wrong place.

## 0.4 Install PyTorch (CPU version for Windows)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

This grabs the CPU-only build — smaller, faster install, doesn't try to use a GPU you don't have.

Test it:
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch version: 2.x.x+cpu
CUDA available: False
```

`CUDA available: False` is correct for your laptop. Don't try to "fix" it.

## 0.5 Install the rest of the Python libraries

```bash
pip install numpy pandas pillow opencv-python scikit-learn matplotlib seaborn tqdm
pip install flask flask-cors python-dotenv gunicorn
pip install jupyterlab
```

## 0.6 Install Node.js (for the React frontend)

1. Go to https://nodejs.org
2. Download the **LTS version** (currently 20.x or 22.x)
3. Run the installer, accept all defaults
4. Restart Git Bash, then test:
   ```bash
   node --version
   npm --version
   ```

## 0.7 Install Visual Studio Code

1. Go to https://code.visualstudio.com/
2. Download and install
3. Open VS Code, click the **Extensions** icon (left sidebar, looks like 4 squares)
4. Install these extensions (search by name):
   - **Python** (by Microsoft)
   - **Pylance** (by Microsoft)
   - **ES7+ React/Redux/React-Native snippets**
   - **Tailwind CSS IntelliSense**
   - **Prettier — Code formatter**

## ✅ Phase 0 done when:
- [ ] `git --version` works in Git Bash
- [ ] `conda activate fitcheck` shows `(fitcheck)` prompt
- [ ] `python -c "import torch; print(torch.__version__)"` works
- [ ] `node --version` works
- [ ] VS Code is installed with the 5 extensions

---

# Phase 1 — Get the Project Code (15 minutes)

## 1.1 Pick a project location

In Git Bash:

```bash
cd ~/Documents
mkdir fitcheck-ai
cd fitcheck-ai
```

## 1.2 Copy all the files from this download

Extract the zip I gave you. Inside is a `fitcheck-ai` folder with all the files. Copy its contents into `~/Documents/fitcheck-ai/`.

Verify by running:
```bash
ls -la
```

You should see folders: `backend/`, `frontend/`, `ml/`, `docs/`, plus files `.gitignore`, `README.md`.

## 1.3 Initialize Git

```bash
git init
git add .
git commit -m "Initial project setup"
```

## ✅ Phase 1 done when:
- [ ] `ls` shows all the project folders
- [ ] `git status` shows a clean working tree

---

# Phase 2 — Download the Dataset (30 minutes)

## 2.1 Get a Kaggle account and API token

1. Go to https://www.kaggle.com and create a free account
2. Click your profile picture → **Settings**
3. Scroll to "API" → click **"Create New Token"**
4. A file named `kaggle.json` downloads. **Save it!**

## 2.2 Place the Kaggle token correctly

In Git Bash:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## 2.3 Download the dataset

```bash
pip install kaggle
cd ~/Documents/fitcheck-ai/ml/data/raw
kaggle datasets download -d paramaggarwal/fashion-product-images-small
```

This downloads ~280 MB. Then unzip:

```bash
unzip fashion-product-images-small.zip
rm fashion-product-images-small.zip
```

You should now have:
```
ml/data/raw/
├── images/        ← ~44,000 JPG files
├── styles.csv     ← labels for each image
└── myntradataset/ ← (if present, can ignore)
```

> **If `unzip` is not found:** install it with `pacman -S unzip` in Git Bash, OR just right-click the zip in File Explorer → "Extract All".

## 2.4 Prepare the data

```bash
cd ~/Documents/fitcheck-ai/ml/scripts
python 01_prepare_data.py
```

This will:
- Filter to the top 10 most common clothing types
- Resize all images to 224x224
- Split into train/val/test sets

**Takes ~5-10 minutes on your Dell.** You'll see a progress bar.

When done, you'll have:
```
ml/data/processed/
├── train/Tshirts/*.jpg, train/Jeans/*.jpg, ...
├── val/...
├── test/...
└── metadata.csv
```

## ✅ Phase 2 done when:
- [ ] `ml/data/processed/train/` exists with subfolders per category
- [ ] Each subfolder has hundreds of JPG files
- [ ] You see a "DONE." message at the end

---

# Phase 3 — Train the Model on Google Colab (45 minutes)

## 3.1 Zip the processed data

In Git Bash:

```bash
cd ~/Documents/fitcheck-ai/ml/data
# This uses Python because Windows zip command line is awkward
python -c "import shutil; shutil.make_archive('processed', 'zip', 'processed')"
```

This creates `processed.zip` (~300 MB).

## 3.2 Upload it to Google Drive

1. Go to https://drive.google.com
2. Drag and drop `processed.zip` into the root of your Drive
3. Wait for upload to finish (a few minutes on average wifi)

## 3.3 Open the Colab notebook

1. Go to https://colab.research.google.com
2. Click **File → Upload notebook**
3. Upload `ml/notebooks/colab_train.ipynb` from your project

## 3.4 Switch to GPU mode

**This is the critical step.** In Colab top menu:

**Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save**

If you skip this, training runs on CPU and takes hours.

## 3.5 Run each cell

Click each cell and press **Shift+Enter** to run it. Run cells in order:

1. **Cell 1 (verify GPU):** Should print `CUDA available: True` and `Device: Tesla T4`
2. **Cell 2 (mount Drive):** A popup asks for permission — grant it
3. **Cell 3 (unzip data):** This loads your data from Drive
4. **Cell 4 (training):** This is the long one — ~30 minutes. You'll see epoch progress like:
   ```
   Epoch 01/15 | train_acc=0.756 val_acc=0.812 time=95s
   Epoch 02/15 | train_acc=0.834 val_acc=0.851 time=92s
     ✓ saved best model (val_acc=0.851)
   ...
   ```
5. **Cell 5 (download):** Two files download to your laptop: `fitcheck_mobilenet.pth` and `labels.json`

> **If Colab disconnects mid-training:** click **Runtime → Reconnect**, then re-run cells 2, 3, 4. Free Colab disconnects after ~12 hours of idle, but training is only 30 min so this rarely happens.

## 3.6 Put the trained model in your project

The two downloaded files go in **two places**:

```bash
# In Git Bash on your laptop:
cd ~/Downloads

# Copy to the ml/models folder (for evaluation scripts)
cp fitcheck_mobilenet.pth ~/Documents/fitcheck-ai/ml/models/
cp labels.json ~/Documents/fitcheck-ai/ml/models/

# Copy to the backend folder (so the Flask server can use it)
cp fitcheck_mobilenet.pth ~/Documents/fitcheck-ai/backend/app/model.pth
cp labels.json ~/Documents/fitcheck-ai/backend/app/labels.json
```

Note: in the backend folder, the model file is renamed to `model.pth` (no prefix).

## 3.7 Evaluate the model

```bash
cd ~/Documents/fitcheck-ai/ml/scripts
python 03_evaluate.py
```

You'll see a classification report. **Look at the "accuracy" row near the bottom.** Should be ≥0.85 (85%). Also check `ml/reports/confusion_matrix.png` — open the image to see how well it does per category.

**📸 Screenshot this for your FYP report.**

## 3.8 Benchmark inference speed

```bash
python 04_benchmark.py
```

You should see something like:
```
Average inference time: 80-150 ms per image
```

Anything under 500ms is great. Your <3 second KPI is safe.

## ✅ Phase 3 done when:
- [ ] `backend/app/model.pth` exists (~14 MB)
- [ ] `backend/app/labels.json` exists
- [ ] Test accuracy ≥85%
- [ ] Inference benchmark under 500ms

---

# Phase 4 — Run the Backend Locally (15 minutes)

## 4.1 Install backend dependencies

```bash
cd ~/Documents/fitcheck-ai/backend
pip install -r requirements.txt
```

## 4.2 Start the server

```bash
cd ~/Documents/fitcheck-ai/backend
python -m app.main
```

You'll see:
```
[inference] Loading model with 10 classes...
[inference] Model loaded and ready.
 * Running on http://127.0.0.1:5001
 * Running on http://192.168.x.x:5001
```

**Keep this terminal window open!** The server runs until you close it.

## 4.3 Test it from a second terminal

Open a **new** Git Bash window (don't close the first one). Run:

```bash
curl http://127.0.0.1:5001/health
```

Should print:
```
{"service":"fitcheck-ai","status":"ok"}
```

Now test with an image:

```bash
cd ~/Documents/fitcheck-ai
curl -X POST http://127.0.0.1:5001/analyze \
  -F "image=@ml/data/processed/test/Tshirts/$(ls ml/data/processed/test/Tshirts | head -1)" \
  -F "occasion=Casual"
```

You should see a JSON response with detected category, style score, dominant color, and suggested matches.

## ✅ Phase 4 done when:
- [ ] Backend starts without errors
- [ ] `/health` returns `{"status":"ok"}`
- [ ] `/analyze` returns a real prediction for a test image

## Common backend mistakes
- **"ModuleNotFoundError: No module named 'app'":** You ran `python main.py` instead of `python -m app.main`. The `-m` matters.
- **"FileNotFoundError: model.pth":** You didn't copy the trained model into `backend/app/`. Re-do step 3.6.
- **Port 5001 already in use:** Some other program is using that port. In `main.py`, change `port=5001` to `port=5002`, and update the frontend `.env.development` accordingly.

---

# Phase 5 — Build and Run the Frontend (30 minutes)

## 5.1 Create the Vite project

The `frontend/` folder already has the custom files I made, but you still need Vite to scaffold the boilerplate (package.json, node_modules, etc.).

```bash
cd ~/Documents/fitcheck-ai/frontend
npm create vite@latest . -- --template react
```

When prompted:
- "Current directory is not empty. Please choose how to proceed:" → pick **"Ignore files and continue"** (this preserves the files I gave you)

Then install dependencies:

```bash
npm install
```

## 5.2 Install Tailwind CSS

```bash
npm install -D tailwindcss@3 postcss autoprefixer
npx tailwindcss init -p
```

This creates two files (`tailwind.config.js` and `postcss.config.js`). The `tailwind.config.js` you already have **will be overwritten** by the npx command — that's fine, just open it and replace its contents with what I gave you originally.

To make sure, paste this into `frontend/tailwind.config.js`:

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}"
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
```

## 5.3 Make sure the right files are in place

Double-check these files match what I gave you:
- `frontend/src/App.jsx` (the React component)
- `frontend/src/index.css` (Tailwind imports)
- `frontend/index.html` (title set to FitCheck AI)
- `frontend/.env.development` (`VITE_API_URL=http://127.0.0.1:5001`)
- `frontend/tailwind.config.js` (content paths)

If any got overwritten by Vite, copy the originals back from the download.

## 5.4 Run the frontend

**Make sure the backend is still running in another terminal.** Then in this one:

```bash
npm run dev
```

You'll see:
```
  VITE v5.x.x  ready in 234 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.x.x:5173/
```

Open http://localhost:5173 in Chrome.

You should see the FitCheck AI page. Upload a clothing photo, pick "Casual" or another occasion, hit "Get my FitCheck", and watch the result panel populate.

## ✅ Phase 5 done when:
- [ ] Page loads at localhost:5173 with the FitCheck branding
- [ ] Upload → analyze → result flow works end-to-end
- [ ] Style score shows, color swatch shows, suggested pairs show

## Common frontend mistakes
- **CORS error in browser console:** Backend isn't running, or the frontend can't reach it. Re-check that `python -m app.main` is running and the URL in `.env.development` matches.
- **"Failed to fetch" error:** Same as above.
- **Page loads but looks ugly:** Tailwind didn't install correctly. Re-run step 5.2.
- **`npm create vite@latest` asks scary questions:** Always pick: framework = **React**, variant = **JavaScript**, then "Ignore files and continue".

---

# Phase 6 — Deploy to the Internet (45 minutes)

## 6.1 Push your code to GitHub

1. Go to https://github.com and create a free account if you don't have one
2. Click **+ → New repository**
3. Name it `fitcheck-ai`, keep it public, **don't** initialize with README
4. Click **Create repository**
5. Copy the URL it gives you (looks like `https://github.com/YOUR_USERNAME/fitcheck-ai.git`)

Back in Git Bash:

```bash
cd ~/Documents/fitcheck-ai
git add .
git commit -m "Working prototype"
git remote add origin https://github.com/YOUR_USERNAME/fitcheck-ai.git
git branch -M main
git push -u origin main
```

GitHub will ask for credentials. Use a **Personal Access Token** as your password (not your GitHub login password):
- GitHub → Settings → Developer Settings → Personal access tokens → Tokens (classic) → Generate new token
- Give it `repo` scope, copy the token, paste it as the password when Git asks

## 6.2 Deploy the backend to Hugging Face Spaces

1. Sign up at https://huggingface.co (free)
2. Click your profile → **New Space**
3. Fill in:
   - **Space name:** `fitcheck-ai-backend`
   - **License:** MIT
   - **Space SDK:** **Docker** (this is important!)
   - **Hardware:** **CPU basic (free)**
   - Visibility: Public
4. Click **Create Space**

The Space URL will be: `https://huggingface.co/spaces/YOUR_USERNAME/fitcheck-ai-backend`

Now clone the empty Space and push your backend code into it:

```bash
cd ~/Documents
git clone https://huggingface.co/spaces/YOUR_USERNAME/fitcheck-ai-backend
cd fitcheck-ai-backend
```

When prompted for credentials, use your Hugging Face username and an **access token** (get one at https://huggingface.co/settings/tokens — "Write" permission).

Copy your backend files in:

```bash
cp -r ~/Documents/fitcheck-ai/backend/* .
```

Set up Git LFS for the model file (it's 14 MB):

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

Commit and push:

```bash
git add .
git commit -m "Initial backend deployment"
git push
```

After ~3 minutes, your backend will be live at:
```
https://YOUR_USERNAME-fitcheck-ai-backend.hf.space
```

Test it:
```bash
curl https://YOUR_USERNAME-fitcheck-ai-backend.hf.space/health
```

## 6.3 Update the frontend production URL

Edit `frontend/.env.production` and replace the placeholder URL with your actual HF Space URL:

```
VITE_API_URL=https://YOUR_USERNAME-fitcheck-ai-backend.hf.space
```

Commit and push:

```bash
cd ~/Documents/fitcheck-ai
git add .
git commit -m "Set production backend URL"
git push
```

## 6.4 Deploy the frontend to Vercel

1. Sign up at https://vercel.com using your GitHub account
2. After login, click **Add New → Project**
3. Pick your `fitcheck-ai` repo from the list
4. Configure:
   - **Root Directory:** click "Edit" → select `frontend`
   - **Framework Preset:** Vite (auto-detected)
   - **Environment Variables:** click "Add"
     - Name: `VITE_API_URL`
     - Value: `https://YOUR_USERNAME-fitcheck-ai-backend.hf.space`
5. Click **Deploy**

After ~45 seconds, you'll get a URL like:
```
https://fitcheck-ai.vercel.app
```

Open it. **You should see your app live on the internet.** 🎉

## 6.5 Test the deployed app

- Open the Vercel URL on your phone
- Upload a clothing photo
- Get a result

**If it works on a device you've never opened the project on, you've shipped a real product.**

## ✅ Phase 6 done when:
- [ ] Code is on GitHub
- [ ] Backend live on Hugging Face Spaces
- [ ] Frontend live on Vercel
- [ ] App works end-to-end from your phone

---

# Phase 7 — Defense Day Checklist

Print this and tick boxes the day before:

### Technical
- [ ] Hit your Vercel URL 1 minute before demo (HF Spaces sleeps after inactivity, takes ~20s to wake up)
- [ ] Backend `/health` returns OK
- [ ] Tested on phone Safari/Chrome (not just laptop)
- [ ] 5+ test images on hand

### Polish
- [ ] README on GitHub has live URL + screenshot
- [ ] Confusion matrix saved as image for your slides
- [ ] Classification report screenshot for your slides
- [ ] QR code generated for the Vercel URL (use https://www.qr-code-generator.com)
- [ ] Page title says "FitCheck AI" (check by hovering browser tab)

### Story
- [ ] You can explain why MobileNetV2 not VGG16 (smaller, 10× faster, same accuracy)
- [ ] You can explain transfer learning in one sentence (use a pretrained model and only retrain the last layer)
- [ ] You can explain the style score formula (60% confidence + 40% color quality)

---

# Common total-stuck moments and what to do

| Problem | Fix |
|---|---|
| `conda: command not found` | Close and reopen Git Bash. If still broken, the PATH wasn't set during Miniforge install — reinstall and check "Add to PATH" |
| Training takes hours in Colab | You forgot Runtime → Change runtime type → T4 GPU |
| Backend gives "model.pth not found" | The .pth file is in the wrong folder. It needs to be at `backend/app/model.pth` |
| Frontend "Failed to fetch" | Backend not running, or `.env.development` URL is wrong |
| Vercel build fails | Check that root directory is set to `frontend`, not the repo root |
| HF Space build fails | Check the build logs — usually a missing file in `backend/`. Make sure the `model.pth` actually pushed (Git LFS can fail silently — verify file size in the HF file viewer is ~14 MB, not a few bytes) |
| Pushing to HF asks for username/password forever | You need an HF access token, not your password. Settings → Access Tokens → New token with "Write" permission |
| Image uploads to backend hang | Image is bigger than 8 MB. Either resize before upload, or change `MAX_CONTENT_LENGTH` in `main.py` |

---

# What "done" looks like

You have:
1. A working live URL anyone can visit
2. A GitHub repo proving the code works
3. A trained model that hits ≥85% accuracy
4. Documentation (this guide + README) for your supervisor to verify
5. Screenshots for your final defense slides

That's a complete FYP-quality project. The total realistic timeline from this point is **2-3 focused weekends**. The hardest part is Phase 0 (one-time setup) — after that, everything else is following recipes.

Good luck.
