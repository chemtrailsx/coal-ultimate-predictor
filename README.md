# Coal Ultimate Analysis Predictor

Predicts the **ultimate analysis** of coal (C, H, N, S, O) from its **proximate analysis** (moisture, volatile matter, ash, fixed carbon) with a neural-network ensemble. It works with US data (ASTM D3172/D3176, as-received basis) and Indian data (IS 1350, air-dried basis), and with dry-basis data.

## Contents

| Path | What it is |
|---|---|
| `Coal_Proximate_to_Ultimate_Model.ipynb` | The full model notebook: data, cleaning, feature engineering, model comparison, validation, final training and export. Submit this one. |
| `data/` | Training and test data (USGS COALQUAL, USGS–GSI India, USGS World Coal Quality Inventory) |
| `model/coal_model.json` | The trained model (network weights as plain numbers), written by the notebook |
| `coal_model.py` | Prediction code using numpy only, shared by the app and the tests |
| `app.py` | Streamlit web app: single sample, CSV batch, model information |
| `web/index.html` | Stand-alone browser version, no server needed (for GitHub Pages) |
| `tests/` | Consistency tests (Python vs JavaScript, closure to 100 %, input checks) |
| `build_notebook.py`, `run_nb.py` | Regenerate and execute the notebook |

## Method

1. Organic matter is Org = 100 − M − Ash, and volatile matter on a dry, ash-free basis is VM_daf = 100·VM/Org.
2. Inputs to the network are VM_daf, Ash/Org and M/Org. The ensemble averages 10 neural networks (32 × 32, ReLU), blended 70 : 30 with a quadratic regression. It predicts C, H, N and S on a dry, ash-free basis.
3. O_daf = 100 − C − H − N − S. Each element is converted back to the input basis, so C + H + N + S + O + M + Ash = 100 % exactly.
4. If a measured total sulphur value is given, it replaces the predicted one and oxygen is recalculated.

Validation works on whole sites, so the model is always scored on coal it never saw:
- Indian mines are held out one at a time (leave-one-site-out).
- US sample groups are held out in 5 folds.
- 60 other countries from the World Coal Quality Inventory are used as an external test only.

The exact error tables are in the notebook.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
python tests/test_consistency.py      # needs Node.js for the JavaScript check
```

To retrain, run `pip install scikit-learn matplotlib nbformat nbclient ipykernel`, then `python build_notebook.py` and `python run_nb.py`. The notebook rewrites `model/coal_model.json`, and both the app and the web page use it automatically.

## Deploy (free, public URL)

**Streamlit Community Cloud**
1. Push this folder to a public GitHub repository.
2. Go to https://share.streamlit.io, sign in with GitHub, choose **Create app**, then pick the repository, branch `main` and file `app.py`.
3. Click **Deploy**. You get a public `https://<name>.streamlit.app` link.

**GitHub Pages** (the browser version)
1. In the same repository, go to Settings → Pages → Deploy from branch → `main`, folder `/ (root)`.
2. The predictor is then at `https://<user>.github.io/<repo>/web/`. It loads `model/coal_model.json` and runs entirely in the browser.

## Data sources

- Palmer, C.A., et al. (2015). *U.S. Geological Survey COALQUAL database version 3.0*. USGS Data Series 975.
- Trippi, M.H. & Tewalt, S.J. (2011). *Geographic information system of coal-quality data for India and Bangladesh*. USGS Open-File Report 2011-1296 (with the Geological Survey of India).
- Tewalt, S.J., et al. (2010). *A digital compilation of world coal quality data (WoCQI)*. USGS Open-File Report 2010-1196.

## Limitations

- Sulphur cannot be predicted from a proximate analysis, so enter a measured value when you have one.
- The Indian training data cover the Sohagpur coalfield and Rajasthan lignite, so errors may be larger for other coalfields.
- Ash stands in for mineral matter, so coals rich in carbonates or pyrite get only an approximate correction.
