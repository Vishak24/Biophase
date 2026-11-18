# BioPhase AI

Classifies which growth phase a bacterial batch culture is in (**Lag, Log, Stationary or Death**) from optical density (OD600), growth rate, pH and nutrient level. A Random Forest model is exported to ONNX and runs entirely in the browser, with no server.

**Live demo:** [biophase-ai.web.app](https://biophase-ai.web.app) (static site on Firebase Hosting)

## Why it's interesting

Deciding when a culture moves from exponential growth to stationary phase matters for harvesting, antibiotic timing and contamination monitoring. It is usually judged by eye from a growth curve. This project explores whether a small model over a few cheap, standard lab measurements can make that call, and how to ship such a model with zero backend.

| Phase | Behaviour | Typical signals |
|---|---|---|
| Lag | cells adapting, little division | low OD600, growth rate ≈ 0, nutrients high, pH neutral |
| Log | exponential division | rising OD600, high growth rate, nutrients falling |
| Stationary | division ≈ death | OD600 at plateau, growth rate ≈ 0, nutrients low, pH dropping |
| Death | death exceeds division | OD600 falling, negative growth rate, nutrients ≈ 0, acidic |

## How it works

**Training data is simulated.** Growth curves were generated with a logistic rise, exponential decline, nutrient depletion and pH acidification, plus Gaussian noise and Savitzky–Golay smoothing to resemble spectrophotometer readings. The simulation script is not included in this repository; only the trained artefacts are.

**Features (8).** The model takes four measured values plus four derived from the time series (see `inference.py`):

| Feature | Meaning |
|---|---|
| `od600` | smoothed optical density |
| `gr` | growth rate, dOD/dt |
| `acc` | rate of change of growth rate |
| `nut` | nutrient level (fraction of initial, 0–1) |
| `ph` | medium pH |
| `od_std` | rolling standard deviation of OD (5 points) |
| `log_od` | log(1 + OD) |
| `tnorm` | time as a fraction of the run |

**Model.** scikit-learn `RandomForestClassifier` with `StandardScaler`, stratified 80/20 split. The model is exported to `bacteria_model.onnx` for the browser, and the scaler parameters to `scaler.json`.

**Two inference paths**

- `inference.py`: given a full time series (time, OD600, nutrients, pH), it computes all eight features properly and predicts a phase per time point.
- The web dashboard (`index.html`, `app.js`): takes a *single* reading from four sliders and runs the ONNX model with onnxruntime-web. A single point has no history, so three features are approximated: `acc = 0`, `od_std = 0.02`, and `tnorm` estimated from OD600. The slider units are also rescaled to the training ranges (nutrients g/L → fraction, growth rate re-centred).

## Results (and how to read them)

On the held-out **simulated** test set (240 samples, 60 per phase), the model scores 1.00 precision and recall for every class. That shows the model recovers the rules used to generate the data. It is **not** evidence of accuracy on real cultures, which has not been measured. The phase boundaries, especially Lag/Log, are where real data would be hardest.

Reported feature importances: nutrients ≈ 29%, pH ≈ 19%, growth rate ≈ 14%, OD600 ≈ 13%.

## Run locally

```bash
# Web dashboard (any static server; the ONNX model is fetched relative to the page)
python3 -m http.server 8000      # then open http://localhost:8000

# Python inference on a time series
pip install numpy pandas scipy scikit-learn
python3 -c "from inference import predict_phases; help(predict_phases)"
```

## Repository contents

```
index.html, app.js          dashboard UI (D3 growth curve, sliders, in-browser ONNX inference)
bacteria_model.onnx         model for the browser
bacteria_phase_model.pkl    model + scaler + feature list for inference.py (pickle; load only files you trust)
scaler.json                 scaler mean/scale for the browser
inference.py                feature engineering + per-time-point prediction
firebase.json               Firebase Hosting config
```

The dashboard UI was designed with Google Stitch.

## Limitations and next steps

- Validate against real OD600 time series, for example public growth-curve datasets or plate-reader exports, and report accuracy per phase boundary.
- Commit the simulation and training scripts so the results are reproducible.
- Feed the dashboard a short history of readings instead of one point, so the derived features are real rather than approximated.
- Try sequence models (for example a temporal CNN) for multi-step phase detection.

## License

[MIT](LICENSE)
