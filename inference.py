
import pickle, numpy as np, pandas as pd
from scipy.signal import savgol_filter

PHASE_MAP = {0: "Lag", 1: "Log (Exponential)", 2: "Stationary", 3: "Death"}
PHASE_META = {
    "Lag":              {"color": "#4e9af1", "icon": "🔬",
                         "desc": "Bacteria adapting — RNA & enzyme synthesis. No reproduction yet."},
    "Log (Exponential)":{"color": "#2ecc71", "icon": "🚀",
                         "desc": "Peak health — rapid binary fission, doubling at a constant rate."},
    "Stationary":       {"color": "#f39c12", "icon": "⚖️",
                         "desc": "Birth rate = death rate. Nutrients depleted, waste accumulates."},
    "Death":            {"color": "#e74c3c", "icon": "💀",
                         "desc": "Cell death exceeds division. Population crashes logarithmically."}
}

def load_model(path="bacteria_phase_model.pkl"):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"], bundle["features"]

def engineer_features(time_arr, od600_arr, nutrient_arr, ph_arr):
    """
    Given time-series arrays, compute all engineered features.
    Returns a DataFrame of shape (n_timepoints, n_features).
    """
    od  = savgol_filter(np.maximum(od600_arr, 0.001), 7, 3)
    dt  = np.diff(time_arr).mean()
    gr  = np.gradient(od, dt)
    acc = np.gradient(gr, dt)
    od_std = pd.Series(od).rolling(5, min_periods=1).std().fillna(0).values
    T = time_arr[-1]
    return pd.DataFrame({
        "od600":  od,
        "gr":     gr,
        "acc":    acc,
        "nut":    nutrient_arr,
        "ph":     ph_arr,
        "od_std": od_std,
        "log_od": np.log1p(od),
        "tnorm":  time_arr / T,
    })

def predict_phases(time_arr, od600_arr, nutrient_arr, ph_arr, model_path="bacteria_phase_model.pkl"):
    model, scaler, features = load_model(model_path)
    feat_df = engineer_features(time_arr, od600_arr, nutrient_arr, ph_arr)
    X_scaled = scaler.transform(feat_df[features].values)
    preds    = model.predict(X_scaled)
    probs    = model.predict_proba(X_scaled)
    results  = []
    for i, (p, prob) in enumerate(zip(preds, probs)):
        phase = PHASE_MAP[p]
        results.append({
            "time":        float(time_arr[i]),
            "phase":       phase,
            "confidence":  float(prob.max()),
            "color":       PHASE_META[phase]["color"],
            "icon":        PHASE_META[phase]["icon"],
            "description": PHASE_META[phase]["desc"],
        })
    return results

# Example usage:
# time, od, nut, ph = ... (numpy arrays of same length)
# results = predict_phases(time, od, nut, ph)
# print(results[0])  # → {'time': 0.5, 'phase': 'Lag', 'confidence': 0.97, ...}
