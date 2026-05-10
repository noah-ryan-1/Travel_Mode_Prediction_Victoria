# SES & Age vs Travel Mode + Primary Mode deep dives (CSV only, no PNGs)
# - Adds age_stage_4 to SES vs travel_mode_3
# - SES vs primary_mode_raw (overall)
# - Within Private: Vehicle driver vs Vehicle passenger
# - Within Active: Walking vs Cycling
# - Within Public: Train vs Tram vs Bus
#
# Outputs: CSV summaries + crosstabs

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency
from sklearn.metrics import normalized_mutual_info_score

DATA_PATH = "vista_stage2_collapsed.csv"
OUT_DIR   = Path("vista_week1_outputs"); OUT_DIR.mkdir(exist_ok=True)

TARGET_COARSE = "travel_mode_3"        # Active / Public / Private / Other
TARGET_FINE   = "primary_mode_raw"     # detailed (train, tram, bus, driver, walking, etc.)

# Core SES + Age (add or remove as needed)
SES_VARS = [
    "occupation_5",
    "edu_status_3",
    "hh_income_3",
    "homesubregion_asgs_person",
    "age_stage_4",
]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    needed = [TARGET_COARSE, TARGET_FINE] + SES_VARS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected column(s): {missing}. "
                           f"Use vista_stage2_collapsed.csv from pipeline.")
    for c in needed:
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()
    return df

def cramers_v(values: np.ndarray) -> float:
    chi2, _, _, _ = chi2_contingency(values, correction=False)
    n = values.sum()
    if n == 0: return np.nan
    r, k = values.shape
    denom = n * (min(r - 1, k - 1))
    if denom <= 0: return np.nan
    return np.sqrt(chi2 / denom)

def assoc_summary(df: pd.DataFrame, feature: str, target: str):
    ct = pd.crosstab(df[feature], df[target], dropna=False)
    chi2, p, dof, _ = chi2_contingency(ct.values, correction=False)
    v = cramers_v(ct.values)
    # NMI on integer codes
    x_codes = pd.Categorical(df[feature]).codes
    y_codes = pd.Categorical(df[target]).codes
    nmi = normalized_mutual_info_score(x_codes, y_codes, average_method="min")
    return {"feature": feature, "ct": ct, "chi2": chi2, "p": p, "dof": dof, "v": v, "nmi": nmi}

def save_summary(results, summary_name_prefix, target_label):
    rows = []
    for r in results:
        rows.append({
            "feature": r["feature"],
            "chi2": round(r["chi2"], 3),
            "dof": r["dof"],
            "p_value": float(r["p"]),
            "cramers_v": round(r["v"], 3) if r["v"] == r["v"] else np.nan,
            "nmi": round(r["nmi"], 3),
        })
        fname = r["feature"].replace(" ", "_")
        r["ct"].to_csv(OUT_DIR / f"crosstab_{fname}_vs_{target_label}.csv")
    summary = pd.DataFrame(rows).sort_values(by=["cramers_v","nmi"], ascending=False)
    summary.to_csv(OUT_DIR / f"{summary_name_prefix}.csv", index=False)
    return summary

# --- Helpers for submode classification
def norm_primary_mode(x: str) -> str:
    return str(x).strip().lower()

def private_submode(x: str) -> str:
    s = norm_primary_mode(x)
    if "driver" in s: return "Vehicle driver"
    if "passenger" in s: return "Vehicle passenger"
    if "uber" in s or "taxi" in s or "ride" in s: return "Other private"
    return "Other private"

def active_submode(x: str) -> str:
    s = norm_primary_mode(x)
    if "walk" in s: return "Walking"
    if "cycle" in s or "bike" in s: return "Cycling"
    return "Other active"

def public_submode(x: str) -> str:
    s = norm_primary_mode(x)
    if "train" in s or "rail" in s: return "Train"
    if "tram" in s or "light rail" in s: return "Tram"
    if "bus" in s: return "Bus"
    return "Other public"

def run():
    df = load_data(DATA_PATH)

    # A) SES + Age vs travel_mode_3 (coarse)
    results_coarse = [assoc_summary(df, f, TARGET_COARSE) for f in SES_VARS]
    save_summary(results_coarse, "ses_vs_mode_summary", TARGET_COARSE)

    # B) SES + Age vs primary_mode_raw (fine)
    results_fine = [assoc_summary(df, f, TARGET_FINE) for f in SES_VARS]
    save_summary(results_fine, "ses_vs_primary_summary", TARGET_FINE)

    # C1) Private only: Driver vs Passenger
    df_priv = df[df[TARGET_COARSE].str.lower() == "private"].copy()
    if not df_priv.empty:
        df_priv["private_sub"] = df_priv[TARGET_FINE].apply(private_submode)
        mask_two = df_priv["private_sub"].isin(["Vehicle driver","Vehicle passenger"])
        df_priv_two = df_priv[mask_two].copy()
        if not df_priv_two.empty and df_priv_two["private_sub"].nunique() > 1:
            results_priv = [assoc_summary(df_priv_two, f, "private_sub") for f in SES_VARS]
            save_summary(results_priv, "within_private_driver_vs_passenger", "private_sub")

    # C2) Active only: Walking vs Cycling
    df_act = df[df[TARGET_COARSE].str.lower() == "active"].copy()
    if not df_act.empty:
        df_act["active_sub"] = df_act[TARGET_FINE].apply(active_submode)
        mask_two = df_act["active_sub"].isin(["Walking","Cycling"])
        df_act_two = df_act[mask_two].copy()
        if not df_act_two.empty and df_act_two["active_sub"].nunique() > 1:
            results_act = [assoc_summary(df_act_two, f, "active_sub") for f in SES_VARS]
            save_summary(results_act, "within_active_walk_vs_cycle", "active_sub")

    # C3) Public only: Train vs Tram vs Bus
    df_pub = df[df[TARGET_COARSE].str.lower() == "public"].copy()
    if not df_pub.empty:
        df_pub["public_sub"] = df_pub[TARGET_FINE].apply(public_submode)
        keep3 = df_pub["public_sub"].isin(["Train","Tram","Bus"])
        df_pub3 = df_pub[keep3].copy()
        if not df_pub3.empty and df_pub3["public_sub"].nunique() > 1:
            results_pub = [assoc_summary(df_pub3, f, "public_sub") for f in SES_VARS]
            save_summary(results_pub, "within_public_train_tram_bus", "public_sub")

    print("Done, Outputs in:", OUT_DIR.resolve())
    print("Created:")
    print(" - ses_vs_mode_summary.csv")
    print(" - ses_vs_primary_summary.csv")
    print(" - within_private_driver_vs_passenger.csv (if data exists)")
    print(" - within_active_walk_vs_cycle.csv (if data exists)")
    print(" - within_public_train_tram_bus.csv (if data exists)")

if __name__ == "__main__":
    run()
