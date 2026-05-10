# vista_filter_merge.py
# Usage: python vista_filter_merge.py
# Output: vista_cleaned_base.csv
# Reads travel mode ONLY from 'main_journey_mode' in JTW/JTE.
# Preserves raw categories like "Vehicle Driver" and "Vehicle Passenger".

import pandas as pd
from pathlib import Path
from collections import Counter

# ---- FILES (edit if needed) ----
HOUSEHOLD_CSV = "household_vista_2023_2024.csv"
PERSON_CSV    = "person_vista_2023_2024.csv"
JTW_CSV       = "journey_to_work_vista_2023_2024 (1).csv"
JTE_CSV       = "journey_to_education_vista_2023_2024.csv"
TRIPS_CSV     = "trips_vista_2023_2024.csv"

OUTPUT_BASE   = "vista_cleaned_base.csv"

# ---- Helpers ----
def read_csv_lower(path, id_cols=("hhid","persid","tripid")):
    """Lower-case columns and coerce id-like columns to string to avoid dtype merge issues."""
    hdr = pd.read_csv(path, nrows=0).columns
    low = hdr.str.lower()
    dtype = {c: "string" for c in id_cols if c in low}
    df = pd.read_csv(path, low_memory=False, dtype=dtype)
    df.columns = df.columns.str.lower()
    return df

def extract_main_mode_series(jdf, source_name="work"):
    """
    Extract a single main mode per person using ONLY the 'main_journey_mode' column.
    - If multiple rows per person: majority vote; tie -> first encountered.
    Returns a Series indexed by persid with the chosen raw mode (unmodified labels).
    """
    if "persid" not in jdf.columns or "main_journey_mode" not in jdf.columns:
        return pd.Series(dtype="object", name=f"primary_mode_{source_name}")

    # Normalise text lightly (strip), but keep original labels (no lower-casing)
    tmp = jdf[["persid", "main_journey_mode"]].copy()
    tmp["main_journey_mode"] = tmp["main_journey_mode"].astype(str).str.strip()
    tmp = tmp[tmp["main_journey_mode"].notna() & (tmp["main_journey_mode"] != "")]

    winners = {}
    for pid, g in tmp.groupby("persid", sort=False):
        vals = g["main_journey_mode"].tolist()
        if not vals:
            continue
        counts = Counter(vals)
        best = max(counts.values())
        candidates = [v for v, ct in counts.items() if ct == best]
        if len(candidates) == 1:
            winners[pid] = candidates[0]
        else:
            # tie-break by first appearance order
            for v in vals:
                if v in candidates:
                    winners[pid] = v
                    break
    return pd.Series(winners, name=f"primary_mode_{source_name}")

def combine_modes_pref_work(work_s, edu_s):
    """
    Combine Work and Edu: prefer Work if both exist; else whichever exists.
    If neither exists for a valid person, leave as None (we'll fill 'Unknown' later).
    """
    idx = work_s.index.union(edu_s.index)
    combined = pd.Series(index=idx, dtype="object", name="primary_mode_raw")
    combined.loc[idx] = edu_s.reindex(idx)
    combined.loc[work_s.index] = work_s
    return combined

def normalize_to_three(mode_raw):
    """
    Map raw labels to Private/Public/Active for analysis (keeps raw intact).
    This mapping preserves Vehicle Driver/Passenger information in the raw column.
    """
    if not isinstance(mode_raw, str):
        return None
    s = mode_raw.strip().lower()
    # Private
    if any(k in s for k in ["vehicle driver", "vehicle passenger", "car", "motorcycle", "uber", "taxi", "rideshare", "ride share", "ride-share", "hire car"]):
        return "Private"
    # Public
    if any(k in s for k in ["train", "tram", "bus", "ferry", "v/line", "coach"]):
        return "Public"
    # Active
    if any(k in s for k in ["walk", "walking", "bicycle", "bike", "cycle", "e-scooter", "escooter", "scooter (electric)"]):
        return "Active"
    return "Other"

def drop_license_columns(df):
    """Remove licence flags to avoid confusion (you asked to retain Vehicle Driver/Passenger as categories, not licence fields)."""
    drop_like = {"carlicence","mbikelicence","otherlicence","nolicence","licence","license"}
    to_drop = [c for c in df.columns if any(k in c for k in drop_like)]
    return df.drop(columns=to_drop, errors="ignore")

# ---- Pipeline ----
def main():
    household = read_csv_lower(HOUSEHOLD_CSV, id_cols=("hhid",))
    person    = read_csv_lower(PERSON_CSV,    id_cols=("hhid","persid"))
    jtw       = read_csv_lower(JTW_CSV,       id_cols=("hhid","persid"))
    jte       = read_csv_lower(JTE_CSV,       id_cols=("hhid","persid"))
    trips     = read_csv_lower(TRIPS_CSV,     id_cols=("hhid","persid","tripid"))

    print("Shapes:")
    for n, df in [("household",household),("person",person),("jtw",jtw),("jte",jte),("trips",trips)]:
        print(f"  {n:9s}: {df.shape}")

    # Valid persons: Person ∩ Trips ∩ (Work ∪ Edu)
    person_ids = set(person["persid"].dropna())
    trip_ids   = set(trips["persid"].dropna())
    work_ids   = set(jtw["persid"].dropna())
    edu_ids    = set(jte["persid"].dropna())
    valid_ids  = person_ids & trip_ids & (work_ids | edu_ids)

    print(f"\nInitial persons: {len(person_ids)}")
    print(f"Valid persons (Person ∩ Trips ∩ (Work ∪ Edu)): {len(valid_ids)}")

    # Merge base (Person + Household)
    base = person[person["persid"].isin(valid_ids)].copy()
    base = base.merge(household, on="hhid", how="inner", suffixes=("_person","_hh"))
    print(f"After merge (person+household): {base.shape}")

    # Extract per-person mode from ONLY 'main_journey_mode'
    work_mode = extract_main_mode_series(jtw, "work")
    edu_mode  = extract_main_mode_series(jte, "edu")
    mode_comb = combine_modes_pref_work(work_mode, edu_mode)

    # Attach mode; fill Unknown when missing
    base["primary_mode_raw"] = base["persid"].map(mode_comb.to_dict())
    base["primary_mode_raw"] = base["primary_mode_raw"].fillna("Unknown")

    # 3-bucket mapping (keeps raw labels like Vehicle Driver/Passenger intact)
    base["travel_mode_3"] = base["primary_mode_raw"].apply(normalize_to_three)

    # Drop licence flags (you keep vehicle driver/passenger via the raw mode, not licence fields)
    base = drop_license_columns(base)

    # Put useful columns first
    front = [
        "persid","hhid","persno",
        "primary_mode_raw","travel_mode_3",
        "agegroup","sex","relationship","studying","anywork",
        "anzsco1","anzsco2","emptype","persinc",
        "hhinc_group","totalvehs","owndwell","homelga",
        "homesubregion_asgs_person","homeregion_asgs_person",
        "homesubregion_asgs_hh","homeregion_asgs_hh",
        "perspoststratweight","hhpoststratweight",
    ]
    front = [c for c in front if c in base.columns]
    ordered = base[front].join(base.drop(columns=front, errors="ignore"))
    print(ordered)

    # Save
    ordered.to_csv(OUTPUT_BASE, index=False)
    print(f"\nSaved: {OUTPUT_BASE} -> {Path(OUTPUT_BASE).resolve()}")

    # Quick checks
    print("\nTop raw modes (preserved labels):")
    print(ordered["primary_mode_raw"].value_counts(dropna=False).head(20))
    print("\ntravel_mode_3 distribution:")
    print(ordered["travel_mode_3"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
