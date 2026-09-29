"""
generate_data.py
Creates a SYNTHETIC but realistic Mumbai-style housing dataset (data/mumbai_house_data.csv).

IMPORTANT: this data is simulated using approximate 2024-25 locality price levels (Rs per sq ft).
It is meant so that the whole pipeline runs out-of-the-box. For your final submission, replace it with a
real dataset (e.g. Kaggle "Mumbai House Prices") that has the same columns - see README.md.
"""
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 6000

# locality: (region, approx base price per sq ft in Rs, station distance typical km)
LOCALITIES = {
    "Malabar Hill":   ("South Mumbai",     65000, 2.5),
    "Worli":          ("South Mumbai",     52000, 2.0),
    "Lower Parel":    ("South Mumbai",     46000, 0.8),
    "Byculla":        ("South Mumbai",     32000, 0.6),
    "Bandra West":    ("Western Suburbs",  50000, 1.0),
    "Juhu":           ("Western Suburbs",  46000, 3.0),
    "Andheri West":   ("Western Suburbs",  28000, 1.2),
    "Goregaon West":  ("Western Suburbs",  21000, 1.5),
    "Malad West":     ("Western Suburbs",  18500, 1.4),
    "Kandivali West": ("Western Suburbs",  16500, 1.6),
    "Borivali West":  ("Western Suburbs",  16000, 1.3),
    "Dadar":          ("Central Suburbs",  40000, 0.5),
    "Kurla":          ("Central Suburbs",  19000, 0.8),
    "Ghatkopar":      ("Central Suburbs",  23000, 0.9),
    "Powai":          ("Central Suburbs",  23500, 4.0),
    "Vikhroli":       ("Central Suburbs",  22000, 1.5),
    "Mulund West":    ("Central Suburbs",  20000, 1.2),
    "Chembur":        ("Harbour",          26000, 1.0),
    "Vashi":          ("Navi Mumbai",      15000, 0.9),
    "Kharghar":       ("Navi Mumbai",      10000, 1.8),
    "Thane West":     ("Thane",            13500, 1.7),
}

# BHK -> (min area, max area) in sq ft, and a small price-per-sqft premium for compact homes
BHK_AREA = {1: (300, 620), 2: (520, 1000), 3: (850, 1600), 4: (1300, 2800), 5: (2200, 4500)}
BHK_PSF_ADJ = {1: 1.06, 2: 1.00, 3: 0.98, 4: 1.02, 5: 1.08}
BHK_PROB = [0.22, 0.42, 0.27, 0.08, 0.01]
FURN = ["Unfurnished", "Semi-Furnished", "Furnished"]
FURN_ADJ = {"Unfurnished": 1.00, "Semi-Furnished": 1.04, "Furnished": 1.09}


def make_row():
    loc = RNG.choice(list(LOCALITIES))
    region, base_psf, st_typ = LOCALITIES[loc]
    bhk = int(RNG.choice([1, 2, 3, 4, 5], p=BHK_PROB))
    if region == "South Mumbai" and bhk == 1 and RNG.random() < 0.5:
        bhk = 2
    lo, hi = BHK_AREA[bhk]
    area = int(RNG.triangular(lo, (lo + hi) / 2.2, hi))
    total_floors = int(RNG.choice([4, 7, 10, 14, 20, 28, 40, 55], p=[.12, .13, .18, .17, .18, .12, .07, .03]))
    floor = int(RNG.integers(1, total_floors + 1))
    status = RNG.choice(["Ready to Move", "Under Construction"], p=[0.78, 0.22])
    age = 0 if status == "Under Construction" else int(np.clip(RNG.gamma(2.2, 5.0), 0, 45))
    furnishing = RNG.choice(FURN, p=[0.42, 0.38, 0.20]) if status == "Ready to Move" else "Unfurnished"
    parking = int(RNG.choice([0, 1, 2], p=[0.28, 0.55, 0.17]))
    amenities = int(np.clip(RNG.normal(4.5 + (2 if total_floors >= 14 else 0) - age * 0.06, 2), 0, 10))
    dist_station = round(float(np.clip(RNG.gamma(2.0, st_typ / 2.0), 0.1, 9.0)), 1)
    sea_facing = int(RNG.random() < (0.35 if loc in ("Worli", "Malabar Hill", "Juhu", "Bandra West") else 0.02))

    psf = base_psf * BHK_PSF_ADJ[bhk] * FURN_ADJ[furnishing]
    psf *= 1 - min(age * 0.008, 0.28)                       # depreciation with age
    psf *= 1 + 0.15 * (floor / total_floors) * (total_floors >= 10)   # higher floor premium in towers
    psf *= 0.90 if status == "Under Construction" else 1.0  # under construction discount
    psf *= 1 + 0.035 * parking
    psf *= 1 + 0.012 * (amenities - 5)
    psf *= 1 - 0.012 * min(dist_station, 5)                 # farther from station = cheaper
    psf *= 1.10 if sea_facing else 1.0
    psf *= float(np.exp(RNG.normal(0, 0.09)))               # unexplained market noise

    price_lakh = round(area * psf / 1e5, 2)
    return dict(locality=loc, region=region, bhk=bhk, area_sqft=area, floor=floor,
                total_floors=total_floors, age_years=age, status=status, furnishing=furnishing,
                parking=parking, amenities_score=amenities, dist_station_km=dist_station,
                sea_facing=sea_facing, price_lakh=price_lakh)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame([make_row() for _ in range(N)])
    # add a little realistic messiness: a few missing values
    for col in ["age_years", "amenities_score"]:
        df.loc[df.sample(frac=0.01, random_state=1).index, col] = np.nan
    df.to_csv("data/mumbai_house_data.csv", index=False)
    print(df.shape)
    print(df.describe().round(2).T)
