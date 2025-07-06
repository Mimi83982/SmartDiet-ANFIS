#!/usr/bin/env python3
"""
build_recipes.py
────────────────
Create a clean, 200‑row sample **data/recipes.csv** from whichever raw source
CSV is present in *data/*.

Priority order:
1. RAW_recipes.csv  (Kaggle “RAW_recipes”)
2. PP_recipes.csv   (Prep‑Processed dataset)
If neither exists, the script aborts with a helpful message.

Run from project root **or** any sub‑folder:

    python build_recipes.py
"""

from __future__ import annotations
import ast
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. Locate /data and detect source file
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1] if (__file__).endswith("src" + str(Path(__file__).suffix)) else Path(__file__).resolve().parent
DATA_DIR = (ROOT / "data").resolve()

RAW     = DATA_DIR / "RAW_recipes.csv"
PP      = DATA_DIR / "PP_recipes.csv"

if RAW.exists():
    src_path = RAW
    df = pd.read_csv(src_path, usecols=["name", "ingredients", "nutrition"])
    df["nutrition"]  = df["nutrition"].fillna("[]").astype(str)
    df["calories"]   = (
        df["nutrition"]
        .apply(lambda s: ast.literal_eval(s)[0] if s.startswith("[") else np.nan)
        .astype(float)
    )

elif PP.exists():
    src_path = PP
    df = pd.read_csv(src_path)

    if "calories" not in df.columns:
        sys.exit("❌ PP_recipes.csv is missing a 'calories' column.")

    # build `ingredients` by concatenating ingredient_X columns
    ing_cols = [c for c in df.columns if c.startswith("ingredient_")]
    if not ing_cols:
        sys.exit("❌ PP_recipes.csv has no ingredient_* columns.")

    df["ingredients"] = df[ing_cols].astype(str).agg(", ".join, axis=1)

    if "nutrition" not in df.columns:
        df["nutrition"] = df["calories"].apply(lambda c: [c] + [0] * 6)
    df["nutrition"] = df["nutrition"].astype(str)

else:
    sys.exit(
        "❌ No source file found in /data.\n"
        "   Expected RAW_recipes.csv or PP_recipes.csv."
    )

print(f"📦 Loaded {len(df):,} rows from {src_path.name}")

# ------------------------------------------------------------------
# 2. Basic cleaning
# ------------------------------------------------------------------
df = df.dropna(subset=["name", "ingredients", "calories"])
df = df[df["calories"].astype(float) > 0]

# ------------------------------------------------------------------
# 3. Diet‑type tagging
# ------------------------------------------------------------------
def tag_diet(row) -> str:
    text = row["ingredients"].lower()
    cal  = row["calories"]
    if any(w in text for w in ["tempeh", "tofu", "lentil", "beans", "chickpea"]):
        return "vegan"
    if cal < 350 and any(w in text for w in ["salad", "greens", "lettuce", "cauliflower"]):
        return "low_carb"
    if any(w in text for w in ["chicken", "turkey", "beef", "fish"]):
        return "high_protein"
    return "balanced"

df["diet_type"] = df.apply(tag_diet, axis=1)
df = df[df["diet_type"].isin(["vegan", "balanced", "high_protein", "low_carb"])]

# ------------------------------------------------------------------
# 4. Even sample (≈200) + add prep_time & meal_type
# ------------------------------------------------------------------
target = 200
sampled = (
    df.groupby("diet_type", group_keys=False)
      .apply(lambda g: g.sample(min(target // 4, len(g)), random_state=42))
      .reset_index(drop=True)
)

rng = random.Random(42)
sampled["prep_time"] = [rng.randint(5, 30) for _ in range(len(sampled))]

def meal(minutes):  # simple heuristic
    if minutes <= 10:  return "breakfast"
    if minutes <= 20:  return "lunch"
    return "dinner"

sampled["meal_type"] = sampled["prep_time"].apply(meal)

# ------------------------------------------------------------------
# 5. Reorder / save
# ------------------------------------------------------------------
sampled.insert(0, "recipe_id", range(1, len(sampled) + 1))
sampled = sampled[
    ["recipe_id", "name", "nutrition", "ingredients",
     "calories", "diet_type", "prep_time", "meal_type"]
]

out_path = DATA_DIR / "recipes.csv"
sampled.to_csv(out_path, index=False)
print(f"✅ Wrote {len(sampled)} recipes → {out_path.relative_to(ROOT)}")
