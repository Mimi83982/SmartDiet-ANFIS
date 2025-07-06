#!/usr/bin/env python3
# gui_diet_app.py  – SmartDietNF desktop GUI
# • 4‑column table, varied scores
# • ingredients popup restored
# • works from project root with src/ sibling folder

import csv
import datetime as dt
import sys
import webbrowser
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui  import QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout,
    QLabel, QSpinBox, QComboBox, QPushButton, QTableWidget,
    QTableWidgetItem, QMessageBox, QHeaderView
)

# ─── make project_root/src importable ───────────────────────────
ROOT = Path(__file__).resolve().parent          # project root
SRC  = ROOT / "src"
if str(SRC) not in sys.path:                    # add only once
    sys.path.insert(0, str(SRC))
# ────────────────────────────────────────────────────────────────

from utils import data_loader
from engine import recommender
from fuzzy_logic import rules      # ← local fuzzy package (with __init__.py)

# ─── stylesheet ─────────────────────────────────────────────────
STYLE = """
QWidget{background:#2b2b2b;color:#f0f0f0;font-family:Arial;font-size:13px;}
QPushButton{background:#0078d7;color:#fff;font-weight:bold;height:34px;border-radius:4px;font-size:12pt;}
QPushButton:hover{background:#2893ff;}
QTableWidget{background:#1e1e1e;alternate-background-color:#2a2a2a;selection-background-color:#444;}
QHeaderView::section{background:#444;padding:4px;border:1px solid #333;font-weight:bold;}
"""

# ─── feedback CSV ───────────────────────────────────────────────
FEEDBACK_CSV = ROOT / "data" / "user_feedback.csv"
FEEDBACK_CSV.parent.mkdir(exist_ok=True)

def append_feedback(profile, plan):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with FEEDBACK_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        for _, r in plan.iterrows():
            w.writerow([
                now,
                profile["age"], profile["height"], profile["weight"], profile["activity_level"],
                profile["satiety"],
                r["recipe_id"], r["name"], r["diet_type"],
                r["calories"], r["total_fat"], r["sugar"], r["sodium"],
                r["protein"], r["saturated_fat"], r["carbs"]
            ])

# ─── GUI class ─────────────────────────────────────────────────
class DietApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SmartDietNF – Meal‑Plan Recommender")
        self.resize(850, 580)
        self.setStyleSheet(STYLE)

        central = QWidget(); self.setCentralWidget(central)
        lay     = QVBoxLayout(central)

        # ---------- input form ----------
        form = QFormLayout(); form.setSpacing(10)
        self.age = QSpinBox(); self.age.setRange(0, 80);  self.age.setSpecialValueText("")
        self.hgt = QSpinBox(); self.hgt.setRange(0, 250); self.hgt.setSpecialValueText("")
        self.wgt = QSpinBox(); self.wgt.setRange(0, 200); self.wgt.setSpecialValueText("")
        self.act = QComboBox(); self.act.addItems(["Low", "Medium", "High"]); self.act.setCurrentIndex(-1)
        self.sat = QSpinBox(); self.sat.setRange(0, 5)
        for lbl, wdg in [("Age", self.age), ("Height (cm)", self.hgt),
                         ("Weight (kg)", self.wgt), ("Activity", self.act),
                         ("Satiety (0–5)", self.sat)]:
            form.addRow(lbl, wdg)
        lay.addLayout(form)

        # ---------- run button ----------
        btn = QPushButton("Generate Meal Plan")
        btn.setFont(QFont("", 12, QFont.Weight.Bold))
        btn.clicked.connect(self.recommend)
        lay.addWidget(btn)

        # ---------- info labels ----------
        self.bmi_lbl  = QLabel("BMI: –")
        self.diet_lbl = QLabel("Recommended Diet: –")
        lay.addWidget(self.bmi_lbl); lay.addWidget(self.diet_lbl)

        # ---------- result table ----------
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Meal", "Recipe", "Calories", "Diet Type", "Score"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.cellClicked.connect(self.detail)
        lay.addWidget(self.table)

        self.status = QLabel(""); lay.addWidget(self.status)

        # ---------- load recipes once ----------
        self.recipes_df, _ = data_loader.load_data()

    # ─── recommendation ────────────────────────────────────────
    def recommend(self):
        if not (self.hgt.value() and self.wgt.value() and self.age.value() and self.act.currentIndex() != -1):
            QMessageBox.warning(self, "Missing Info", "Fill all fields.")
            return

        profile = {
            "age": self.age.value(),
            "height": self.hgt.value(),
            "weight": self.wgt.value(),
            "activity_level": self.act.currentText(),
            "satiety": self.sat.value()
        }

        bmi = data_loader.compute_bmi(profile["weight"], profile["height"])
        self.bmi_lbl.setText(f"BMI: {bmi:.1f}")

        try:
            rules.get_fuzzy_memberships(profile, bmi)
            fz = rules.get_fuzzy_output(); assert "diet_type" in fz
        except Exception:
            fz = {"diet_type": {"balanced": 1.0}}

        best = max(fz["diet_type"], key=fz["diet_type"].get)
        profile["diet_type"] = best
        self.diet_lbl.setText(f"Recommended Diet: {best.capitalize()}")

        plan = recommender.plan_day(profile, fz, self.recipes_df, per_session=3)

        # fill table
        self.table.clearContents()
        self.table.setRowCount(len(plan))
        for r, row in plan.reset_index(drop=True).iterrows():
            for c, val in enumerate([
                row["meal_type"],
                row["name"].title(),
                f"{row['calories']:.0f}",
                row["diet_type"].replace("_", " ").title(),
                f"{row['score']:.2f}"
            ]):
                it = QTableWidgetItem(str(val))
                it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, it)

        if plan.empty:
            QMessageBox.information(self, "No Recipes", "No matches."); return

        append_feedback(profile, plan)
        self.status.setText("✔ Plan updated & feedback saved.")

    # ─── detail popup ───────────────────────────────────────────
    def detail(self, row, _):
        name = self.table.item(row, 1).text().lower()
        rec  = self.recipes_df[self.recipes_df["name"].str.lower() == name]
        if rec.empty: return
        url = rec.iloc[0].get("url", "")
        if url and url.startswith("http"):
            webbrowser.open(url)
        else:
            ing = rec.iloc[0].get("ingredients", "Ingredients not available.")
            QMessageBox.information(self, "Recipe Details",
                                    f"<b>{rec.iloc[0]['name'].title()}</b><br><br>{ing}")

# ─── launch app ────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DietApp()
    win.show()
    sys.exit(app.exec())
