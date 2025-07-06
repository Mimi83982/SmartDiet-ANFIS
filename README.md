Got it! Here's a **simple and clear `README.md`** for your [SmartDiet-ANFIS](https://github.com/Mimi83982/SmartDiet-ANFIS) project:

---

````markdown
# SmartDiet-ANFIS

A personalized diet recommender system using fuzzy logic and ANFIS (Adaptive Neuro-Fuzzy Inference System).

---

## 🔧 Setup Instructions

1. **Create a new project** in PyCharm (or another IDE).
2. **Set Python interpreter** to version `3.10.13`.
3. **Download the project ZIP** from this GitHub repo and extract it.
4. **Move all files** into your project folder (e.g. inside `PyCharmProjects/YourProjectName`).
5. Open a terminal and run:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
````

---

## 📦 Data Preparation

1. **Install Kaggle API** (if not installed):

   ```bash
   pip install kaggle
   ```

2. **Get your Kaggle API key**:

   * Create a Kaggle account
   * Go to [https://www.kaggle.com/account](https://www.kaggle.com/account)
   * Click **"Create New API Token"** → download `kaggle.json`
   * Move `kaggle.json` to `~/.kaggle/` on your machine

3. **Run these scripts in order**:

   ```bash
   python src/download_recipes.py
   python src/build_recipes.py
   python src/build_training_data.py
   ```

---

## 🤖 Model Training

Run the ANFIS training script:

```bash
python src/anfis_local/train_satisfaction.py
```

---

## 🖥️ Run the App

Start the GUI:

```bash
python src/gui/gui_diet_app.py
```

Make sure the Python interpreter is still set to **3.10.13**.

---

## ✅ Notes

* You need internet access to download the Kaggle dataset.
* Make sure `data/` and `models/` folders exist before training and running the app.

---

Enjoy your SmartDiet experience!
