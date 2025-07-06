SmartDiet-ANFIS

SmartDiet is a personalized diet recommender that combines fuzzy logic with a Neuro-Fuzzy Inference System (ANFIS). It learns from your BMI, activity level, and user feedback to suggest suitable meals.

Getting Started:

1. Clone the repository (recommended) by pasting the link into your IDE (e.g. PyCharm):
   https://github.com/Mimi83982/SmartDiet-ANFIS.git

   Or download the ZIP file, extract it, and move the contents into your project folder.

2. Set the Python interpreter to version 3.10.13.

3. Install the required dependencies:
   pip install --upgrade pip  
   pip install -r requirements.txt

4. To download the dataset, you’ll need a Kaggle account.
   - Go to your Kaggle account settings and generate a new API token.
   - Place the downloaded kaggle.json file in:
     - Windows: C:\Users\<YourUser>\.kaggle\
     - macOS/Linux: ~/.kaggle/
   - Install the Kaggle API if not already installed:
     pip install kaggle
   - Then run:
     python src/download_recipes.py

5. Prepare the dataset by running:
   python src/build_recipes.py  
   python src/build_training_data.py

6. Train the ANFIS model:
   python src/anfis_local/train_satisfaction.py

7. Launch the GUI app:
   python src/gui/gui_diet_app.py

You can now get meal recommendations based on your input and satisfaction levels. The more you interact, the smarter the system becomes.
