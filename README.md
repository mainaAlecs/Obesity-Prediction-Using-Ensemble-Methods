Obesity Prediction Using Ensemble Methods

This project presents a machine learning pipeline for predicting obesity levels using a range of lifestyle, physical, and dietary indicators. Developed as part of a Kaggle competition, the final solution uses ensemble learning—specifically a Random Forest classifier—to achieve robust and accurate predictions across seven obesity categories


Problem Statement
Accurately predicting obesity levels is vital for healthcare insights and targeted interventions. Given labeled data on individuals' eating habits, physical conditions, and activity levels, the goal was to classify each instance into one of seven predefined obesity categories.

🚀 Project Highlights
Achieved 81.68% accuracy, up from a baseline Decision Tree model's 47.55%.

Employed Random Forest with systematic hyperparameter tuning (200 trees, max depth 20).

Used feature engineering (e.g., dietary combinations, interaction terms) to capture subtle behavioral patterns.

Performed feature selection via Information Gain and Select by Weights to improve model clarity and efficiency.

Applied 10-Fold Stratified Cross-Validation to ensure robust performance evaluation.

🛠️ Technologies Used
Python

Scikit-learn

Pandas, NumPy

RapidMiner (for initial prototyping and feature ranking)

Matplotlib / Seaborn (for EDA and visualization)

📁 Project Structure
├── data/               # Dataset and preprocessing scripts
├── notebooks/          # Jupyter Notebooks with EDA and model development
├── src/                # Source code for preprocessing, feature engineering, modeling
├── results/            # Confusion matrices, evaluation metrics, and final model artifacts
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies



⚙️ Model Performance
The final Random Forest classifier demonstrated strong generalization, especially in distinguishing between extreme classes. Some confusion persisted between adjacent classes like "Overweight I" and "Overweight II," attributed to feature overlap.

📌 Future Work
Experiment with deep learning models for feature extraction.

Integrate additional lifestyle factors (e.g., sleep, stress).

Deploy as a web-based risk assessment tool for public use.
