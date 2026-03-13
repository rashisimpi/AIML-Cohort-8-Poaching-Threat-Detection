Team 3 — Model Development
Cohort 8 | IIMSTC | SDG Goal 15 : Life on Land
Project: Poaching Threat Detection using Acoustic Sensors
Overview
Team 3 focuses solely on model development — selecting algorithms, training models, and tuning hyperparameters to build classifiers (Random Forest, KNN, SVM) using preprocessed FSC22 audio features from Team 2 for binary threat vs. wildlife detection. No evaluation or deployment; outputs trained models ready for next stages

Member ResponsibilitiesDarshan Reddy — Model Selection & Random Forest
Darshan selects algorithms based on classification needs (KNN, SVM, Random Forest) and develops Random Forest using ensemble decision trees for robust handling of audio 
Sameer — Training & KNN
Sameer performs training with train/validation splits and k-fold cross-validation; develops KNN model for distance-based neighbor predictions.
Saikeerthi — Hyperparameter Tuning & SVM
Saikeerthi tunes parameters via grid/random search; develops SVM with optimal kernel/regularization for threat-wildlife separation
⚙️ Model Development Pipeline
Team 2 Features (2025 samples)
↓
Model Selection → RF/KNN/SVM (Darshan)
↓
Training → Splits + CV → Fit (Sameer)
↓
Tuning → Grid search → Optimize (Saikeerthi)
↓
Trained Models → .pkl files for next team📊 Output Summary
Total models developed: 3
Saved as: random_forest_model.pkl, knn_model.pkl, svm_model.pkl📄 Output Files
models/random_forest_model.pkl (Darshan)knn_model.pkl (Sameer)svm_model.pkl (Saikeerthi)
tuning_logs.csv — Params & CV scores🚀 How to RunInstall dependenciespip install scikit-learn pandas numpy joblibLoad featuresdf = pd.read_csv("../team2/metadata.csv")  # Features from Team 2
X_train, X_val = train_test_split(df[features], df['label_encoded'])
Handoff to Next Team
Next team loads:import joblib
rf = joblib.load("models/random_forest_model.pkl")
# Ready for evaluation/deploymentSee tuning_logs.csv for best params used.📚 References
Scikit-learn: https://scikit-learn.org
Team 2 Output: ../team2/README_team2.md
