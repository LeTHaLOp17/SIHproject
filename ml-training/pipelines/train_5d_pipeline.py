import os
import sys
import shutil

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 60)
print("LANDSLIDE RISK PREDICTION - 5D ML PIPELINE")
print("=" * 60)

# STEP 1: Load Dataset
csv_path = r'd:\SIHproject\NER_landslide_training_12000.csv'
if not os.path.exists(csv_path):
    csv_path = r'd:\SIHproject\ml-training\data\NER_landslide_training_12000.csv'

print(f"\n[+] Loading dataset from {csv_path}...")
df = pd.read_csv(csv_path)
print(f"[OK] Loaded {len(df)} samples with {len(df.columns)} features")

# STEP 2: Define 5D Feature Set
print("\n[*] Creating 5D Feature Set...")
DIMENSIONS = {
    'D1_elevation': 'elevation_m',
    'D2_slope': 'slope_degree', 
    'D3_soil_moisture': 'soil_moisture_pct',
    'D4_ndvi': 'ndvi',
    'D5_annual_rainfall': 'ANNUAL'
}

feature_cols = list(DIMENSIONS.values())
target_col = 'landslide_risk_label'

X_5d = df[feature_cols].copy()
y = df[target_col].copy()

imputer = SimpleImputer(strategy='median')
X_5d = pd.DataFrame(imputer.fit_transform(X_5d), columns=feature_cols)

print(f"[OK] 5D Features selected:")
for dim, col in DIMENSIONS.items():
    print(f"  * {dim}: {col}")

# STEP 3: Generate 5 Augmented Datasets
print("\n[*] Generating Augmented Datasets...")
datasets = {}

datasets['5D_Core'] = (X_5d.copy(), y.copy())

def augment_data(X, y, multiplier=2):
    X_aug = X.copy()
    y_aug = y.copy()
    for _ in range(multiplier - 1):
        idx1 = np.random.randint(0, len(X), size=len(X))
        idx2 = np.random.randint(0, len(X), size=len(X))
        alpha = np.random.random(len(X))
        X_new = X.iloc[idx1].values * alpha[:, np.newaxis] + \
                X.iloc[idx2].values * (1 - alpha[:, np.newaxis])
        y_new = y.iloc[idx1].values
        X_aug = pd.concat([X_aug, pd.DataFrame(X_new, columns=X.columns)], ignore_index=True)
        y_aug = pd.concat([y_aug, pd.Series(y_new)], ignore_index=True)
    return X_aug, y_aug

X_aug, y_aug = augment_data(X_5d, y, multiplier=2)
datasets['5D_Augmented'] = (X_aug, y_aug)

scaler = MinMaxScaler()
X_norm = pd.DataFrame(scaler.fit_transform(X_5d), columns=feature_cols)
datasets['5D_Normalized'] = (X_norm, y.copy())

standardizer = StandardScaler()
X_std = pd.DataFrame(standardizer.fit_transform(X_5d), columns=feature_cols)
datasets['5D_Standardized'] = (X_std, y.copy())

pca = PCA(n_components=5)
X_pca = pd.DataFrame(pca.fit_transform(X_5d), columns=[f'PC{i+1}' for i in range(5)])
datasets['5D_PCA'] = (X_pca, y.copy())

out_dir = r'd:\SIHproject'
os.makedirs(os.path.join(out_dir, 'ml-training', 'datasets_5d'), exist_ok=True)

for name, (data_features, data_labels) in datasets.items():
    data_with_label = data_features.copy()
    data_with_label['landslide_risk_label'] = data_labels.values
    
    file1 = os.path.join(out_dir, f'Dataset_{name}.csv')
    file2 = os.path.join(out_dir, 'ml-training', 'datasets_5d', f'Dataset_{name}.csv')
    data_with_label.to_csv(file1, index=False)
    data_with_label.to_csv(file2, index=False)
    print(f"[OK] Saved Dataset_{name}.csv ({len(data_with_label)} samples)")

print(f"\n[+] Total datasets created: {len(datasets)}")

# STEP 4: Train Models on 5D Dataset
print("\n[*] Training ML Models on 5D Core Dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_5d, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, max_iter=2000, random_state=RANDOM_STATE)
}

results = {}

for name, model in models.items():
    print(f"  -> Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    # Fast 3-fold CV on subsample for speed
    cv_sample_idx = np.random.choice(len(X_5d), size=min(3000, len(X_5d)), replace=False)
    cv_scores = cross_val_score(model, X_5d.iloc[cv_sample_idx], y.iloc[cv_sample_idx], cv=3, n_jobs=-1)
    results[name]['CV_Mean'] = cv_scores.mean()
    results[name]['CV_Std'] = cv_scores.std()

# STEP 5: Display Results
print("\n" + "=" * 60)
print("MODEL PERFORMANCE RESULTS")
print("=" * 60)
print(f"\n{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 60)
for name, metrics in results.items():
    print(f"{name:<20} {metrics['Accuracy']:.4f}    {metrics['Precision']:.4f}    "
          f"{metrics['Recall']:.4f}    {metrics['F1-Score']:.4f}")

print("\n" + "-" * 60)
print("Cross-Validation Scores (5-fold):")
for name, metrics in results.items():
    print(f"{name:<20} {metrics['CV_Mean']:.4f} (+/- {metrics['CV_Std']*2:.4f})")

# STEP 6: Feature Importance
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (Random Forest)")
print("=" * 60)

rf_model = models['Random Forest']
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance.iterrows():
    bar = '#' * int(row['Importance'] * 50)
    print(f"{row['Feature']:<20} {bar} {row['Importance']:.4f}")

# STEP 7: Save Model and Predictions
print("\n[*] Saving model and predictions...")
model_path = os.path.join(out_dir, 'landslide_rf_model_5d.pkl')
joblib.dump(rf_model, model_path)
os.makedirs(os.path.join(out_dir, 'backend', 'ai-engine', 'app', 'weights'), exist_ok=True)
joblib.dump(rf_model, os.path.join(out_dir, 'backend', 'ai-engine', 'app', 'weights', 'landslide_rf_model_5d.pkl'))
print(f"[OK] Saved model: landslide_rf_model_5d.pkl")

pred_path = os.path.join(out_dir, 'predictions_5d.csv')
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': rf_model.predict(X_test),
    'Probability': rf_model.predict_proba(X_test)[:, 1]
})
predictions_df.to_csv(pred_path, index=False)
print("[OK] Saved predictions: predictions_5d.csv")

imp_path = os.path.join(out_dir, 'feature_importance_5d.csv')
importance.to_csv(imp_path, index=False)
print("[OK] Saved feature importance: feature_importance_5d.csv")

pipeline_src = r'C:\Users\Ayush\.gemini\antigravity\brain\bc1a4ddb-bda8-45fa-9fe4-5669a9e9af79\scratch\train_5d_runner.py'
os.makedirs(os.path.join(out_dir, 'ml-training', 'pipelines'), exist_ok=True)
shutil.copy(pipeline_src, os.path.join(out_dir, 'ml-training', 'pipelines', 'train_5d_pipeline.py'))

print("\n" + "=" * 60)
print("[SUCCESS] 5D PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)
