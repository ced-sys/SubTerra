# subterra_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Optional XGBoost import
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Global state
models = {}
feature_names = []
scaler = StandardScaler()

# Load and preprocess

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    drop_cols = [col for col in df.columns if any(k in col.lower() for k in ['fid', 'path', 'layer', 'id'])]
    df_cleaned = df.drop(columns=drop_cols)
    if 'label' not in df_cleaned.columns:
        raise ValueError("Dataset must contain a 'label' column")
    X = df_cleaned.drop(columns=['label']).apply(pd.to_numeric, errors='coerce')
    y = df_cleaned['label']
    X.fillna(X.median(), inplace=True)
    global feature_names
    feature_names = X.columns.tolist()
    return X, y

# Split and scale

def split_data(X, y, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

# Train models

def train_models(X_train, y_train, X_train_scaled, random_state=42):
    global models
    models['decision_tree'] = DecisionTreeClassifier(max_depth=8, min_samples_split=10,
                                                    min_samples_leaf=5, random_state=random_state)
    models['random_forest'] = RandomForestClassifier(n_estimators=100, max_depth=10,
                                                    min_samples_split=10, random_state=random_state, n_jobs=-1)
    if XGBOOST_AVAILABLE:
        models['xgboost'] = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                        random_state=random_state, eval_metric='logloss',
                                        use_label_encoder=False)
    for name, model in models.items():
        if name == 'xgboost':
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
    return models

# Evaluate models

def evaluate_models(X_test, y_test, X_test_scaled):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled if name == 'xgboost' else X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
    best_model = max(results, key=results.get)
    return results, best_model

# Feature importance

def get_feature_importance(model_name):
    if model_name in models and hasattr(models[model_name], 'feature_importances_'):
        importances = models[model_name].feature_importances_
        return dict(zip(feature_names, importances))
    return {}

# Predict single sample

def predict_single_sample(features: dict, model_name='decision_tree'):
    if model_name not in models:
        raise ValueError(f"Model {model_name} not available")
    model = models[model_name]
    feature_array = np.array([features.get(name, 0) for name in feature_names]).reshape(1, -1)
    if model_name == 'xgboost':
        feature_array = scaler.transform(feature_array)
    prediction = model.predict(feature_array)[0]
    probabilities = model.predict_proba(feature_array)[0] if hasattr(model, 'predict_proba') else None
    return {
        'prediction': int(prediction),
        'label': 'Positive' if prediction == 1 else 'Negative',
        'probabilities': probabilities,
        'model': model_name
    }

