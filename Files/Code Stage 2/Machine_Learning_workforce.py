# -*- coding: utf-8 -*-
"""
Workforce Travel Mode Prediction Pipeline
Simplified version without resampling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """Load and validate dataset with UTF-8 encoding"""
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        print(f"✅ Data loaded successfully: {data.shape}")
        return data
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(file_path, encoding='latin-1')
            print("✅ Data loaded with latin-1 encoding")
            return data
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None


def filter_workforce_subgroup(data):
    """Filter dataset for workforce population only"""
    workforce_data = data[data['anywork'] == 'Yes'].copy()
    print(f"✅ Workforce subgroup filtered: {workforce_data.shape}")
    
    # Analyze target variable distribution
    target_dist = workforce_data['primary_mode_raw'].value_counts(normalize=True) * 100
    target_counts = workforce_data['primary_mode_raw'].value_counts()
    
    print("\n🎯 Target Variable Distribution:")
    for mode, percentage in target_dist.items():
        count = target_counts[mode]
        print(f"   {mode}: {percentage:.1f}% ({count} samples)")
        
    return workforce_data


def handle_rare_classes(workforce_data, target_column='primary_mode_raw', min_samples=10):
    """Handle rare classes by grouping them into 'Other' category"""
    processed_data = workforce_data.copy()
    
    # Count samples per class
    class_counts = processed_data[target_column].value_counts()
    print(f"\n📊 Class distribution before handling rare classes:")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count} samples")
    
    # Identify rare classes
    rare_classes = class_counts[class_counts < min_samples].index.tolist()
    
    if rare_classes:
        print(f"\n⚠️  Rare classes detected (less than {min_samples} samples): {rare_classes}")
        
        # Group rare classes into "Other" category
        processed_data[target_column] = processed_data[target_column].apply(
            lambda x: 'Other' if x in rare_classes else x
        )
        
        print(f"✅ Rare classes grouped into 'Other' category")
        
        # Show new distribution
        new_class_counts = processed_data[target_column].value_counts()
        print(f"\n📊 Class distribution after handling rare classes:")
        for class_name, count in new_class_counts.items():
            print(f"   {class_name}: {count} samples")
    
    return processed_data


def preprocess_workforce_features(workforce_data):
    """Handle feature engineering and preprocessing for workforce data"""
    processed_data = workforce_data.copy()
    
    # Drop constant and irrelevant features
    columns_to_drop = [
        'anywork',  # Constant for workforce
        'persid', 'hhid', 'persno',  # ID columns
        'hhpoststratweight', 'perspoststratweight',  # Weight columns
        'travel_mode_3'  # Potentially leaky feature
    ]
    processed_data = processed_data.drop(columns=columns_to_drop, errors='ignore')
    
    # Create occupation-based features
    if 'occupation_5' in processed_data.columns:
        print("\n👔 Occupation distribution in workforce:")
        occupation_dist = processed_data['occupation_5'].value_counts(normalize=True) * 100
        for occupation, percentage in occupation_dist.items():
            print(f"   {occupation}: {percentage:.1f}%")
    
    # Handle categorical variables
    categorical_columns = processed_data.select_dtypes(include=['object']).columns
    categorical_columns = categorical_columns[categorical_columns != 'primary_mode_raw']
    
    # Exclude overly specific categorical columns
    # 'homelga'
    exclude_columns = [ 'hhinc_group', 'anzsco1', 'homesubregion_asgs_person']
    categorical_columns = [col for col in categorical_columns if col not in exclude_columns]
    
    processed_data = pd.get_dummies(processed_data, columns=categorical_columns, drop_first=True)
    
    print(f"✅ Workforce features preprocessed. Final shape: {processed_data.shape}")
    return processed_data


def analyze_workforce_correlations(data, target_column='primary_mode_raw'):
    """Analyze feature correlations for workforce data"""
    # For correlation analysis, encode target variable
    data_encoded = data.copy()
    le = LabelEncoder()
    data_encoded[target_column] = le.fit_transform(data_encoded[target_column])
    
    correlation_matrix = data_encoded.corr(numeric_only=True)
    target_correlations = correlation_matrix[target_column].abs().sort_values(ascending=False)
    
    print("\n🔍 Top feature correlations with target:")
    for feature, corr in target_correlations.head(10).items():
        if feature != target_column:
            print(f"   {feature}: {corr:.3f}")
        
    return target_correlations


def select_workforce_features(data, target_column='primary_mode_raw', k=15):
    """Select most relevant features using mutual information"""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Ensure all data is numeric
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Use mutual information for categorical targets
    mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
    
    # Create feature scores
    feature_scores = dict(zip(X_numeric.columns, mi_scores))
    
    # Select top k features with scores > 0
    selected_features_with_scores = [(feat, score) for feat, score in feature_scores.items() if score > 0]
    selected_features_with_scores = sorted(selected_features_with_scores, key=lambda x: x[1], reverse=True)[:k]
    
    # Extract feature names
    selected_feature_names = [feat for feat, score in selected_features_with_scores]
    
    print(f"\n🎯 Selected {len(selected_feature_names)} most relevant features using Mutual Information:")
    for feature, score in selected_features_with_scores[:10]:
        print(f"   {feature}: {score:.4f}")
        
    return selected_feature_names


def prepare_workforce_training_data(data, target_column='primary_mode_raw', features=None):
    """Prepare features and target for workforce training"""
    if features:
        X = data[features]
    else:
        X = data.drop(columns=[target_column])
        
    y = data[target_column]
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le


def evaluate_workforce_model(model, X_test, y_test, le, use_scaler=False, scaler=None):
    """Comprehensive model evaluation"""
    if use_scaler and scaler is not None:
        X_test_processed = scaler.transform(X_test)
    else:
        X_test_processed = X_test
    
    y_pred = model.predict(X_test_processed)
    y_pred_original = le.inverse_transform(y_pred)
    y_test_original = le.inverse_transform(y_test)
    
    accuracy = accuracy_score(y_test_original, y_pred_original)
    balanced_acc = balanced_accuracy_score(y_test_original, y_pred_original)
    f1_macro = f1_score(y_test_original, y_pred_original, average='macro')
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'predictions': y_pred_original
    }


def train_workforce_logistic_regression(X_train, y_train, X_test, y_test, le):
    """Train Logistic Regression without class weights"""
    print("📊 Training Logistic Regression...")
    
    # Handle string columns
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    string_columns = X_train_encoded.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        print(f"   Encoding {len(string_columns)} string columns")
        for col in string_columns:
            le_col = LabelEncoder()
            X_train_encoded[col] = le_col.fit_transform(X_train_encoded[col].astype(str))
            
            test_values = X_test_encoded[col].astype(str)
            unseen_mask = ~test_values.isin(le_col.classes_)
            if unseen_mask.any():
                test_values[unseen_mask] = 'UNSEEN_CATEGORY'
                if 'UNSEEN_CATEGORY' not in le_col.classes_:
                    le_col.classes_ = np.append(le_col.classes_, 'UNSEEN_CATEGORY')
            X_test_encoded[col] = le_col.transform(test_values)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    metrics = evaluate_workforce_model(model, X_test_encoded, y_test, le, use_scaler=True, scaler=scaler)
    return model, metrics, scaler


def train_workforce_random_forest(X_train, y_train, X_test, y_test, le):
    """Train Random Forest without class weights"""
    print("📊 Training Random Forest...")
    
    # Handle string columns
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    string_columns = X_train_encoded.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        print(f"   Encoding {len(string_columns)} string columns")
        for col in string_columns:
            le_col = LabelEncoder()
            X_train_encoded[col] = le_col.fit_transform(X_train_encoded[col].astype(str))
            
            test_values = X_test_encoded[col].astype(str)
            unseen_mask = ~test_values.isin(le_col.classes_)
            if unseen_mask.any():
                test_values[unseen_mask] = 'UNSEEN_CATEGORY'
                if 'UNSEEN_CATEGORY' not in le_col.classes_:
                    le_col.classes_ = np.append(le_col.classes_, 'UNSEEN_CATEGORY')
            X_test_encoded[col] = le_col.transform(test_values)
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_encoded, y_train)
    
    metrics = evaluate_workforce_model(model, X_test_encoded, y_test, le)
    return model, metrics, None


def train_workforce_decision_tree(X_train, y_train, X_test, y_test, le):
    """Train Decision Tree without class weights"""
    print("📊 Training Decision Tree...")
    
    # Handle string columns
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    string_columns = X_train_encoded.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        print(f"   Encoding {len(string_columns)} string columns")
        for col in string_columns:
            le_col = LabelEncoder()
            X_train_encoded[col] = le_col.fit_transform(X_train_encoded[col].astype(str))
            
            test_values = X_test_encoded[col].astype(str)
            unseen_mask = ~test_values.isin(le_col.classes_)
            if unseen_mask.any():
                test_values[unseen_mask] = 'UNSEEN_CATEGORY'
                if 'UNSEEN_CATEGORY' not in le_col.classes_:
                    le_col.classes_ = np.append(le_col.classes_, 'UNSEEN_CATEGORY')
            X_test_encoded[col] = le_col.transform(test_values)
    
    model = DecisionTreeClassifier(random_state=42, max_depth=10)
    model.fit(X_train_encoded, y_train)
    
    metrics = evaluate_workforce_model(model, X_test_encoded, y_test, le)
    return model, metrics, None


def train_all_workforce_models(data, target_column='primary_mode_raw', test_size=0.2):
    """Train and compare all models for workforce"""
    print("\n🚀 Training Workforce Models...")
    
    # Prepare data
    X, y, le = prepare_workforce_training_data(data, target_column)
    
    # Check class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"\n📊 Final class distribution for modeling:")
    for class_idx, count in zip(unique_classes, class_counts):
        class_name = le.inverse_transform([class_idx])[0]
        print(f"   {class_name}: {count} samples")
    
    # Use stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print("✅ Used stratified train-test split")
    except ValueError as e:
        print(f"⚠️  Stratified split failed: {e}")
        print("🔄 Using regular train-test split instead")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
    results = {}
    
    # Train all models
    lr_model, lr_metrics, lr_scaler = train_workforce_logistic_regression(X_train, y_train, X_test, y_test, le)
    results['logistic_regression'] = {
        'model': lr_model,
        'metrics': lr_metrics,
        'scaler': lr_scaler,
        'feature_names': X.columns.tolist()
    }
    
    rf_model, rf_metrics, _ = train_workforce_random_forest(X_train, y_train, X_test, y_test, le)
    results['random_forest'] = {
        'model': rf_model,
        'metrics': rf_metrics,
        'scaler': None,
        'feature_names': X.columns.tolist()
    }
    
    dt_model, dt_metrics, _ = train_workforce_decision_tree(X_train, y_train, X_test, y_test, le)
    results['decision_tree'] = {
        'model': dt_model,
        'metrics': dt_metrics,
        'scaler': None,
        'feature_names': X.columns.tolist()
    }
    
    # Print results summary
    print("\n📊 Model Performance Summary:")
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"   ✅ {model_name}")
        print(f"      📈 Accuracy: {metrics['accuracy']:.3f}")
        print(f"      ⚖️  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
        print(f"      🎯 F1 Macro: {metrics['f1_macro']:.3f}")
    
    return results


def save_best_workforce_model(results, filename='best_workforce_travel_model.pkl'):
    """Save the best performing workforce model"""
    # Use balanced accuracy as primary metric for imbalanced data
    best_model_name = max(results.items(), key=lambda x: x[1]['metrics']['balanced_accuracy'])[0]
    best_model_data = results[best_model_name]
    
    import joblib
    model_artifacts = {
        'model': best_model_data['model'],
        'model_type': best_model_name,
        'feature_names': best_model_data['feature_names'],
        'scaler': best_model_data['scaler'],
        'metrics': best_model_data['metrics']
    }
    
    joblib.dump(model_artifacts, filename)
    print(f"💾 Best workforce model ({best_model_name}) saved as '{filename}'")
    
    return best_model_name, best_model_data


def main():
    """Main execution function for workforce prediction"""
    print("=" * 60)
    print("💼 WORKFORCE TRAVEL MODE PREDICTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Data Preparation
    print("\n1️⃣ DATA PREPARATION")
    print("-" * 30)
    
    data = load_data('workforce_data.csv')
    if data is None:
        return
    
    workforce_data = filter_workforce_subgroup(data)
    workforce_data = handle_rare_classes(workforce_data, min_samples=10)
    processed_data = preprocess_workforce_features(workforce_data)
    
    # Step 2: Feature Analysis
    print("\n2️⃣ FEATURE ANALYSIS")
    print("-" * 30)
    
    correlations = analyze_workforce_correlations(processed_data)
    selected_features = select_workforce_features(processed_data)
    
    # Step 3: Model Training
    print("\n3️⃣ MODEL TRAINING & COMPARISON")
    print("-" * 30)
    
    results = train_all_workforce_models(processed_data)
    
    # Step 4: Results Summary
    print("\n4️⃣ FINAL RESULTS SUMMARY")
    print("-" * 30)
    
    best_model_name, best_model_data = save_best_workforce_model(results)
    
    metrics = best_model_data['metrics']
    print(f"\n🏆 BEST WORKFOeRCE MODEL: {best_model_name.upper()}")
    print(f"   📈 Accuracy: {metrics['accuracy']:.3f}")
    print(f"   ⚖️  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"   🎯 F1 Macro: {metrics['f1_macro']:.3f}")
    
    print("\n✅ Workforce pipeline execution completed!")


if __name__ == "__main__":
    main()