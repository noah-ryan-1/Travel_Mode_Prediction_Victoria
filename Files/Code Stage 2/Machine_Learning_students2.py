# -*- coding: utf-8 -*-
"""
Student Travel Mode Prediction Pipeline
Methodologically sound implementation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import chi2_contingency, f_oneway
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


def filter_student_subgroup(data):
    """Filter dataset for student population only"""
    student_data = data[data['occupation_5'] == 'Student'].copy()
    print(f"✅ Student subgroup filtered: {student_data.shape}")
    
    # Analyze target variable distribution
    target_dist = student_data['primary_mode_raw'].value_counts(normalize=True) * 100
    print("\n🎯 Target Variable Distribution:")
    for mode, percentage in target_dist.items():
        print(f"   {mode}: {percentage:.1f}%")
        
    return student_data


def handle_rare_classes(student_data, target_column='primary_mode_raw', min_samples=2):
    """Handle rare classes by either grouping them or removing them"""
    processed_data = student_data.copy()
    
    # Count samples per class
    class_counts = processed_data[target_column].value_counts()
    print(f"\n📊 Class distribution before handling rare classes:")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count} samples")
    
    # Identify rare classes
    rare_classes = class_counts[class_counts < min_samples].index.tolist()
    
    if rare_classes:
        print(f"\n⚠️  Rare classes detected (less than {min_samples} samples): {rare_classes}")
        
        # Option 1: Group rare classes into "Other" category
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


def cramers_v(x, y):
    """Calculate Cramér's V for categorical-categorical relationships"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def analyze_associations(data, target_column='primary_mode_raw'):
    """
    ✅ CORRECTED: Analyze feature associations using appropriate measures
    Returns: Series with association strengths (same format as original)
    """
    results = {}
    
    for feature in data.columns:
        if feature != target_column:
            # Numerical feature vs Categorical target → ANOVA F-statistic
            if pd.api.types.is_numeric_dtype(data[feature]):
                # Remove NaN values for ANOVA
                valid_data = data[[feature, target_column]].dropna()
                if len(valid_data) > 0:
                    groups = [valid_data[valid_data[target_column] == category][feature] 
                             for category in valid_data[target_column].unique()]
                    # Ensure all groups have data and at least 2 samples for ANOVA
                    if all(len(group) >= 2 for group in groups) and len(groups) >= 2:
                        try:
                            f_stat, _ = f_oneway(*groups)
                            results[feature] = f_stat
                        except:
                            results[feature] = 0
                    else:
                        results[feature] = 0
                else:
                    results[feature] = 0
            
            # Categorical feature vs Categorical target → Cramér's V  
            elif pd.api.types.is_object_dtype(data[feature]):
                valid_data = data[[feature, target_column]].dropna()
                if len(valid_data) > 0:
                    try:
                        cramers_v_value = cramers_v(valid_data[feature], valid_data[target_column])
                        results[feature] = cramers_v_value
                    except:
                        results[feature] = 0
                else:
                    results[feature] = 0
    
    # Convert to Series sorted by association strength
    target_associations = pd.Series(results).sort_values(ascending=False)
    
    print("\n🔍 Top feature associations with target:")
    for feature, strength in target_associations.head(10).items():
        print(f"   {feature}: {strength:.3f}")
        
    return target_associations


def preprocess_features(student_data, target_column='primary_mode_raw'):
    """Handle feature engineering and preprocessing for student data"""
    # Create a copy for preprocessing
    processed_data = student_data.copy()
    
    # Drop constant and irrelevant features for students
    columns_to_drop = [
        'occupation_5',  # Constant for students
        'persid', 'hhid', 'persno',  # ID columns - not predictive
        'hhpoststratweight', 'perspoststratweight',  # Weight columns - not for prediction
        'travel_mode_3' # Processed form of specific travel modes
    ]
    processed_data = processed_data.drop(columns=columns_to_drop, errors='ignore')
    
    # Separate features and target BEFORE encoding
    y = processed_data[target_column]
    X = processed_data.drop(columns=[target_column])
    
    # Handle categorical variables using one-hot encoding
    categorical_columns = X.select_dtypes(include=['object']).columns
    
    # DEBUG: Print what categorical columns we found
    print(f"🔍 Found categorical columns: {list(categorical_columns)}")
    
    # Also exclude other non-predictive categorical columns
    exclude_columns = ['homelga', 'hhinc_group', 'anzsco1', 'homesubregion_asgs_person', 'travel_mode_3']
    categorical_columns = [col for col in categorical_columns if col not in exclude_columns]
    
    print(f"🔍 Encoding categorical columns: {list(categorical_columns)}")
    
    # Use one-hot encoding for categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    # DEBUG: Check if there are any remaining string columns
    remaining_string_cols = X_encoded.select_dtypes(include=['object']).columns
    if len(remaining_string_cols) > 0:
        print(f"⚠️  WARNING: Still have string columns after encoding: {list(remaining_string_cols)}")
        print("🔄 Converting remaining string columns using label encoding...")
        
        for col in remaining_string_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    
    # Recombine with target
    processed_data = pd.concat([X_encoded, y], axis=1)
    
    print(f"✅ Features preprocessed. Final shape: {processed_data.shape}")
    print(f"✅ Data types after preprocessing:")
    print(processed_data.dtypes.value_counts())
    
    return processed_data


def select_relevant_features(data, target_column='primary_mode_raw', k=10):
    """Select most relevant features using mutual information"""
    from sklearn.feature_selection import mutual_info_classif
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Ensure all data is numeric (should be after preprocessing)
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Use mutual information which works better with categorical targets
    mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
    
    # Create feature scores dictionary
    feature_scores = dict(zip(X_numeric.columns, mi_scores))
    
    # Select top k features with scores > 0
    selected_features_with_scores = [(feat, score) for feat, score in feature_scores.items() if score > 0]
    selected_features_with_scores = sorted(selected_features_with_scores, key=lambda x: x[1], reverse=True)[:k]
    
    # Extract just the feature names for return
    selected_feature_names = [feat for feat, score in selected_features_with_scores]
    
    print(f"\n🎯 Selected {len(selected_feature_names)} most relevant features using Mutual Information:")
    for feature, score in selected_features_with_scores[:10]:
        print(f"   {feature}: {score:.4f}")
        
    return selected_feature_names


def prepare_training_data(data, target_column='primary_mode_raw', features=None):
    """Prepare features and target for training"""
    if features:
        X = data[features]
    else:
        X = data.drop(columns=[target_column])
        
    y = data[target_column]
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le


def evaluate_model(model, X_test, y_test, le, use_scaler=False, scaler=None):
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


def train_logistic_regression(X_train, y_train, X_test, y_test, le):
    """Train and evaluate Logistic Regression model"""
    print("📊 Training Logistic Regression...")
    
    # Features should already be numeric after preprocessing
    # Just scale them for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    metrics = evaluate_model(model, X_test, y_test, le, use_scaler=True, scaler=scaler)
    return model, metrics, scaler


def train_random_forest(X_train, y_train, X_test, y_test, le):
    """Train and evaluate Random Forest model"""
    print("📊 Training Random Forest...")
    
    # Features should already be numeric after preprocessing
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    metrics = evaluate_model(model, X_test, y_test, le)
    return model, metrics, None


def train_decision_tree(X_train, y_train, X_test, y_test, le):
    """Train and evaluate Decision Tree model"""
    print("📊 Training Decision Tree...")
    
    # Features should already be numeric after preprocessing
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    metrics = evaluate_model(model, X_test, y_test, le)
    return model, metrics, None


def train_all_models(data, target_column='primary_mode_raw', test_size=0.2):
    """Train and compare all models"""
    print("\n🚀 Training Models...")
    
    # Prepare data
    X, y, le = prepare_training_data(data, target_column)
    
    # Check if we have enough samples for stratified split
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"\n📊 Final class distribution for modeling:")
    for class_idx, count in zip(unique_classes, class_counts):
        class_name = le.inverse_transform([class_idx])[0]
        print(f"   {class_name}: {count} samples")
    
    # Use regular train-test split if stratified fails
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
    
    # Train Logistic Regression
    lr_model, lr_metrics, lr_scaler = train_logistic_regression(X_train, y_train, X_test, y_test, le)
    results['logistic_regression'] = {
        'model': lr_model,
        'metrics': lr_metrics,
        'scaler': lr_scaler,
        'feature_names': X.columns.tolist()
    }
    
    # Train Random Forest
    rf_model, rf_metrics, _ = train_random_forest(X_train, y_train, X_test, y_test, le)
    results['random_forest'] = {
        'model': rf_model,
        'metrics': rf_metrics,
        'scaler': None,
        'feature_names': X.columns.tolist()
    }
    
    # Train Decision Tree
    dt_model, dt_metrics, _ = train_decision_tree(X_train, y_train, X_test, y_test, le)
    results['decision_tree'] = {
        'model': dt_model,
        'metrics': dt_metrics,
        'scaler': None,
        'feature_names': X.columns.tolist()
    }
    
    # Print results
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"   ✅ {model_name} trained successfully")
        print(f"      📈 Accuracy: {metrics['accuracy']:.3f}")
        print(f"      ⚖️  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
        print(f"      🎯 F1 Macro: {metrics['f1_macro']:.3f}")
    
    return results


def save_best_model(results, filename='best_student_travel_model.pkl'):
    """Save the best performing model"""
    best_model_name = max(results.items(), key=lambda x: x[1]['metrics']['accuracy'])[0]
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
    print(f"💾 Best model ({best_model_name}) saved as '{filename}'")
    
    return best_model_name, best_model_data


def main():
    """Main execution function"""
    print("=" * 60)
    print("🎓 STUDENT TRAVEL MODE PREDICTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Data Preparation
    print("\n1️⃣ DATA PREPARATION")
    print("-" * 30)
    
    data = load_data('students_data.csv')
    if data is None:
        return
    
    student_data = filter_student_subgroup(data)
    
    # Handle rare classes before preprocessing
    student_data = handle_rare_classes(student_data)
    
    processed_data = preprocess_features(student_data)
    
    # Step 2: Feature Analysis
    print("\n2️⃣ FEATURE ANALYSIS")
    print("-" * 30)
    
    # ✅ CORRECTED: Use appropriate association measures
    associations = analyze_associations(processed_data)
    selected_features = select_relevant_features(processed_data)
    
    # Step 3: Model Training
    print("\n3️⃣ MODEL TRAINING & COMPARISON")
    print("-" * 30)
    
    results = train_all_models(processed_data)
    
    # Step 4: Results Summary and Model Saving
    print("\n4️⃣ FINAL RESULTS SUMMARY")
    print("-" * 30)
    
    best_model_name, best_model_data = save_best_model(results)
    
    metrics = best_model_data['metrics']
    print(f"\n🏆 BEST MODEL: {best_model_name.upper()}")
    print(f"   📈 Accuracy: {metrics['accuracy']:.3f}")
    print(f"   ⚖️  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"   🎯 F1 Macro: {metrics['f1_macro']:.3f}")
    
    print("\n✅ Pipeline execution completed!")


if __name__ == "__main__":
    main()