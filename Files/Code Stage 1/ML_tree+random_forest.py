import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION & DATA LOADING
# =============================================================================

def load_and_configure_data():
    """Load data and set global configuration"""
    DATA_PATH = "vista_stage2_collapsed.csv"
    df = pd.read_csv(DATA_PATH)
    
    # Configuration
    target_column = 'travel_mode_3'
    features_to_drop = ['persid', 'hhid', 'primary_mode_raw', 'persno', 
                       'hhpoststratweight', 'perspoststratweight', 
                       # Features dropped based on feature selection:
                    #    'homelga', 'region_3', # Due to likely overfitting
                       'agegroup', 'studying'
                       ]
    
    return df, target_column, features_to_drop

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

def get_model_configurations():
    """
    Define all model types to compare with their configurations
    Easy to add new models here. Models are abbreviated for readability.
    """

    return {
        'decision_tree': {
            'name': 'Standard Decision Tree',
            'class': DecisionTreeClassifier,
            'params': {'random_state': 42, 'max_depth': 15}
        },
        'balanced_tree': {
            'name': 'Balanced Decision Tree', 
            'class': DecisionTreeClassifier,
            'params': {'random_state': 42, 'max_depth': 15, 'class_weight': 'balanced'}
        },
        'random_forest': {
            'name': 'Random Forest',
            'class': RandomForestClassifier, 
            'params': {'n_estimators': 100, 'random_state': 42, 'max_depth': 15}
        },
        'balanced_forest': {
            'name': 'Balanced Random Forest',
            'class': RandomForestClassifier,
            'params': {'n_estimators': 100, 'random_state': 42, 'max_depth': 15, 'class_weight': 'balanced'}
        }
    }

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def preprocess_data(df, target_col, features_to_drop):
    """Preprocess the main dataset for initial analysis"""

    # Rename annoying column names :) (To implement properly in step 1)
    rename = {"homesubregion_asgs_person":'subreg_person',"homeregion_asgs_person":'reg_person',
        "homesubregion_asgs_hh":'home_subreg',"homeregion_asgs_hh":'home_reg'}
    
    names = [rename.get(name, name) for name in df.columns]
    print(names)

    X = df.drop(columns=[target_col] + features_to_drop)
    y = df[target_col]
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Handle missing values
    X = X.fillna(X.mode().iloc[0])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    return X, X_train, X_test, y, y_encoded, y_train, y_test, le_target, label_encoders

# =============================================================================
# FEATURE COMPARISON ENGINE
# =============================================================================

def create_comparison_datasets(df, target_col, feature1, feature2, features_to_drop):
    """
    Create datasets for feature comparison without data leakage
    Returns base features list for transparency
    """
    base_features = [col for col in df.columns 
                     if col not in [target_col, feature1, feature2] 
                     and col not in features_to_drop]
    
    return base_features

def evaluate_model_configuration(X_data, y_encoded, model_config, dataset_name, cv_folds=5):
    """
    Evaluate a single model configuration on a specific dataset
    """
    # Initialize model with specified configuration
    model_class = model_config['class']
    model_params = model_config['params'].copy()
    model = model_class(**model_params)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_data, y_encoded, cv=cv_folds, scoring='accuracy')
    
    # Train-test split for detailed metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Fit model and make predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Calculate comprehensive metrics Remove if unecessary
    # precision, recall, f1, support = precision_recall_fscore_support(
    #     y_test, y_pred, average='weighted', zero_division=0
    # )
    # Overall Metrics
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Feature importance (if available)
    top_features = pd.DataFrame({
        'feature': X_data.columns,
        'importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else [0] * len(X_data.columns)
    }).sort_values('importance', ascending=False).head(10)
    
    # Compile results
    results = {
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'test_accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'classification_report': class_report,
        'top_features': top_features,
        'model_name': model_config['name'],
        'model_type': model_config['class'].__name__,
        'cv_scores': cv_scores  # Store full CV scores for statistical testing
    }
    
    return results, model

def print_classification_report(results, config_name, model_name):
    """
    Print a formatted classification report to the console to allow for easier insight.
    """
    print(f"\n📊 Classification Report - {model_name} | {config_name}")
    print("=" * 60)
    
    report = results['classification_report']
    
    # Print header
    print(f"{'':<12} {'precision':<10} {'recall':<10} {'f1-score':<10} {'support':<10}")
    print("-" * 60)

    # Printing each class
    for class_name, metrics in report.items():
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue # Explain this step - hard for human interprability
        
        # Print specific items for each class
        print(f"{class_name:<12} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} "
              f"{metrics['f1-score']:<10.2f} {metrics['support']:<10}")
        
    # Print accuracy and averages
    print("-" * 60)
    print(f"{'accuracy':<12} {'':<10} {'':<10} {'':<10} {report['accuracy']:<10.0f}")
    print(f"{'macro avg':<12} {report['macro avg']['precision']:<10.2f} {report['macro avg']['recall']:<10.2f} "
          f"{report['macro avg']['f1-score']:<10.2f} {report['macro avg']['support']:<10.0f}")
    print(f"{'weighted avg':<12} {report['weighted avg']['precision']:<10.2f} {report['weighted avg']['recall']:<10.2f} "
          f"{report['weighted avg']['f1-score']:<10.2f} {report['weighted avg']['support']:<10.0f}")

def run_comprehensive_comparison(df, target_col, feature1, feature2, features_to_drop, cv_folds=5):
    """
    Run comprehensive comparison across all model types and feature configurations
    """

    # Get base features for transparency
    base_features = create_comparison_datasets(df, target_col, feature1, feature2, features_to_drop)
    
    print("🔍 BASE FEATURES ANALYSIS:")
    print(f"Total base features: {len(base_features)}")
    print("Key base features:", [f for f in base_features if f in [
        'veh_own_3', 'totalvehs', 'agegroup', 'studying', 'anywork', 
        'homelga', 'region_3', 'anzsco1'
    ]]) # This srsly has to be fixed because it is not always correct if we convert one to an independent variable
    print()
    
    # Encode target variable once for consistency
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(df[target_col])
    
    # Get all model configurations
    model_configs = get_model_configurations()
    
    # Store all results
    all_results = {}
    
    # Test configurations
    feature_configs = {
        'Base Only': base_features,
        f'Base + {feature1}': base_features + [feature1],
        f'Base + {feature2}': base_features + [feature2],
        f'Base + Both': base_features + [feature1, feature2]
    }
    
    # Iterate through each model type
    for model_key, model_config in model_configs.items():
        print(f"\n{'='*80}")
        print(f"🧠 MODEL: {model_config['name']}")
        print(f"{'='*80}")
        
        model_results = {}
        
        # Iterate through each feature configuration
        for config_name, features in feature_configs.items():
            print(f"\n📊 Testing: {config_name}")
            
            # Prepare dataset for this configuration
            X_config = df[features].copy()
            
            # Encode categorical variables for this specific dataset
            for col in X_config.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X_config[col] = le.fit_transform(X_config[col].astype(str))
            
            # Handle missing values
            X_config = X_config.fillna(X_config.mode().iloc[0] if len(X_config.mode()) > 0 else 0)
            
            # Evaluate this configuration
            results, model = evaluate_model_configuration(
                X_config, y_encoded, model_config, config_name, cv_folds
            )
            
            model_results[config_name] = results
            
            # Print overall metrics from results
            print(f"   ✅ Accuracy: {results['test_accuracy']:.4f}, F1: {results['weighted_f1']:.4f}")

            # Now print the detailed classification report
            print_classification_report(results, config_name, model_config['name']) 
        
            print(f"   📈 Top 3 features: {list(results['top_features']['feature'].head(3))}")
        
        all_results[model_key] = model_results
    
    return all_results, le_target, base_features

# =============================================================================
# COMPREHENSIVE VISUALIZATION
# =============================================================================

def create_comprehensive_visualization(all_results, feature1, feature2):
    """
    Create comprehensive visualization comparing all models and feature configurations
    """
    # Create abbreviations for names
    MODEL_ABBREVIATIONS = {
        'Standard Decision Tree': 'Std DT',
        'Balanced Decision Tree': 'Bal DT', 
        'Random Forest': 'RF',
        'Balanced Random Forest': 'Bal RF'
    }

    CONFIG_ABBREVIATIONS = {
        'Base Only': 'Base',
        f'Base + {feature1}': 'Base+IncGrp',
        f'Base + {feature2}': 'Base+Inc3', 
        'Base + Both': 'Base+Both'
    }

    FEATURE_ABBREVIATIONS = {
        'homesubregion_asgs_person': 'subreg_p',
        'homeregion_asgs_person': 'reg_p',
        'homesubregion_asgs_hh': 'subreg_hh',
        'homeregion_asgs_hh': 'reg_hh',
        'perspoststratweight': 'p_weight',
        'hhpoststratweight': 'hh_weight',
        'hhinc_group': 'inc_grp',
        'hh_income_3': 'inc3'
    }
    
    # Prepare data for plotting
    plot_data = []
    importance_data = []
    
    for model_key, model_results in all_results.items():
        for config_name, results in model_results.items():
            # Use abbreviated names
            short_model = MODEL_ABBREVIATIONS.get(results['model_name'], results['model_name'])
            short_config = CONFIG_ABBREVIATIONS.get(config_name, config_name)
            
            # Performance data - FIXED: Use correct metric names
            plot_data.append({
                'Model': short_model,
                'Configuration': short_config,
                'Test_Accuracy': results['test_accuracy'],
                'F1_Score': results['weighted_f1'],  # FIXED: was 'weighted_f1'
                'CV_Accuracy': results['cv_accuracy_mean'],
                'CV_Std': results['cv_accuracy_std'],
            })
            
            # Feature importance data with feature abbreviations - FIXED: Handle missing feature_importance
            if 'feature_importance' in results and results['feature_importance']:
                for feature, importance in results['feature_importance'].items():
                    short_feature = FEATURE_ABBREVIATIONS.get(feature, feature)
                    importance_data.append({
                        'Model': short_model,
                        'Configuration': short_config,
                        'Feature': short_feature,
                        'Importance': importance
                    })
            else:
                # Use top_features if feature_importance is not available
                if 'top_features' in results and not results['top_features'].empty:
                    for _, row in results['top_features'].iterrows():
                        short_feature = FEATURE_ABBREVIATIONS.get(row['feature'], row['feature'])
                        importance_data.append({
                            'Model': short_model,
                            'Configuration': short_config,
                            'Feature': short_feature,
                            'Importance': row['importance']
                        })
    
    plot_df = pd.DataFrame(plot_data)
    importance_df = pd.DataFrame(importance_data)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Comprehensive Model Comparison: {feature1} vs {feature2}', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    # 1. Test Accuracy by Model and Configuration
    if not plot_df.empty:
        pivot_acc = plot_df.pivot_table(index='Configuration', columns='Model', values='Test_Accuracy')
        pivot_acc.plot(kind='bar', ax=axes[0,0], width=0.7)
        axes[0,0].set_title('Test Accuracy by Configuration', fontsize=11, fontweight='bold')
        axes[0,0].set_ylabel('Accuracy', fontsize=10)
        axes[0,0].set_xlabel('')
        axes[0,0].tick_params(axis='x', rotation=45, labelsize=9)
        axes[0,0].legend(fontsize=8, bbox_to_anchor=(1.05, 1))
        axes[0,0].grid(True, alpha=0.3, axis='y')
        
    # 2. F1-Score by Model and Configuration
    if not plot_df.empty:
        pivot_f1 = plot_df.pivot_table(index='Configuration', columns='Model', values='F1_Score')
        pivot_f1.plot(kind='bar', ax=axes[0,1], width=0.7)
        axes[0,1].set_title('F1-Score by Configuration', fontsize=11, fontweight='bold')
        axes[0,1].set_ylabel('F1-Score', fontsize=10)
        axes[0,1].set_xlabel('')
        axes[0,1].tick_params(axis='x', rotation=45, labelsize=9)
        axes[0,1].legend(fontsize=8, bbox_to_anchor=(1.05, 1))
        axes[0,1].grid(True, alpha=0.3, axis='y')
    
    # 3. Average Model Performance
    if not plot_df.empty:
        model_performance = plot_df.groupby('Model')[['Test_Accuracy', 'F1_Score']].mean()
        
        x_pos = np.arange(len(model_performance))
        width = 0.35
        
        axes[0,2].bar(x_pos - width/2, model_performance['Test_Accuracy'], width, 
                     label='Accuracy', alpha=0.8, color='skyblue')
        axes[0,2].bar(x_pos + width/2, model_performance['F1_Score'], width, 
                     label='F1-Score', alpha=0.8, color='lightcoral')
        
        axes[0,2].set_title('Average Model Performance', fontsize=11, fontweight='bold')
        axes[0,2].set_xlabel('Model Type', fontsize=10)
        axes[0,2].set_ylabel('Score', fontsize=10)
        axes[0,2].set_xticks(x_pos)
        axes[0,2].set_xticklabels(model_performance.index, rotation=45, fontsize=9)
        axes[0,2].legend(fontsize=9)
        axes[0,2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (acc, f1) in enumerate(zip(model_performance['Test_Accuracy'], model_performance['F1_Score'])):
            axes[0,2].text(i - width/2, acc + 0.01, f'{acc:.3f}', 
                          ha='center', va='bottom', fontsize=8)
            axes[0,2].text(i + width/2, f1 + 0.01, f'{f1:.3f}', 
                          ha='center', va='bottom', fontsize=8)
    
    # 4. Feature Importance Heatmap
    if not importance_df.empty:
        # Get top features across all configurations
        top_features = (importance_df.groupby('Feature')['Importance']
                       .mean().nlargest(8).index)  # Reduced to top 8 for clarity
        
        top_importance_df = importance_df[importance_df['Feature'].isin(top_features)]
        
        if not top_importance_df.empty:
            # Create pivot table for heatmap
            pivot_data = top_importance_df.groupby(['Model', 'Feature'])['Importance'].mean().unstack().fillna(0)
            
            # Create heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=axes[1,0], cbar_kws={'label': 'Importance'}, 
                       linewidths=0.5, linecolor='gray', annot_kws={'size': 7})
            axes[1,0].set_title('Top Feature Importances by Model', fontsize=11, fontweight='bold')
            axes[1,0].set_xlabel('Features', fontsize=10)
            axes[1,0].set_ylabel('Models', fontsize=10)
            axes[1,0].tick_params(axis='x', rotation=45, labelsize=8)
            axes[1,0].tick_params(axis='y', rotation=0, labelsize=8)
    
    else:
        axes[1,0].text(0.5, 0.5, 'No feature importance data', 
                      ha='center', va='center', transform=axes[1,0].transAxes, fontsize=10)
        axes[1,0].set_title('Feature Importance', fontsize=11, fontweight='bold')
        axes[1,0].axis('off')
    
    # 5. Feature Impact Analysis - FIXED: Use correct metric names
    impact_data = []
    for model_key, model_results in all_results.items():
        base_f1 = model_results['Base Only']['weighted_f1']  # FIXED: was 'f1_score'
        short_model = MODEL_ABBREVIATIONS.get(model_results['Base Only']['model_name'], 
                                            model_results['Base Only']['model_name'])
        
        for config_name, results in model_results.items():
            if config_name != 'Base Only':
                feature_name = config_name.split(' + ')[1] if ' + ' in config_name else config_name
                short_feature = FEATURE_ABBREVIATIONS.get(feature_name, feature_name)
                
                improvement = results['weighted_f1'] - base_f1  # FIXED: was 'f1_score'
                impact_data.append({
                    'Model': short_model,
                    'Feature': short_feature,
                    'F1_Improvement': improvement
                })
    
    impact_df = pd.DataFrame(impact_data)
    if not impact_df.empty:
        # Use the abbreviated feature names
        impact_df['Feature'] = impact_df['Feature'].map(lambda x: FEATURE_ABBREVIATIONS.get(x, x))
        
        sns.barplot(data=impact_df, x='Feature', y='F1_Improvement', hue='Model',
                   ax=axes[1,1], palette='Set2')
        axes[1,1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1,1].set_title('F1-Score Improvement Over Base', fontsize=11, fontweight='bold')
        axes[1,1].set_xlabel('Feature Added', fontsize=10)
        axes[1,1].set_ylabel('Δ F1-Score', fontsize=10)
        axes[1,1].tick_params(axis='x', rotation=45, labelsize=9)
        axes[1,1].legend(fontsize=8, bbox_to_anchor=(1.05, 1))
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels for significant improvements only
        for container in axes[1,1].containers:
            axes[1,1].bar_label(container, fmt='%+.3f', padding=2, fontsize=7, 
                              label_type='edge' if any(abs(x) > 0.01 for x in container.datavalues) else 'center')
    
    else:
        axes[1,1].text(0.5, 0.5, 'No impact data available', 
                      ha='center', va='center', transform=axes[1,1].transAxes, fontsize=10)
        axes[1,1].set_title('Feature Impact Analysis', fontsize=11, fontweight='bold')
        axes[1,1].axis('off')
    
    # 6. Performance Summary Table
    if not plot_df.empty:
        # Calculate model rankings
        model_ranking = plot_df.groupby('Model')['F1_Score'].mean().sort_values(ascending=False)
        
        # Create table data
        table_data = []
        for i, (model, score) in enumerate(model_ranking.items(), 1):
            table_data.append([f"{i}", model, f"{score:.4f}"])
        
        # Create table
        table = axes[1,2].table(cellText=table_data,
                               colLabels=['Rank', 'Model', 'Avg F1'],
                               cellLoc='center',
                               loc='center',
                               bbox=[0.2, 0.2, 0.6, 0.6])  # Smaller bbox
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # Style header
        for i in range(3):
            table[(0, i)].set_facecolor('#4d4d4d')
            table[(0, i)].set_text_props(weight='bold', color='white', size=9)
        
        # Style alternating rows
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(3):
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        axes[1,2].set_title('Model Performance Ranking', fontsize=11, fontweight='bold')
        axes[1,2].axis('off')
    
    else:
        axes[1,2].text(0.5, 0.5, 'No performance data', 
                      ha='center', va='center', transform=axes[1,2].transAxes, fontsize=10)
        axes[1,2].set_title('Performance Ranking', fontsize=11, fontweight='bold')
        axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.5, wspace=0.4)
    plt.show()
    
    return plot_df, importance_df

# =============================================================================
# ADAPTABLE MAIN EXECUTION
# =============================================================================

def run_adaptable_comparison(feature1='hhinc_group', feature2='hh_income_3', cv_folds=5):
    """
    Main function that can be easily adapted for different feature comparisons
    """
    print("🚀 COMPREHENSIVE MODEL & FEATURE COMPARISON ANALYSIS")
    print(f"🔍 Comparing features: {feature1} vs {feature2}")
    
    # Load data
    df, target_column, features_to_drop = load_and_configure_data()
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Run comprehensive comparison
    all_results, le_target, base_features = run_comprehensive_comparison(
        df=df,
        target_col=target_column,
        feature1=feature1,
        feature2=feature2,
        features_to_drop=features_to_drop,
        cv_folds=cv_folds
    )
    
    # Create visualizations
    print("\n📊 GENERATING COMPREHENSIVE VISUALIZATIONS...")
    plot_df, importance_df = create_comprehensive_visualization(all_results, feature1, feature2)
    
    return all_results, plot_df, importance_df

# =============================================================================
# EASY ADAPTATION EXAMPLES
# =============================================================================

def compare_different_features():
    """
    Example of how to easily adapt the code for different feature comparisons
    """
    # Example 1: Compare vehicle-related features
    print("EXAMPLE 1: Comparing vehicle features")
    results1, _, _ = run_adaptable_comparison(
        feature1='veh_own_3', 
        feature2='totalvehs',
        cv_folds=5
    )
    
    # Example 2: Compare geographic features  
    print("\n\nEXAMPLE 2: Comparing geographic features")
    results2, _, _ = run_adaptable_comparison(
        feature1='homelga',
        feature2='region_3', 
        cv_folds=5
    )
    
    # Example 3: Compare demographic features
    print("\n\nEXAMPLE 3: Comparing demographic features")
    results3, _, _ = run_adaptable_comparison(
        feature1='agegroup',
        feature2='studying',
        cv_folds=5
    )
    
    return results1, results2, results3


# Add this import at the top
from sklearn.model_selection import GridSearchCV

# =============================================================================
# SIMPLE RANDOM FOREST HYPERPARAMETER TUNING
# =============================================================================

def tune_random_forest_simple(X_train, y_train, param_grid=None, cv_folds=3):
    """
    Simple Random Forest hyperparameter tuning
    
    Parameters:
    - X_train, y_train: Training data
    - param_grid: Dictionary of parameters to tune (if None, uses default grid)
    - cv_folds: Number of cross-validation folds
    """
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    print("🔧 Tuning Random Forest Hyperparameters...")
    print(f"Parameters being tuned: {list(param_grid.keys())}")
    
    # Create and fit the grid search
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Print results
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search

# =============================================================================
# INTEGRATED TUNING FUNCTION FOR YOUR COMPARISON PIPELINE
# =============================================================================

def tune_and_compare_forests(X_train, y_train, X_test, y_test, param_grid=None):
    """
    Tune both regular and balanced Random Forests and compare results
    """
    
    print("🌲 TUNING REGULAR RANDOM FOREST")
    print("=" * 50)
    regular_search = tune_random_forest_simple(X_train, y_train, param_grid)
    
    print("\n🌲 TUNING BALANCED RANDOM FOREST") 
    print("=" * 50)
    # Add class_weight to parameter grid for balanced forest
    if param_grid is None:
        balanced_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        }
    else:
        balanced_param_grid = param_grid.copy()
        if 'class_weight' not in balanced_param_grid:
            balanced_param_grid['class_weight'] = ['balanced', 'balanced_subsample']
    
    balanced_search = tune_random_forest_simple(X_train, y_train, balanced_param_grid)
    
    # Compare performance on test set
    print("\n📊 FINAL COMPARISON ON TEST SET")
    print("=" * 50)
    
    # Regular Forest
    regular_model = regular_search.best_estimator_
    regular_test_score = regular_model.score(X_test, y_test)
    print(f"Regular Forest Test Accuracy: {regular_test_score:.4f}")
    
    # Balanced Forest  
    balanced_model = balanced_search.best_estimator_
    balanced_test_score = balanced_model.score(X_test, y_test)
    print(f"Balanced Forest Test Accuracy: {balanced_test_score:.4f}")
    
    return regular_search, balanced_search

# Example of how to use it in your main execution:
def run_with_tuning(feature1='hhinc_group', feature2='hh_income_3'):
    """
    Run the comparison with hyperparameter tuning
    """
    # Load and preprocess data
    df, target_column, features_to_drop = load_and_configure_data()
    X, X_train, X_test, y, y_encoded, y_train, y_test, le_target, label_encoders = preprocess_data(
        df, target_column, features_to_drop
    )
    
    # Define your custom parameter grid (or use default)
    custom_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Run tuning
    regular_tuned, balanced_tuned = tune_and_compare_forests(
        X_train, y_train, X_test, y_test, custom_param_grid
    )
    
    return regular_tuned, balanced_tuned

# You can call this from your main execution:
if __name__ == "__main__":
    # Run your original comparison
    all_results, plot_df, importance_df = run_adaptable_comparison(
        feature1='homelga', 
        feature2='region_3', 
        cv_folds=5
    )
    
    # Then run tuning on the best feature set
    print("\n" + "="*60)
    print("STARTING HYPERPARAMETER TUNING")
    print("="*60)
    
    regular_tuned, balanced_tuned = run_with_tuning()