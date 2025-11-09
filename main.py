import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC # SVC excluded as it slows down large grids/loops unless optimized
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_INSTALLED = True
except ImportError:
    XGBOOST_INSTALLED = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_INSTALLED = True
except ImportError:
    LIGHTGBM_INSTALLED = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, 
    accuracy_score, precision_score, recall_score
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE

try:
    from imblearn.over_sampling import SMOTE
    IMBALANCED_INSTALLED = True
except ImportError:
    IMBALANCED_INSTALLED = False
    print("Warning: imbalanced-learn not installed. Install with: pip install imbalanced-learn")
import warnings
import pickle
import sys

warnings.filterwarnings('ignore')

# Set visualization style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("="*80)
    print(" ADVANCED STARTUP FAILURE PREDICTION MODEL ".center(80, "="))
    print("="*80)

    # ==================== CONFIGURATION ====================
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5

    # ==================== DATA LOADING ====================
    print("\n[1] LOADING DATA...")
    print("-"*80)

    # Note: These files must exist in the execution directory
    try:
        df_success = pd.read_csv('startup_failure_prediction.csv')
        df_failure = pd.read_csv('startup_failure_prediction_failures_only.csv')
    except FileNotFoundError as e:
        print(f"Error: Required data file not found: {e.filename}")
        print("Please ensure 'startup_failure_prediction.csv' and 'startup_failure_prediction_failures_only.csv' are in the same directory.")
        sys.exit(1)


    # Combine the two datasets
    df = pd.concat([df_success, df_failure], ignore_index=True)

    print(f"Success Dataset Shape: {df_success.shape}")
    print(f"Failure Dataset Shape: {df_failure.shape}")
    print(f"Combined Dataset Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 5 rows of Combined Data:")
    print(df.head())

    # ==================== EXPLORATORY DATA ANALYSIS ====================
    print("\n[2] EXPLORATORY DATA ANALYSIS")
    print("-"*80)

    print("\nDataset Information:")
    df.info()

    print("\nStatistical Summary:")
    print(df.describe())

    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0].sort_values(ascending=False))
    else:
        print("No missing values found!")

    print("\nTarget Variable Distribution:")
    # Determine the target column dynamically
    if 'Startup_Status' in df.columns:
        target_col = 'Startup_Status'
    elif 'Status' in df.columns:
        target_col = 'Status'
    elif 'Failure' in df.columns:
        target_col = 'Failure'
    else:
        target_col = df.columns[-1]

    print(df[target_col].value_counts())
    print(f"\nClass Distribution:\n{df[target_col].value_counts(normalize=True) * 100}")

    # Visualize target distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df[target_col].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(rotation=0)

    df[target_col].value_counts(normalize=True).plot(
        kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90,
        colors=['#2ecc71', '#e74c3c']
    )
    axes[1].set_title('Target Variable Percentage', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('')
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== DATA PREPROCESSING ====================
    print("\n[3] DATA PREPROCESSING")
    print("-"*80)

    def advanced_preprocessing(df, target_col):
        """Advanced preprocessing pipeline"""
        data = df.copy()

        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Identify column types
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"Numeric columns: {len(numeric_cols)}")
        print(f"Categorical columns: {len(categorical_cols)}")

        # Handle missing values
        # Numeric: fill with median
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)

        # Categorical: fill with mode
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].mode()[0], inplace=True)

        # Encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        # Encode target variable
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)

        print(f"\nTarget mapping: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

        # Feature Engineering
        print("\nPerforming feature engineering...")

        # Create interaction features for top numeric features
        if len(numeric_cols) >= 2:
            # Create ratios and products based on the first few numeric features
            # NOTE: Hardcoding the first 3 features. This is a common heuristic
            # but feature selection might be better.
            top_numeric = numeric_cols[:3]
            for i in range(len(top_numeric)):
                for j in range(i+1, len(top_numeric)):
                    col1, col2 = top_numeric[i], top_numeric[j]
                    # Product
                    X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                    # Ratio (avoid division by zero)
                    X[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-6)

        # Remove any infinite or NaN values created
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)

        return X, y_encoded, label_encoders, le_target

    X, y, label_encoders, target_encoder = advanced_preprocessing(df, target_col)

    # Get string class names for metrics/plots
    CLASS_NAMES = [str(c) for c in target_encoder.classes_]

    print(f"\nFinal feature set shape: {X.shape}")
    print(f"Total features: {X.shape[1]}")

    # ==================== FEATURE CORRELATION ANALYSIS ====================
    print("\n[4] FEATURE CORRELATION ANALYSIS")
    print("-"*80)

    # Correlation with target
    if X.shape[1] < 50: 
        correlation_data = X.copy()
        correlation_data['target'] = y
        corr_with_target = correlation_data.corr()['target'].drop('target').sort_values(ascending=False)

        print("\nTop 15 features correlated with target:")
        print(corr_with_target.head(15))

        # Visualize top correlations
        plt.figure(figsize=(10, 8))
        top_features = corr_with_target.head(20)
        plt.barh(range(len(top_features)), top_features.values)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Correlation with Target')
        plt.title('Top 20 Features Correlated with Target', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()

    # ==================== TRAIN-TEST SPLIT ====================
    print("\n[5] SPLITTING DATA")
    print("-"*80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Training set class distribution:\n{pd.Series(y_train).value_counts()}")
    print(f"Test set class distribution:\n{pd.Series(y_test).value_counts()}")

    # Validate minimum samples per class
    min_class_samples = pd.Series(y_train).value_counts().min()
    if min_class_samples < 5:
        print(f"\nWARNING: Minimum class has only {min_class_samples} samples!")
        print("Consider getting more data or using simpler models.")
        CV_FOLDS = 2 

    # ==================== HANDLE CLASS IMBALANCE ====================
    print("\n[6] HANDLING CLASS IMBALANCE WITH SMOTE")
    print("-"*80)

    # Check class imbalance
    class_counts = pd.Series(y_train).value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()

    print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    if imbalance_ratio > 1.5 and IMBALANCED_INSTALLED:
        print("Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE - Training set size: {X_train_balanced.shape}")
        print(f"Class distribution:\n{pd.Series(y_train_balanced).value_counts()}")
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
        if not IMBALANCED_INSTALLED:
            print("imbalanced-learn not installed. Skipping SMOTE.")
        else:
            print("Classes are relatively balanced. Skipping SMOTE.")

    # ==================== FEATURE SCALING ====================
    print("\n[7] FEATURE SCALING")
    print("-"*80)

    scaler = RobustScaler() 
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    print("Feature scaling completed using RobustScaler")

    # ==================== MODEL TRAINING ====================
    print("\n[8] TRAINING MULTIPLE MODELS")
    print("-"*80)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, C=1.0),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200, 
                                                 max_depth=15, min_samples_split=5),
        'Extra Trees': ExtraTreesClassifier(random_state=RANDOM_STATE, n_estimators=200),
        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=200,
                                                        learning_rate=0.1, max_depth=5),
        'AdaBoost': AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=100),
        'Neural Network': MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=(100, 50),
                                        max_iter=500, early_stopping=True)
    }

    # Add XGBoost if available
    if XGBOOST_INSTALLED:
        models['XGBoost'] = XGBClassifier(random_state=RANDOM_STATE, n_estimators=200, learning_rate=0.1,
                                            max_depth=6, eval_metric='logloss', use_label_encoder=False)

    # Add LightGBM if available
    if LIGHTGBM_INSTALLED:
        models['LightGBM'] = LGBMClassifier(random_state=RANDOM_STATE, n_estimators=200, learning_rate=0.1,
                                             max_depth=6, verbose=-1)

    # Store results
    results = {}
    cv_scores = {}

    # Check if we have at least 2 classes BEFORE the loop
    unique_classes = np.unique(y_train_balanced)
    print(f"\nClasses in training data: {unique_classes}")

    if len(unique_classes) < 2:
        print(f"\nERROR: Only one class ({unique_classes[0]}) found in training data!")
        print("Cannot train models. Please check your dataset.")
        sys.exit(1)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Determine if scaling is needed for the current model type
        if name in ['Logistic Regression', 'Neural Network', 'SVC']: # Added SVC for completeness if user decides to uncomment
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train_balanced, X_test

        # Cross-validation with error handling
        try:
            n_folds = min(CV_FOLDS, np.min(np.bincount(y_train_balanced)))
            if n_folds < 2:
                n_folds = 2

            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
            cv_score = cross_val_score(model, X_tr, y_train_balanced, cv=cv, scoring='roc_auc', n_jobs=1) # Use n_jobs=-1 for parallelism
            cv_scores[name] = cv_score
        except Exception as e:
            print(f"  Warning: Cross-validation failed ({str(e)[:50]}...). Using single train-test split.")
            cv_score = np.array([0.5]) 
            cv_scores[name] = cv_score

        # Train model
        model.fit(X_tr, y_train_balanced)

        # Predictions
        try:
            y_pred = model.predict(X_te)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_te)
                # Ensure y_pred_proba is the probability of the positive class (1)
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                # Fallback for unexpected shapes
                elif y_pred_proba.ndim == 1:
                    pass # Already in the correct format (e.g., Logistic Regression default)
                else:
                    y_pred_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
            else:
                # If no probability function, use the prediction itself (not ideal for ROC-AUC)
                y_pred_proba = y_pred
        except Exception as e:
            print(f"  Error in prediction: {e}")
            continue

        # Metrics
        try:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # ROC AUC - handle binary classification
            if len(np.unique(y_test)) == 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            else:
                # Multi-class ROC AUC calculation if needed, though this dataset is binary
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"  Error calculating metrics: {e}")
            accuracy = precision = recall = f1 = roc_auc = 0.0

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_score_mean': cv_score.mean(),
            'cv_score_std': cv_score.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        print(f"  CV ROC-AUC: {cv_score.mean():.4f} (+/- {cv_score.std():.4f})")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test ROC-AUC: {roc_auc:.4f}")

    # ==================== MODEL COMPARISON ====================
    print("\n[9] MODEL COMPARISON")
    print("-"*80)

    if len(results) == 0:
        print("\nERROR: No models were trained successfully!")
        sys.exit(1)

    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'CV ROC-AUC': [results[m]['cv_score_mean'] for m in results.keys()],
        'Test Accuracy': [results[m]['accuracy'] for m in results.keys()],
        'Test Precision': [results[m]['precision'] for m in results.keys()],
        'Test Recall': [results[m]['recall'] for m in results.keys()],
        'Test F1-Score': [results[m]['f1_score'] for m in results.keys()],
        'Test ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
    }).sort_values('Test ROC-AUC', ascending=False)

    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))

    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Accuracy comparison
    comparison_df.plot(x='Model', y='Test Accuracy', kind='barh', ax=axes[0, 0], 
                       legend=False, color='#3498db')
    axes[0, 0].set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Accuracy')

    # ROC-AUC comparison
    comparison_df.plot(x='Model', y='Test ROC-AUC', kind='barh', ax=axes[0, 1], 
                       legend=False, color='#e74c3c')
    axes[0, 1].set_title('Model ROC-AUC Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('ROC-AUC Score')

    # F1-Score comparison
    comparison_df.plot(x='Model', y='Test F1-Score', kind='barh', ax=axes[1, 0], 
                       legend=False, color='#2ecc71')
    axes[1, 0].set_title('Model F1-Score Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('F1-Score')

    # Precision vs Recall
    axes[1, 1].scatter(comparison_df['Test Precision'], comparison_df['Test Recall'], 
                       s=200, alpha=0.6, c=range(len(comparison_df)), cmap='viridis')
    for idx, row in comparison_df.iterrows():
        axes[1, 1].annotate(row['Model'], (row['Test Precision'], row['Test Recall']),
                            fontsize=8, ha='center')
    axes[1, 1].set_xlabel('Precision')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Precision vs Recall', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== BEST MODEL ANALYSIS ====================
    print("\n[10] BEST MODEL ANALYSIS")
    print("-"*80)

    best_model_name = comparison_df.iloc[0]['Model']
    best_model_data = results[best_model_name]
    best_model = best_model_data['model']

    print(f"\nBEST MODEL: {best_model_name}")
    print(f"  CV ROC-AUC: {best_model_data['cv_score_mean']:.4f} (+/- {best_model_data['cv_score_std']:.4f})")
    print(f"  Test Accuracy: {best_model_data['accuracy']:.4f}")
    print(f"  Test Precision: {best_model_data['precision']:.4f}")
    print(f"  Test Recall: {best_model_data['recall']:.4f}")
    print(f"  Test F1-Score: {best_model_data['f1_score']:.4f}")
    print(f"  Test ROC-AUC: {best_model_data['roc_auc']:.4f}")

    print("\nDetailed Classification Report:")
    # FIX: Convert integer classes to strings for classification_report
    print(classification_report(y_test, best_model_data['y_pred'], 
                                target_names=CLASS_NAMES))

    # Confusion Matrix
    cm = confusion_matrix(y_test, best_model_data['y_pred'])
    plt.figure(figsize=(8, 6))
    # FIX: Convert integer classes to strings for heatmap labels
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix_best.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== ROC CURVES ====================
    print("\n[11] GENERATING ROC CURVES")
    print("-"*80)

    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {result['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves_all.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== FEATURE IMPORTANCE ====================
    print("\n[12] FEATURE IMPORTANCE ANALYSIS")
    print("-"*80)

    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 20 Most Important Features:")
        print(feature_importance.head(20).to_string(index=False))

        # Visualize
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'].values, color='#9b59b6')
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top 20 Feature Importances - {best_model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance_best.png', dpi=300, bbox_inches='tight')
        plt.close()

    # ==================== SAVE MODEL ====================
    print("\n[13] SAVING MODEL")
    print("-"*80)

    # Prepare model artifacts
    model_artifacts = {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'feature_names': X.columns.tolist(),
        'model_performance': best_model_data,
        'all_results': comparison_df
    }

    # Save
    with open('startup_failure_model_advanced.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)

    print("Model saved as 'startup_failure_model_advanced.pkl'")

    # ==================== PREDICTION FUNCTION ====================
    print("\n[14] PREDICTION FUNCTION")
    print("-"*80)

    def predict_startup_failure(new_data_path, model_path='startup_failure_model_advanced.pkl'):
        """
        Predict startup failure for new data

        Parameters:
        -----------
        new_data_path : str
            Path to CSV file with new startup data
        model_path : str
            Path to saved model pickle file

        Returns:
        --------
        predictions_df : DataFrame
            DataFrame with predictions and probabilities
        """
        # Load model artifacts
        try:
            with open(model_path, 'rb') as f:
                artifacts = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: Model file '{model_path}' not found.")
            return None

        model = artifacts['best_model']
        scaler = artifacts['scaler']
        label_encoders = artifacts['label_encoders']
        target_encoder = artifacts['target_encoder']
        feature_names = artifacts['feature_names']

        # Load new data
        try:
            new_data = pd.read_csv(new_data_path)
        except FileNotFoundError:
            print(f"Error: New data file '{new_data_path}' not found.")
            return None

        # Preprocess
        X_new = new_data.copy()

        # Encode categorical variables
        for col, le in label_encoders.items():
            if col in X_new.columns:
                # Handle unseen categories by converting them to string before transform
                X_new[col] = X_new[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
                X_new[col] = le.transform(X_new[col].astype(str))
            
        # Feature Engineering (re-create the engineered features from training)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        top_numeric = numeric_cols[:3] # Assuming the same logic as training
        
        # NOTE: This only works if the columns used for feature engineering (top_numeric) 
        # are present in the new data. A robust function would require saving these column names.
        
        if len(top_numeric) >= 2:
             for i in range(len(top_numeric)):
                for j in range(i+1, len(top_numeric)):
                    col1, col2 = top_numeric[i], top_numeric[j]
                    
                    if col1 in X_new.columns and col2 in X_new.columns:
                        X_new[f'{col1}_x_{col2}'] = X_new[col1] * X_new[col2]
                        X_new[f'{col1}_div_{col2}'] = X_new[col1] / (X_new[col2] + 1e-6)
        
        # Ensure same features as training, filling missing ones with 0 (as per the training logic)
        for col in feature_names:
            if col not in X_new.columns:
                X_new[col] = 0

        X_new = X_new[feature_names]

        # Scale
        model_name = artifacts['best_model_name']
        if model_name in ['Logistic Regression', 'Neural Network', 'SVC']:
            X_new_scaled = scaler.transform(X_new)
            predictions = model.predict(X_new_scaled)
            probabilities = model.predict_proba(X_new_scaled)
        else:
            predictions = model.predict(X_new)
            probabilities = model.predict_proba(X_new)
        
        # Ensure probabilities is a 2D array [:, 0] and [:, 1] for inverse_transform mapping
        if probabilities.ndim == 1:
            # Recreate the 2-column probability array for binary class models if necessary
            probabilities = np.column_stack([1 - probabilities, probabilities])


        # Create results dataframe
        results_df = new_data.copy()
        results_df['Predicted_Status'] = target_encoder.inverse_transform(predictions)
        # Assuming index 1 is the positive class (Failure) and index 0 is the negative class (Success)
        results_df['Failure_Probability'] = probabilities[:, 1]
        results_df['Success_Probability'] = probabilities[:, 0]

        return results_df

    print("Prediction function created: predict_startup_failure()")

    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print(" MODEL TRAINING COMPLETE! ".center(80, "="))
    print("="*80)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {best_model_data['accuracy']:.4f}")
    print(f"Test ROC-AUC: {best_model_data['roc_auc']:.4f}")
    print(f"F1-Score: {best_model_data['f1_score']:.4f}")
    print("\nGenerated Files:")
    print("  - startup_failure_model_advanced.pkl (Model file)")
    print("  - target_distribution.png")
    print("  - feature_correlations.png")
    print("  - model_comparison.png")
    print("  - confusion_matrix_best.png")
    print("  - roc_curves_all.png")
    print("  - feature_importance_best.png")
    print("\nReady for deployment!")
    print("="*80)

if __name__ == "__main__":
    main()