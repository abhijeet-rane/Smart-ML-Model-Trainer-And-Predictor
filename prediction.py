import numpy as np
import pandas as pd
import warnings
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

warnings.filterwarnings("ignore")

def train_models(data_path, target_column=None, test_size=0.2, random_state=42, save_dir='models'):
    """
    Train multiple machine learning models on any dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing the dataset
    target_column : str
        Name of the column to predict. If None, the user will be prompted to select one.
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random seed for reproducibility
    save_dir : str
        Directory to save the trained models and preprocessing objects
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load the dataset
    df = pd.read_csv(data_path)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Display available columns and let user select target if not provided
    if target_column is None:
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns):
            print(f"{i+1}. {col}")
        
        target_idx = int(input("\nEnter the number of the column you want to predict: ")) - 1
        target_column = df.columns[target_idx]
    
    print(f"\nSelected target column: {target_column}")
    
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    # Identify ID columns (usually not useful for prediction)
    id_columns = [col for col in df.columns if 'id' in col.lower()]
    print(f"Identified potential ID columns: {id_columns}")
    
    # Automatically drop ID columns without asking
    if id_columns:
        df = df.drop(columns=id_columns)
        print(f"Dropped ID columns: {id_columns}")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Check if target is categorical or numerical
    unique_values = y.nunique()
    if unique_values <= 10:  # Assuming it's a classification problem if <= 10 unique values
        problem_type = 'classification'
        print(f"\nDetected a classification problem with {unique_values} classes.")
        
        # Encode target if it's not numeric
        if not pd.api.types.is_numeric_dtype(y):
            print("Target is not numeric. Encoding target...")
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            joblib.dump(label_encoder, os.path.join(save_dir, "target_encoder.pkl"))
            print(f"Target encoded. Classes: {label_encoder.classes_}")
        
    else:
        problem_type = 'regression'
        print(f"\nDetected a regression problem with {unique_values} unique values.")
    
    # Identify column types
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nIdentified {len(categorical_cols)} categorical columns and {len(numerical_cols)} numerical columns.")
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Define models based on problem type
    if problem_type == 'classification':
        models = {
            'Random Forest': RandomForestClassifier(random_state=random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
            'Decision Tree': DecisionTreeClassifier(random_state=random_state),
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000)
        }
        
        # Define parameter grids for hyperparameter tuning
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'Decision Tree': {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'Logistic Regression': {
                'C': [0.1, 1.0, 10.0]
            }
        }
    else:  # regression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.linear_model import LinearRegression
        
        models = {
            'Random Forest': RandomForestRegressor(random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
            'Decision Tree': DecisionTreeRegressor(random_state=random_state),
            'Linear Regression': LinearRegression()
        }
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'Decision Tree': {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'Linear Regression': {}
        }
    
    # Train and evaluate models
    results = {}
    best_models = {}
    
    print("\nTraining models with hyperparameter tuning...")
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Perform grid search if parameters exist
        if param_grids[model_name]:
            grid_search = GridSearchCV(
                pipeline, 
                param_grid={f'model__{param}': values for param, values in param_grids[model_name].items()},
                cv=5,
                scoring='accuracy' if problem_type == 'classification' else 'r2',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Evaluate model
        if problem_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            results[model_name] = {
                'accuracy': accuracy * 100,
                'report': report,
                'confusion_matrix': conf_matrix
            }
            
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print(f"Classification Report:\n{report}")
        else:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[model_name] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            }
            
            print(f"R² Score: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
        
        # Save the model
        best_models[model_name] = best_model
        joblib.dump(best_model, os.path.join(save_dir, f"{model_name.replace(' ', '_').lower()}.pkl"))
    
    # Save feature names
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, os.path.join(save_dir, "feature_names.pkl"))
    
    # Save column order for future alignment
    column_order = X.columns.tolist()
    joblib.dump(column_order, os.path.join(save_dir, "column_order.pkl"))
    
    print(f"\nSaved column order: {column_order}")
    
    # Save target column name
    joblib.dump(target_column, os.path.join(save_dir, "target_column.pkl"))
    
    # Save problem type
    joblib.dump(problem_type, os.path.join(save_dir, "problem_type.pkl"))
    
    # Save class names for classification problems
    if problem_type == 'classification':
        # Get unique classes
        classes = sorted(list(set(y_test)))
        joblib.dump(classes, os.path.join(save_dir, "classes.pkl"))
    
    # Determine best model
    if problem_type == 'classification':
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_score = results[best_model_name]['accuracy']
        print(f"\nBest model: {best_model_name} with accuracy {best_score:.2f}%")
    else:
        best_model_name = max(results, key=lambda x: results[x]['r2'])
        best_score = results[best_model_name]['r2']
        print(f"\nBest model: {best_model_name} with R² score {best_score:.4f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    if problem_type == 'classification':
        scores = [results[model]['accuracy'] for model in models]
        plt.bar(models.keys(), scores)
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Models')
        plt.ylabel('Accuracy (%)')
    else:
        scores = [results[model]['r2'] for model in models]
        plt.bar(models.keys(), scores)
        plt.title('Model R² Score Comparison')
        plt.xlabel('Models')
        plt.ylabel('R² Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
    plt.close()
    
    print(f"\nAll models and preprocessing objects saved to '{save_dir}' directory.")
    return best_models, results, problem_type

if __name__ == "__main__":
    # Example usage
    data_path = input("Enter the path to your CSV file: ")
    train_models(data_path)
