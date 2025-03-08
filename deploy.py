import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, r2_score, mean_squared_error
)

# Set page configuration
st.set_page_config(
    page_title="Smart ML Model Trainer & Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stDataFrame {
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Add dark mode toggle in sidebar
st.sidebar.title("Settings")
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)

# Apply dark mode CSS if enabled
if dark_mode:
    st.markdown("""
    <style>
        .stApp {
            background-color: #121212;
            color: #FFFFFF;
        }
        .main-header {
            color: #90CAF9;
        }
        .sub-header {
            color: #64B5F6;
        }
        .info-box {
            background-color: #1E3A5F;
            color: #FFFFFF;
        }
        .success-box {
            background-color: #1B5E20;
            color: #FFFFFF;
        }
        .warning-box {
            background-color: #5D4037;
            color: #FFFFFF;
        }
        .error-box {
            background-color: #B71C1C;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #1976D2;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #333333;
            color: white;
        }
        .stSelectbox>div>div>div {
            background-color: #333333;
            color: white;
        }
        .stSlider>div>div>div {
            background-color: #1976D2;
        }
        div[data-testid="stDecoration"] {
            background-image: linear-gradient(90deg, #1976D2, #0D47A1);
        }
        .stDataFrame {
            color: white;
        }
        .stDataFrame [data-testid="stTable"] {
            background-color: #333333;
        }
    </style>
    """, unsafe_allow_html=True)

# Helper functions
def get_img_as_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML"""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique classes from the data if class_names is None
    if class_names is None:
        class_names = sorted(list(set(np.concatenate([y_true, y_pred]))))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    return fig

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        # Get feature importances from the model
        importances = model.feature_importances_
        
        # Create DataFrame for better visualization
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Limit to top N features
        if len(feature_names) > top_n:
            feature_imp = feature_imp.head(top_n)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis')
        plt.title(f'Top {len(feature_imp)} Feature Importances')
        plt.tight_layout()
        return fig
    return None

def plot_prediction_distribution(y_true, y_pred, problem_type):
    """Plot distribution of predictions vs actual values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if problem_type == 'classification':
        # For classification, plot count of each class
        df = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred
        })
        
        # Get unique classes
        classes = sorted(set(list(y_true) + list(y_pred)))
        
        # Count occurrences of each class
        actual_counts = [sum(y_true == cls) for cls in classes]
        pred_counts = [sum(y_pred == cls) for cls in classes]
        
        # Plot
        x = np.arange(len(classes))
        width = 0.35
        ax.bar(x - width/2, actual_counts, width, label='Actual')
        ax.bar(x + width/2, pred_counts, width, label='Predicted')
        
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Actual vs Predicted Classes')
        ax.legend()
        
    else:  # regression
        # For regression, plot scatter and line
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
    
    plt.tight_layout()
    return fig

def train_new_model(data, target_column):
    """Train a new model using the prediction.py script"""
    from prediction import train_models
    
    # Save the uploaded data temporarily
    temp_csv_path = "temp_upload.csv"
    data.to_csv(temp_csv_path, index=False)
    
    # Train models
    with st.spinner("Training models... This may take a few minutes."):
        best_models, results, problem_type = train_models(
            data_path=temp_csv_path,
            target_column=target_column,
            save_dir='models'
        )
    
    # Remove temporary file
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)
    
    return best_models, results, problem_type

def load_models(models_dir='models'):
    """Load all available models from the models directory"""
    if not os.path.exists(models_dir):
        return None, None, None, None, None, None
    
    try:
        # Load problem type
        problem_type = joblib.load(os.path.join(models_dir, "problem_type.pkl"))
        
        # Load target column
        target_column = joblib.load(os.path.join(models_dir, "target_column.pkl"))
        
        # Load feature names
        feature_names = joblib.load(os.path.join(models_dir, "feature_names.pkl"))
        
        # Load column order if available
        column_order_path = os.path.join(models_dir, "column_order.pkl")
        column_order = joblib.load(column_order_path) if os.path.exists(column_order_path) else feature_names
        
        # Load classes for classification problems
        classes_path = os.path.join(models_dir, "classes.pkl")
        classes = joblib.load(classes_path) if os.path.exists(classes_path) else None
        
        # Load all model files
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and not f in 
                          ["problem_type.pkl", "target_column.pkl", "feature_names.pkl", 
                           "target_encoder.pkl", "column_order.pkl", "classes.pkl"]]
        
        models = {}
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
            models[model_name] = joblib.load(os.path.join(models_dir, model_file))
        
        return models, problem_type, target_column, feature_names, column_order, classes
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

def align_dataset_columns(data, column_order):
    """Align dataset columns with the expected model columns"""
    # Create a copy to avoid modifying the original
    aligned_data = data.copy()
    
    # Check for missing columns
    missing_cols = set(column_order) - set(aligned_data.columns)
    if missing_cols:
        st.warning(f"The following columns are missing and will be filled with zeros: {missing_cols}")
        for col in missing_cols:
            aligned_data[col] = 0
    
    # Check for extra columns
    extra_cols = set(aligned_data.columns) - set(column_order)
    if extra_cols:
        st.info(f"The following columns will not be used for prediction: {extra_cols}")
    
    # Reorder columns to match the expected order
    return aligned_data[column_order]

def main():
    st.markdown("<h1 class='main-header'>Smart ML Model Trainer & Predictor</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    This app allows you to:
    <ul>
        <li>Upload any dataset and predict any column</li>
        <li>Train new models or use pre-trained models</li>
        <li>Evaluate model performance with detailed metrics</li>
        <li>Visualize predictions and model insights</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Options")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Try different encodings to handle various CSV formats
        encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
        data = None
        
        for encoding in encodings:
            try:
                data = pd.read_csv(uploaded_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if data is None:
            st.error("Could not decode the CSV file. Please check the file encoding and try again.")
            st.stop()
        
        # Show data preview
        st.markdown("<h2 class='sub-header'>Data Preview</h2>", unsafe_allow_html=True)
        st.dataframe(data.head())
        
        # Display data info
        st.markdown("<h2 class='sub-header'>Data Information</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Rows:** {data.shape[0]}")
            st.write(f"**Columns:** {data.shape[1]}")
        
        with col2:
            # Count data types
            dtypes = data.dtypes.value_counts()
            st.write("**Column Types:**")
            for dtype, count in dtypes.items():
                st.write(f"- {dtype}: {count}")
        
        # Check for missing values
        missing_values = data.isnull().sum().sum()
        if missing_values > 0:
            st.markdown(f"""
            <div class='warning-box'>
            ⚠️ Your dataset contains {missing_values} missing values. 
            Don't worry, our preprocessing pipeline will handle them automatically.
            </div>
            """, unsafe_allow_html=True)
        
        # Load existing models
        models, problem_type, existing_target, feature_names, column_order, classes = load_models()
        
        # Mode selection
        mode = st.sidebar.radio(
            "Choose Mode",
            ["Predict with Existing Models", "Train New Models"],
            index=0 if models else 1
        )
        
        if mode == "Train New Models":
            st.markdown("<h2 class='sub-header'>Train New Models</h2>", unsafe_allow_html=True)
            
            # Select target column
            target_column = st.selectbox(
                "Select the column you want to predict:",
                options=data.columns.tolist()
            )
            
            # Train button
            if st.button("Train Models"):
                best_models, results, problem_type = train_new_model(data, target_column)
                
                st.markdown("""
                <div class='success-box'>
                ✅ Models trained successfully! You can now use them for prediction.
                </div>
                """, unsafe_allow_html=True)
                
                # Reload models to ensure we have the latest
                models, problem_type, existing_target, feature_names, column_order, classes = load_models()
                
                # Show training results
                st.markdown("<h2 class='sub-header'>Training Results</h2>", unsafe_allow_html=True)
                
                # Display model comparison chart
                if os.path.exists('models/model_comparison.png'):
                    st.image('models/model_comparison.png')
        
        elif mode == "Predict with Existing Models" and models:
            st.markdown("<h2 class='sub-header'>Predict with Existing Models</h2>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='info-box'>
            ℹ️ Using pre-trained models for predicting <strong>{existing_target}</strong>.<br>
            Problem type: <strong>{problem_type.title()}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Select model
            model_name = st.selectbox(
                "Select Model:",
                options=list(models.keys())
            )
            
            # Check if target column exists in the data
            if existing_target in data.columns:
                # Split data for evaluation
                test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
                
                if st.button("Evaluate Model"):
                    with st.spinner("Evaluating model..."):
                        # Split data
                        X = data.drop(columns=[existing_target])
                        y = data[existing_target]
                        
                        # Align columns with the expected model columns
                        X = align_dataset_columns(X, column_order)
                        
                        # Check if we need to encode the target
                        target_encoder_path = os.path.join('models', "target_encoder.pkl")
                        if os.path.exists(target_encoder_path) and not pd.api.types.is_numeric_dtype(y):
                            target_encoder = joblib.load(target_encoder_path)
                            y = target_encoder.transform(y)
                        
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=42
                            )
                            
                            # Make predictions
                            selected_model = models[model_name]
                            y_pred = selected_model.predict(X_test)
                            
                            # Display results
                            st.markdown("<h2 class='sub-header'>Evaluation Results</h2>", unsafe_allow_html=True)
                            
                            if problem_type == 'classification':
                                # Classification metrics
                                accuracy = accuracy_score(y_test, y_pred)
                                precision = precision_score(y_test, y_pred, average='weighted')
                                recall = recall_score(y_test, y_pred, average='weighted')
                                f1 = f1_score(y_test, y_pred, average='weighted')
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Accuracy", f"{accuracy:.2%}")
                                    st.metric("Precision", f"{precision:.2%}")
                                
                                with col2:
                                    st.metric("Recall", f"{recall:.2%}")
                                    st.metric("F1 Score", f"{f1:.2%}")
                                
                                # Classification report
                                st.markdown("<h3>Classification Report</h3>", unsafe_allow_html=True)
                                report = classification_report(y_test, y_pred, output_dict=True)
                                
                                # Convert to DataFrame for better display
                                report_df = pd.DataFrame(report).transpose()
                                # Round the values for better display
                                report_df = report_df.round(2)
                                st.dataframe(report_df)
                                
                                # Confusion matrix
                                st.markdown("<h3>Confusion Matrix</h3>", unsafe_allow_html=True)
                                
                                # Use classes from saved model if available
                                class_names = classes if classes is not None else sorted(list(set(np.concatenate([y_test, y_pred]))))
                                
                                fig = plot_confusion_matrix(y_test, y_pred, class_names)
                                st.pyplot(fig)
                                
                            else:  # regression
                                # Regression metrics
                                r2 = r2_score(y_test, y_pred)
                                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("R² Score", f"{r2:.4f}")
                                
                                with col2:
                                    st.metric("RMSE", f"{rmse:.4f}")
                                
                                # Actual vs Predicted plot
                                st.markdown("<h3>Actual vs Predicted Values</h3>", unsafe_allow_html=True)
                                fig = plot_prediction_distribution(y_test, y_pred, problem_type)
                                st.pyplot(fig)
                            
                            # Feature importance (if available)
                            if hasattr(selected_model, 'named_steps') and hasattr(selected_model.named_steps['model'], 'feature_importances_'):
                                st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
                                
                                # Get feature names after preprocessing
                                if hasattr(selected_model.named_steps['preprocessor'], 'get_feature_names_out'):
                                    processed_features = selected_model.named_steps['preprocessor'].get_feature_names_out()
                                else:
                                    processed_features = [f"feature_{i}" for i in range(selected_model.named_steps['model'].feature_importances_.shape[0])]
                                
                                fig = plot_feature_importance(selected_model.named_steps['model'], processed_features)
                                if fig:
                                    st.pyplot(fig)
                            
                            # Prediction distribution
                            st.markdown("<h3>Prediction Distribution</h3>", unsafe_allow_html=True)
                            fig = plot_prediction_distribution(y_test, y_pred, problem_type)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error evaluating model: {str(e)}")
                            st.info("This error might be due to incompatible data formats. Please ensure your dataset is compatible with the trained model.")
                
            # Make predictions on new data
            st.markdown("<h2 class='sub-header'>Make Predictions</h2>", unsafe_allow_html=True)
            
            if st.button("Generate Predictions"):
                with st.spinner("Generating predictions..."):
                    # Check if target column exists and remove it for prediction
                    if existing_target in data.columns:
                        X = data.drop(columns=[existing_target])
                    else:
                        X = data.copy()
                    
                    # Align columns with the expected model columns
                    X = align_dataset_columns(X, column_order)
                    
                    # Make predictions
                    try:
                        selected_model = models[model_name]
                        predictions = selected_model.predict(X)
                        
                        # Add predictions to the original dataframe
                        result_df = data.copy()
                        result_df['Predicted'] = predictions
                        
                        # Display results
                        st.dataframe(result_df)
                        
                        # Download button for predictions
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Visualize predictions
                        if existing_target in data.columns:
                            st.markdown("<h3>Prediction Analysis</h3>", unsafe_allow_html=True)
                            
                            if problem_type == 'classification':
                                # Show distribution of predicted classes
                                fig, ax = plt.subplots(figsize=(10, 6))
                                pd.Series(predictions).value_counts().plot(kind='bar', ax=ax)
                                plt.title('Distribution of Predicted Classes')
                                plt.xlabel('Class')
                                plt.ylabel('Count')
                                st.pyplot(fig)
                            else:
                                # Show histogram of predicted values
                                fig, ax = plt.subplots(figsize=(10, 6))
                                plt.hist(predictions, bins=20)
                                plt.title('Distribution of Predicted Values')
                                plt.xlabel('Predicted Value')
                                plt.ylabel('Frequency')
                                st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error making predictions: {str(e)}")
                        st.info("This error might be due to incompatible data formats. Please ensure your dataset is compatible with the trained model.")
        
        else:
            st.markdown("""
            <div class='error-box'>
            ⚠️ No pre-trained models found. Please train new models first.
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # No file uploaded yet
        st.markdown("""
        <div class='info-box'>
        👈 Please upload a CSV file using the sidebar to get started.
        </div>
        """, unsafe_allow_html=True)
        
        # Show example of what the app can do
        st.markdown("<h2 class='sub-header'>What This App Can Do</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; border-radius: 0.5rem; background-color: #E3F2FD;">
                <h3>Work with Any Dataset</h3>
                <p>Upload any CSV file and the app will automatically analyze and process it.</p>
                <p>✅ Automatic preprocessing</p>
                <p>✅ Handles missing values</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; border-radius: 0.5rem; background-color: #E8F5E9;">
                <h3>Train Multiple Models</h3>
                <p>Train and compare multiple machine learning models with hyperparameter tuning.</p>
                <p>✅ Classification & regression</p>
                <p>✅ Optimized performance</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; border-radius: 0.5rem; background-color: #FFF8E1;">
                <h3>Interactive Visualizations</h3>
                <p>Explore your data and model performance with interactive visualizations.</p>
                <p>✅ Feature importance</p>
                <p>✅ Prediction analysis</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
