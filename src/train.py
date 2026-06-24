import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Import our preprocessing step
from preprocess import get_preprocessing_pipeline

def train_model():
    print("🔄 Step 1: Loading data...")
    # Load the baseline data you just generated
    df = pd.read_csv("data/baseline_training.csv")
    X = df.drop(columns=['default'])
    y = df['default']

    # Split into 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("📊 Step 2: Setting up MLflow tracking...")
    # This will just create a local folder called 'mlruns' to save results
    mlflow.set_experiment("Simple_Credit_Model")

    with mlflow.start_run():
        print("🚀 Step 3: Building and training the model...")
        
        # Combine preprocessing and the Random Forest into one pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', get_preprocessing_pipeline()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # Train it!
        pipeline.fit(X_train, y_train)

        print("📈 Step 4: Evaluating the model...")
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"⭐ Accuracy on test data: {acc * 100:.2f}%")
        
        # Save metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        
        # FIX: Tell MLflow to use the standard cloudpickle format to avoid security panics
        mlflow.sklearn.log_model(pipeline, "model", serialization_format="cloudpickle")

        print("💾 Step 5: Saving the final model file...")
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/model_latest.joblib")
        print("✅ Success! Model saved as 'models/model_latest.joblib'")

if __name__ == "__main__":
    train_model()