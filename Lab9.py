#Task 1: Setting Up the Environment
%pip install mlflow
%pip install scikit-learn
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Task 2: Load and Prepare the Data
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

#Task 3: Set Up an MLflow Experiment
import mlflow
# Set up an MLflow experiment
mlflow.set_experiment("/Shared/MLflow_Lab_Student_Experiment")
print("Experiment set to: /Shared/MLflow_Lab_Student_Experiment")

#Task 4: Define, Train, and Log the Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
# Train and log the model with MLflow
with mlflow.start_run():
    # Define the model
    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_train, y_train)  # Train the model on training data
    # Make predictions and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    # Log parameters
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 3)
    # Log metric
    mlflow.log_metric("accuracy", accuracy)
    # Define an input example and model signature
    input_example = X_test[:5]  # Example input data
    signature = infer_signature(X_train, model.predict(X_train))
    # Log the model
    mlflow.sklearn.log_model(
        model,
        "random_forest_model",
        input_example=input_example,
        signature=signature,
        pip_requirements=["scikit-learn==1.0.2", "cloudpickle==3.1.0"]
    )
    print(f"Model Accuracy: {accuracy}")

#Task 6: Experiment with Different Model Parameters
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
with mlflow.start_run():
    # Define the model with new parameters
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)  # Train the model on training data
    # Make predictions and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 4)
    # Log metric
    mlflow.log_metric("accuracy", accuracy)
    # Define an input example and model signature
    input_example = X_test[:5]  # Example input data
    signature = infer_signature(X_train, model.predict(X_train))
    # Log the model
    mlflow.sklearn.log_model(
        model,
        "random_forest_model",
        input_example=input_example,
        signature=signature,
        pip_requirements=["scikit-learn==1.0.2", "cloudpickle==3.1.0"]
    )
    print(f"Model Accuracy with updated parameters: {accuracy}")

#Task 7: Load and Use the Logged Model
from sklearn.metrics import accuracy_score
import mlflow.sklearn
# Replace with your actual Run ID
run_id = "0a4c01bcdcc8477284c0ef2c48583638"  # Example Run ID
model_uri = f"runs:/{run_id}/random_forest_model"
# Load the saved model
loaded_model = mlflow.sklearn.load_model(model_uri)
# Use the loaded model to make predictions
loaded_predictions = loaded_model.predict(X_test)
# Calculate and print the accuracy of the loaded model
loaded_accuracy = accuracy_score(y_test, loaded_predictions)
print(f"Loaded model accuracy: {loaded_accuracy}")

