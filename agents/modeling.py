import pandas as pd
import os
import joblib #Used to save/load trained models whihc are standard in sklearn workflows

from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class ModelingAgent:
    """
    Trains a baseline ML model in a leakage-safe way
    and stores the model and metrics.
    """
    def run(self,
            dataset_path: str,
            target_column: str,
            task_type: str) -> Dict:
# target_column and task_type is decided by Planner

        df=pd.read_csv(dataset_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        #-----Train-test split (prevents leakage)---
        X_train, X_test,y_train,y_test= train_test_split(
            X, y, test_size=0.7, random_state=42
        )

        #--------Identify column types-----
        categorical_cols = X.select_dtypes(include=["object"]).columns #Select categorical features which have dtype strings
        numeric_cols = X.select_dtypes(exclude=["object"]).columns #Select numerical features with dtype int, float.

        #--------Preprocessing---------
        preprocessor= ColumnTransformer(
             transformers=[
                 ("cat", OneHotEncoder(handle_unknown="ignore"),categorical_cols),
                 ("num", StandardScaler(), numeric_cols),
             ]    
        )

        #-------Model selection-------
        if task_type=="classification":
            model = LogisticRegression(max_iter=1000)
            model_type = "logistic_regression"
        else:
            raise NotImplementedError(" Only classification supported for now.")
        #------Pipeline---------
        pipeline= Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", model),
            ]
        )

        #-------Train----------
        pipeline.fit(X_train, y_train)

        #---------Evaluate--------
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        metrics={
            "accuracy": accuracy
        }

        #------Save model artifact-----
        model_dir = "artifacts/models"
        os.makedirs(model_dir,exist_ok=True)
        model_path= os.path.join(model_dir,"baseline_model.pkl")
        joblib.dump(pipeline, model_path)

        return{
            "model_type":model_type,
            "model_metrics": metrics,
            "model_artifact_path": model_path
        }
                