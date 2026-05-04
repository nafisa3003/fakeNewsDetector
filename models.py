import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

MODEL_PATHS = {
    "lr": "lr.pkl",
    "dt": "dt.pkl",
    "gb": "gb.pkl",
    "rf": "rf.pkl",
    "vectorizer": "vectorizer.pkl"
}

def train_all(xv_train, xv_test, y_train, y_test, vectorizer):
    # trains and evaluates multiple classification models

    # Logistic Regression training and reporting
    print("Training Logistic Regression...")
    LR = LogisticRegression(max_iter=1000)
    LR.fit(xv_train, y_train)
    pred_lr = LR.predict(xv_test)
    print(f"LR Accuracy: {accuracy_score(y_test, pred_lr):.4f}") 
    print(classification_report(y_test, pred_lr))

    # Decision Tree training and reporting
    print("Training Decision Tree...")
    DT = DecisionTreeClassifier()
    DT.fit(xv_train, y_train)
    pred_dt = DT.predict(xv_test)
    print(f"DT Accuracy: {accuracy_score(y_test, pred_dt):.4f}")
    print(classification_report(y_test, pred_dt))

    # Gradient Boosting training and reporting
    print("Training Gradient Boosting...")
    GB = GradientBoostingClassifier(
        random_state=0,
        n_estimators=100,
        max_depth=3,
        subsample=0.5,
        max_features="sqrt"
    )
    GB.fit(xv_train, y_train)
    pred_gb = GB.predict(xv_test)
    print(f"GB Accuracy: {accuracy_score(y_test, pred_gb):.4f}")
    print(classification_report(y_test, pred_gb))

    # Random Forest training and reporting
    print("Training Random Forest...")
    RF = RandomForestClassifier(random_state=0, n_estimators=100)
    RF.fit(xv_train, y_train)
    pred_rf = RF.predict(xv_test)
    print(f"RF Accuracy: {accuracy_score(y_test, pred_rf):.4f}")
    print(classification_report(y_test, pred_rf))

    # serializes models and vectorizer to disk
    print("\nSaving models and vectorizer...")
    pickle.dump(LR, open(MODEL_PATHS["lr"], "wb"))
    pickle.dump(DT, open(MODEL_PATHS["dt"], "wb"))
    pickle.dump(RF, open(MODEL_PATHS["rf"], "wb"))
    pickle.dump(GB, open(MODEL_PATHS["gb"], "wb"))
    pickle.dump(vectorizer, open(MODEL_PATHS["vectorizer"], "wb"))
    print("All models saved successfully!")

    return LR, DT, GB, RF


def load_models():
    # checks if all model files exist and loads them
    models = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            return None
        models[name] = pickle.load(open(path, "rb"))
    
    return (
        models["lr"],
        models["dt"],
        models["gb"],
        models["rf"],
        models["vectorizer"]
    )