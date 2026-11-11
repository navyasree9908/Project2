import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# Folders
UPLOAD_FOLDER = "uploads"
MODELS_FOLDER = "models"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

DATASET_PATH = os.path.join(UPLOAD_FOLDER, "dataset.csv")
GRAPH_PATH = os.path.join(STATIC_FOLDER, "model_comparison.png")
BEST_MODEL_PATH = os.path.join(MODELS_FOLDER, "best_model.pkl")
PREPROCESS_PATH = os.path.join(MODELS_FOLDER, "preprocess.pkl")
METRICS_PATH = os.path.join(MODELS_FOLDER, "metrics.pkl")


# =============== DB INIT ===============

def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            username TEXT UNIQUE,
            password TEXT,
            phone TEXT
        )
    """)
    conn.commit()
    conn.close()


@app.before_first_request
def setup():
    init_db()


def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn


# =============== AUTH ROUTES ===============

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        username = request.form["username"]
        phone = request.form["phone"]
        password = request.form["password"]

        hashed = generate_password_hash(password)

        conn = get_db_connection()
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO users(name,email,username,password,phone) VALUES (?,?,?,?,?)",
                (name, email, username, hashed, phone),
            )
            conn.commit()
            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
        finally:
            conn.close()

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session.clear()
            session["user"] = username
            flash("Logged in successfully.", "success")
            return redirect(url_for("user_home"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    # default: admin / admin123
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "admin":
            session.clear()
            session["admin"] = True
            flash("Admin logged in.", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid admin credentials.", "danger")

    return render_template("admin_login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("index"))


# =============== ADMIN SECTION ===============

@app.route("/admin/dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    dataset_exists = os.path.exists(DATASET_PATH)
    metrics_exists = os.path.exists(METRICS_PATH)

    return render_template(
        "admin_dashboard.html",
        dataset_exists=dataset_exists,
        metrics_exists=metrics_exists,
    )


import pandas as pd
from werkzeug.utils import secure_filename

@app.route("/admin/upload_dataset", methods=["GET", "POST"])
def admin_upload_dataset():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    preview = None

    if request.method == "POST":
        file = request.files.get("dataset")

        if not file or file.filename == "":
            flash("❌ Please select a CSV file.", "danger")
            return redirect(url_for("admin_upload_dataset"))

        filename = secure_filename(file.filename)
        file_path = os.path.join("uploads", filename)
        file.save(file_path)

        # ✅ Read CSV and get first 5 rows
        df = pd.read_csv(file_path)
        preview = df.head().to_html(classes="table table-striped", index=False)

        flash("✅ Dataset uploaded successfully.", "success")

    return render_template("upload.html", preview=preview)

import os
import pandas as pd
from sklearn.model_selection import train_test_split

@app.route("/admin/split_dataset", methods=["GET"])
def admin_split_dataset():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    if not os.path.exists(DATASET_PATH):
        flash("❌ Upload dataset first.", "danger")
        return redirect(url_for("admin_dashboard"))

    # ✅ Read dataset
    df = pd.read_csv(DATASET_PATH)

    total_rows = len(df)

    # ✅ Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_rows = len(train_df)
    test_rows = len(test_df)

    # ✅ Save files
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    train_path = os.path.join(UPLOAD_FOLDER, "train.csv")
    test_path = os.path.join(UPLOAD_FOLDER, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    flash("✅ Dataset successfully split into Train & Test sets.", "success")

    # ✅ Send row counts to HTML
    return render_template(
        "split.html",
        total_rows=total_rows,
        train_rows=train_rows,
        test_rows=test_rows
    )

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

@app.route("/admin/train_models")
def admin_train_models():

    if "admin" not in session:
        return redirect(url_for("admin_login"))

    # ✅ DIRECT CSV FILE PATH (CHANGE THIS TO YOUR FILE)
    csv_path = "bank_transactions_data_1.csv"

    if not os.path.exists(csv_path):
        flash("Dataset file not found!", "danger")
        return redirect(url_for("admin_dashboard"))

    # ======================================================
    # ✅ 1. READ CSV DIRECTLY
    # ======================================================
    df = pd.read_csv(csv_path)

    # ======================================================
    # ✅ 2. DROP UNWANTED COLUMNS
    # ======================================================
    drop_cols = ["TransactionID", "TransactionDate", "IP Address", "DeviceID"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ======================================================
    # ✅ 3. ENCODE CATEGORICAL COLUMNS
    # ======================================================
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_mappings = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        cat_mappings[col] = {cls: int(i) for i, cls in enumerate(le.classes_)}

    # ======================================================
    # ✅ 4. CREATE X AND y
    # ======================================================
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]
    feature_cols = X.columns.tolist()

    # ======================================================
    # ✅ 5. FEATURE SCALING
    # ======================================================
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ======================================================
    # ✅ 6. TRAIN–TEST SPLIT
    # ======================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ======================================================
    # ✅ 7. CLASS WEIGHTS
    # ======================================================
    class_weights = {
        0: 1,
        1: np.sum(y_train == 0) / np.sum(y_train == 1)
    }

    # ======================================================
    # ✅ 8. MODELS
    # ======================================================
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, class_weight=class_weights
        ),

        "Random Forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=300
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=class_weights[1],
            eval_metric="logloss"
        )
    }

    # ======================================================
    # ✅ 9. TRAIN & SELECT BEST
    # ======================================================
    accuracies = {}
    best_model = None
    best_name = ""
    best_acc = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = round(accuracy_score(y_test, preds) * 100, 2)
        accuracies[name] = acc

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    # ======================================================
    # ✅ 10. SAVE BEST MODEL
    # ======================================================
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    # ======================================================
    # ✅ 11. SAVE PREPROCESS INFO
    # ======================================================
    preprocess_info = {
        "drop_cols": drop_cols,
        "cat_cols": cat_cols,
        "feature_cols": feature_cols,
        "cat_mappings": cat_mappings,
        "scaler": scaler,
    }

    with open(PREPROCESS_PATH, "wb") as f:
        pickle.dump(preprocess_info, f)

    # ======================================================
    # ✅ 12. SAVE METRICS
    # ======================================================
    metrics = {
        "accuracies": accuracies,
        "best_model": best_name,
        "best_acc": best_acc,
    }

    with open(METRICS_PATH, "wb") as f:
        pickle.dump(metrics, f)

    # ======================================================
    # ✅ 13. GRAPH GENERATION
    # ======================================================
    plt.figure(figsize=(8, 5))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.ylim(80, 100)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(GRAPH_PATH)
    plt.close()

    flash(f"✅ Models trained successfully. Best = {best_name} ({best_acc:.2f}%)", "success")
    return redirect(url_for("admin_algorithms"))
@app.route("/admin/algorithms")
def admin_algorithms():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    if not os.path.exists(METRICS_PATH):
        flash("Train models first.", "danger")
        return redirect(url_for("admin_dashboard"))

    with open(METRICS_PATH, "rb") as f:
        metrics = pickle.load(f)

    accuracies = metrics["accuracies"]
    best_model = metrics["best_model"]
    best_acc = metrics["best_acc"]

    return render_template(
        "algorithms.html",
        accuracies=accuracies,
        best_model=best_model,
        best_acc=best_acc,
    )

@app.route("/admin/reports")
def admin_reports():

    if "admin" not in session:
        return redirect(url_for("admin_login"))

    if not os.path.exists(METRICS_PATH):
        flash("Train models first.", "danger")
        return redirect(url_for("admin_dashboard"))

    with open(METRICS_PATH, "rb") as f:
        metrics = pickle.load(f)

    accuracies = metrics["accuracies"]
    best_model = metrics["best_model"]
    best_acc = metrics["best_acc"]

    return render_template(
        "admin_reports.html",
        accuracies=accuracies,
        best_model=best_model,
        best_acc=best_acc
    )

# =============== USER SECTION ===============

def load_model_and_preprocess():
    if not os.path.exists(BEST_MODEL_PATH) or not os.path.exists(PREPROCESS_PATH):
        return None, None
    with open(BEST_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(PREPROCESS_PATH, "rb") as f:
        prep = pickle.load(f)
    return model, prep


@app.route("/user/home")
def user_home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("user_home.html")


@app.route("/user/predict", methods=["GET", "POST"])
def user_predict():
    if "user" not in session:
        return redirect(url_for("login"))

    model, prep = load_model_and_preprocess()
    if model is None:
        flash("Admin must train models first.", "danger")
        return redirect(url_for("user_home"))

    feature_cols = prep["feature_cols"]

    if request.method == "POST":
        # build row
        row = {}
        for col in feature_cols:
            val = request.form.get(col)
            if val is None or val == "":
                flash(f"Enter value for {col}", "danger")
                return render_template("user_predict.html", feature_cols=feature_cols)

            if col in prep["cat_cols"]:
                mapping = prep["cat_mappings"][col]
                row[col] = mapping.get(val, 0)
            else:
                row[col] = float(val)

        df_input = pd.DataFrame([row])[feature_cols]
        pred = model.predict(df_input)[0]
        result = "FRAUD TRANSACTION" if int(pred) == 1 else "NORMAL TRANSACTION"
        return render_template("result.html", prediction=result)

    return render_template("user_predict.html", feature_cols=feature_cols)



@app.route("/about")
def about():
    return render_template("about.html")


# =============== RUN ===============

if __name__ == "__main__":
    app.run(debug=True)
