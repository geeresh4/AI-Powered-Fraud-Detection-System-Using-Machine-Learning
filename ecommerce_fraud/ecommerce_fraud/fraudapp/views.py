from django.shortcuts import render, redirect
from django.contrib import messages
from .models import User
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# ============================================================
# GLOBALS
# ============================================================
df = None
x_train = x_test = y_train = y_test = None

accuracy = []
precision = []
recall = []
fscore = []

# Fraud labels (index → name)
LABELS = [
    'account_takeover',
    'identity_theft',
    'synthetic_account',
    'promotion_abuse',
    'legit',
    'refund_fraud',
    'payment_fraud'
]

# Model Feature Columns
FEATURE_COLUMNS = [
    'transaction_id',
    'user_id',
    'transaction_method',
    'transaction_amount',
    'location_distance_km',
    'failed_login_attempts',
    'coupon_used',
    'refund_history',
    'device_change',
    'ip_risk_score',
    'behavior_score'
]


# ============================================================
# BASIC PAGES
# ============================================================
def home(request):
    return render(request, 'home.html')


def admin_login(request):
    if request.method == 'POST':
        uname = request.POST['username']
        pwd = request.POST['password']
        if uname == 'admin' and pwd == 'admin':
            return redirect('admin_dashboard')
        messages.error(request, "Invalid credentials")
        return redirect('admin_login')
    return render(request, 'adminlogin.html')


def admin_dashboard(request):
    return render(request, 'admin_dashboard.html')


# ============================================================
# UPLOAD DATASET
# ============================================================
def upload_dataset(request):
    global df

    if request.method == 'POST' and request.FILES.get('dataset'):
        try:
            csv_file = request.FILES['dataset']
            df = pd.read_csv(csv_file)
            messages.success(request, "Dataset uploaded successfully!")
            return redirect('preprocess_data')

        except Exception as e:
            messages.error(request, f"Error reading file: {e}")
            return redirect('upload_dataset')

    return render(request, 'upload.html')


# ============================================================
# PREPROCESS DATA
# ============================================================
def preprocess_data(request):
    global df

    if df is None:
        return render(request, 'preprocess.html',
                      {'result': "Please upload a dataset first."})

    try:
        nulls = df.isnull().sum()
        dups = df.duplicated().sum()

        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)

        # Encode all categorical columns
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col])

        # Count Plot
        plt.figure(figsize=(10, 8))
        sns.set(style="darkgrid")
        target_col = df.columns[-1]
        ax = sns.countplot(x=target_col, data=df)

        for p in ax.patches:
            ax.annotate(f'{p.get_height()}',
                        (p.get_x() + p.get_width()/2., p.get_height()),
                        ha='center', va='center', fontsize=10,
                        xytext=(0, 5), textcoords='offset points')

        plt.tight_layout()
        plt.savefig("static/countplot.png")
        plt.close()

        output = (
            f"Null values before:<br>{nulls}<br><br>"
            f"Duplicate rows before: {dups}<br><br>"
            f"Shape after preprocessing: {df.shape}"
        )
        return render(request, 'preprocess.html', {'result': output})

    except Exception as e:
        return render(request, 'preprocess.html', {'result': str(e)})


# ============================================================
# SPLIT DATA
# ============================================================
def split_data(request):
    global df, x_train, x_test, y_train, y_test

    if df is None:
        return render(request, 'split.html', {
            'result': "Please preprocess the data first."
        })

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            df.iloc[:, :-1],
            df.iloc[:, -1],
            test_size=0.20,
            random_state=42
        )

        return render(request, 'split.html', {
            'result': f"Train: {x_train.shape} | Test: {x_test.shape}"
        })

    except Exception as e:
        return render(request, 'split.html', {'result': str(e)})


# ============================================================
# METRICS
# ============================================================
def calculateMetrics(algorithm, predict, testY):
    import numpy as np

    predict = np.array(predict).astype(int)
    testY = np.array(testY).astype(int)

    labels_int = list(range(len(LABELS)))

    a = accuracy_score(testY, predict) * 100
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    print(f"\n======= {algorithm} =======")
    print(f"Accuracy: {a:.2f}%")
    print(f"Precision: {p:.2f}%")
    print(f"Recall: {r:.2f}%")
    print(f"F1 Score: {f:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(
        testY, predict, labels=labels_int, target_names=LABELS
    ))

    # Confusion Matrix
    cm = confusion_matrix(testY, predict, labels=labels_int)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f"{algorithm} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


# ============================================================
# TRAIN XGBOOST
# ============================================================
def train_xgboost(request):
    global x_train, y_train, x_test, y_test

    if x_train is None:
        return render(request, 'train.html', {
            'error': "Please split the data first."
        })

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "XGBoostClassifier.pkl")

    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            loaded = True
        else:
            model = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42
            )
            model.fit(x_train, y_train)
            joblib.dump(model, model_path)
            loaded = False

        preds = model.predict(x_test)
        calculateMetrics("XGBoost", preds, y_test)

        return render(request, 'train.html', {
            'accuracy': accuracy[-1],
            'precision': precision[-1],
            'recall': recall[-1],
            'fscore': fscore[-1],
            'loaded': loaded
        })

    except Exception as e:
        return render(request, 'train.html', {'error': str(e)})


# ============================================================
# TRAIN STACKING CLASSIFIER
# ============================================================
def train_stacking(request):
    global x_train, y_train, x_test, y_test

    if x_train is None:
        return render(request, 'train.html', {
            'error': "Please split the data first."
        })

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "StackingClassifier.pkl")

    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            loaded = True

        else:
            base_models = [
                ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
                ("dt", DecisionTreeClassifier(random_state=42)),
                ("xgb", XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="mlogloss",
                    random_state=42
                ))
            ]

            meta = LogisticRegression(max_iter=500)

            model = StackingClassifier(
                estimators=base_models,
                final_estimator=meta,
                stack_method="predict_proba",
                n_jobs=-1
            )

            model.fit(x_train, y_train)
            joblib.dump(model, model_path)
            loaded = False

        preds = model.predict(x_test)
        calculateMetrics("Stacking Classifier", preds, y_test)

        return render(request, 'train.html', {
            'accuracy': accuracy[-1],
            'precision': precision[-1],
            'recall': recall[-1],
            'fscore': fscore[-1],
            'loaded': loaded
        })

    except Exception as e:
        return render(request, 'train.html', {'error': str(e)})


# ============================================================
# USER REGISTRATION & LOGIN
# ============================================================
def register(request):
    if request.method == 'POST':
        name = request.POST['name']
        email = request.POST['email']
        phone = request.POST['phone']
        city = request.POST['city']
        state = request.POST['state']
        country = request.POST['country']
        password = request.POST['password']
        confirm = request.POST['confirm_password']

        if password != confirm:
            messages.error(request, "Passwords do not match")
            return redirect('register')

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already exists")
            return redirect('register')

        User(
            name=name, email=email, phone=phone,
            city=city, state=state, country=country,
            password=password
        ).save()

        messages.success(request, "Registration successful")
        return redirect('user_login')

    return render(request, 'register.html')


def user_login(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']

        try:
            user = User.objects.get(email=email, password=password)
            request.session['user_id'] = user.id
            return redirect('user_input')
        except User.DoesNotExist:
            messages.error(request, "Invalid credentials")
            return redirect('user_login')

    return render(request, 'userlogin.html')


# ============================================================
# USER PREDICTION
# ============================================================
def user_input(request):

    if request.method == 'POST':
        try:
            # ---------------------------------------------------
            # 1. Collect Form Inputs
            # ---------------------------------------------------
            values = [request.POST.get(col) for col in FEATURE_COLUMNS]
            input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)

            # ---------------------------------------------------
            # 2. Manual Label Encoding (same as preprocessing)
            # ---------------------------------------------------

            # transaction_method encoding
            tm_map = {
                "UPI": 0,
                "Credit": 1,
                "Debit": 2,
                "Wallet": 3
            }
            input_df["transaction_method"] = input_df["transaction_method"].map(tm_map).fillna(-1)

            # coupon_used encoding
            coupon_map = {"No": 0, "Yes": 1}
            input_df["coupon_used"] = input_df["coupon_used"].map(coupon_map).fillna(-1)

            # device_change encoding
            device_map = {"No": 0, "Yes": 1}
            input_df["device_change"] = input_df["device_change"].map(device_map).fillna(-1)

            # Convert numeric values
            numeric_cols = [
                'transaction_id', 'user_id', 'transaction_amount',
                'location_distance_km', 'failed_login_attempts',
                'refund_history', 'ip_risk_score', 'behavior_score'
            ]
            for col in numeric_cols:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

            # ---------------------------------------------------
            # 3. Model Selection
            # ---------------------------------------------------
            model_choice = request.POST.get("model_choice")
            model_files = {
                'xgboost': "model/XGBoostClassifier.pkl",
                'stacking': "model/StackingClassifier.pkl"
            }

            if model_choice not in model_files:
                return render(request, 'user_input.html', {
                    'error': "Invalid model selection!"
                })

            model_path = model_files[model_choice]

            if not os.path.exists(model_path):
                return render(request, 'user_input.html', {
                    'error': f"{model_choice} model not trained yet!"
                })

            model = joblib.load(model_path)

            # ---------------------------------------------------
            # 4. Align Features
            # ---------------------------------------------------
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

            # ---------------------------------------------------
            # 5. Prediction
            # ---------------------------------------------------
            pred = model.predict(input_df)[0]

            LABEL_MAP = {
                0: "legit",
                1: "payment_fraud",
                2: "promotion_abuse",
                3: "refund_fraud",
                4: "synthetic_account",
                5: "identity_theft",
                6: "account_takeover"
            }

            predicted_label = LABEL_MAP.get(pred, "Unknown")

            return render(request, "user_input.html", {
                "predicted_class": predicted_label,
                "model_used": model_choice.capitalize()
            })

        except Exception as e:
            return render(request, "user_input.html", {
                "error": f"Error: {e}"
            })

    return render(request, "user_input.html")


# ============================================================
# LOGOUT
# ============================================================
def logout(request):
    request.session.flush()
    return redirect('home')
