import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from ydata_profiling import ProfileReport
import joblib

# ✅ Load data
df = pd.read_csv("Expresso_churn_dataset.csv")  
# ✅ Drop duplicates and missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# ✅ Encode categorical features
label_enc = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    if col != 'user_id':
        df[col] = label_enc.fit_transform(df[col])

# ✅ Define target and features
target = 'CHURN' if 'CHURN' in df.columns else 'MRG'
X = df.drop(columns=['user_id', target])
y = df[target]

# 🖨️ Debug: Check number of features
print("Number of features used in training:", X.shape[1])
print("Feature names:", X.columns.tolist())

# ✅ Scale numeric values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# ✅ Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ✅ Save model and scaler
joblib.dump(model, "expresso_model.pkl")
joblib.dump(scaler, "expresso_scaler.pkl")

# ✅ Generate profiling report
profile = ProfileReport(df, title="Expresso Churn Data Report", explorative=True)
profile.to_file("expresso_churn_report.html")
