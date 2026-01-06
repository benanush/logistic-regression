import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction using Logistic Regression")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("https://github.com/benanush/logistic-regression/blob/main/framingham_heart_disease.csv")

try:
    df = load_data()
    st.success("Dataset loaded successfully!")
except:
    st.error("Dataset not found! Place framingham_heart_disease.csv in the same folder.")
    st.stop()

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("📊 Dataset Preview")
st.write(df.head())

# -----------------------------
# Handle Missing Values
# -----------------------------
df = df.dropna()

# -----------------------------
# Target Distribution
# -----------------------------
st.subheader("📈 Heart Disease Distribution")

target_counts = df['TenYearCHD'].value_counts()

fig1 = plt.figure()
plt.bar(['No Heart Disease', 'Heart Disease'], target_counts.values)
plt.xlabel("Condition")
plt.ylabel("Count")
plt.title("10-Year Coronary Heart Disease Distribution")
st.pyplot(fig1)

# -----------------------------
# Feature & Target Split
# -----------------------------
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("✅ Model Accuracy")
st.write(f"Accuracy: **{accuracy:.2f}**")

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig2 = plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["No CHD", "CHD"])
plt.yticks([0, 1], ["No CHD", "CHD"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
st.pyplot(fig2)

# -----------------------------
# ROC Curve
# -----------------------------
st.subheader("📈 ROC Curve")

y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig3 = plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
st.pyplot(fig3)

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("📊 Feature Importance")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

fig4 = plt.figure(figsize=(10, 6))
plt.barh(coefficients['Feature'], coefficients['Coefficient'])
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Feature Importance (Logistic Regression)")
plt.gca().invert_yaxis()
st.pyplot(fig4)

# -----------------------------
# User Prediction
# -----------------------------
st.subheader("🔮 Predict Heart Disease Risk")

user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

