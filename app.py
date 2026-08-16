import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#Add Application Title
# Page Config
st.set_page_config(
    page_title="Breast Cancer ML Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

.metric-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)


# Colorful Header
st.markdown(
    """
    <div style="
        background: linear-gradient(
            90deg,
            #6A1B9A 0%,
            #1565C0 50%,
            #00897B 100%
        );
        padding: 24px;
        border-radius: 14px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.20);
    ">
        <h1 style="margin: 0; color: white;">
            🔬 Breast Cancer Classification Dashboard
        </h1>
        <p style="margin: 8px 0 0 0; font-size: 18px;">
            BITS ML Assignment-2 | Multiple Classification Model Comparison
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


#Add Model Selection Dropdown
st.sidebar.header("⚙️ Dashboard Controls")

st.sidebar.markdown(
    "Select a model and upload the test dataset."
)

selected_model = st.sidebar.selectbox(
    "Select Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest"
    ]
)

st.sidebar.success(
    f"Selected Model: {selected_model}"
)


#Add CSV Upload Option
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data CSV",
    type=["csv"],
    help="Upload the test_data.csv file containing all 30 dataset features."
)

if uploaded_file is None:
    st.sidebar.info(
        "Please upload test_data.csv to generate predictions."
    )
else:
    st.sidebar.success(
        "CSV file uploaded successfully."
    )

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")

    st.dataframe(test_df.head())

#Load the Selected Model
model = None

if selected_model == "Logistic Regression":
    model = joblib.load("model/logistic_regression.pkl")

elif selected_model == "Decision Tree":
    model = joblib.load("model/decision_tree.pkl")

elif selected_model == "kNN":
    model = joblib.load("model/knn_model.pkl")

elif selected_model == "Naive Bayes":
    model = joblib.load("model/naive_bayes.pkl")

elif selected_model == "Random Forest":
    model = joblib.load("model/random_forest.pkl")


#Generate Predictions Using the Uploaded CSV
if uploaded_file is not None and model is not None:

    predictions = model.predict(test_df)

    result_df = test_df.copy()

    result_df["Prediction"] = predictions
    prediction_counts = (
    result_df["Prediction"]
    .value_counts()
    .reset_index()
     )

    prediction_counts.columns = [
        "Prediction",
        "Count"
    ]
    
    prediction_counts["Prediction"] = (
        prediction_counts["Prediction"]
        .map({
            0: "Malignant",
            1: "Benign"
        })
    )


    st.subheader("📈 Prediction Distribution")

    pie_fig = px.pie(
        prediction_counts,
        names="Prediction",
        values="Count",
        color="Prediction",
        color_discrete_map={
            "Malignant": "#ef4444",
            "Benign": "#10b981"
        }
    )
    
    st.plotly_chart(
        pie_fig,
        use_container_width=True
    )
    st.subheader("Prediction Results")

    st.dataframe(result_df.head())


#Display Evaluation Metrics in Streamlit

# Load Model Comparison Metrics

metrics_df = pd.read_csv("model_comparison_results.csv")
comparison_df = metrics_df.sort_values(
    by="Accuracy",
    ascending=False
)

st.subheader("📋 Complete Model Comparison Table")

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)

st.subheader("🌡️ Model Performance Heatmap")

heatmap_df = metrics_df.set_index(
    "ML Model Name"
)

fig, ax = plt.subplots(figsize=(10, 5))

sns.heatmap(
    heatmap_df,
    annot=True,
    cmap="YlGnBu",
    fmt=".4f",
    linewidths=1,
    linecolor="white",
    cbar=True,
    ax=ax
)

ax.set_title(
    "Machine Learning Model Performance Comparison"
)

plt.tight_layout()

st.pyplot(fig)







#Display Classification Report in Streamlit
# Create Classification Report

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


report_predictions = model.predict(X_test)

report = classification_report(
    y_test,
    report_predictions
)

st.subheader("Classification Report")

report_dict = classification_report(
    y_test,
    report_predictions,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()

st.dataframe(report_df)


# Confusion Matrix

cm = confusion_matrix(
    y_test,
    report_predictions
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Malignant", "Actual Benign"],
    columns=["Predicted Malignant", "Predicted Benign"]
)

st.subheader("Confusion Matrix")
st.dataframe(cm_df)

st.subheader("Confusion Matrix Heatmap")

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    linewidths=1,
    cbar=True,
    ax=ax
)

ax.set_title("Confusion Matrix")
plt.tight_layout()

st.pyplot(fig)

st.info(
    "Rows = Actual Values | Columns = Predicted Values"
)