import streamlit as st
import pandas as pd
import numpy as np
import os
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Student Depression Dashboard", layout="wide")

# ---------------------------------------------------------------
# Palette (Brunswick / Hunter / Fern / Shamrock / Pistachio green)
# ---------------------------------------------------------------
DARKEST = "#344E41"   # Brunswick green
DARK = "#3A5A40"      # Hunter green
MID = "#588157"       # Fern green
LIGHT = "#5A9F68"      # Shamrock green
LIGHTEST = "#BBD58E"  # Pistachio

st.markdown(f"""
<style>
/* Overrides Streamlit's default red accent (slider handle/track, radio
   dot, checkbox) with the theme's light green everywhere it's used. */
:root {{
    --primary-color: {LIGHT};
}}

/* Sidebar widget labels (Gender, Department, Age Range, Stress Level
   Range) default to black text -- force them white so they read on the
   dark green sidebar background. */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stSelectbox {{
    color: white !important;
}}

/* Slider track/handle + numeric min-max labels in the sidebar */
section[data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] {{
    background-color: {LIGHT} !important;
    border-color: {LIGHT} !important;
}}
section[data-testid="stSidebar"] div[data-testid="stTickBar"],
section[data-testid="stSidebar"] div[data-baseweb="slider"] {{
    color: {LIGHTEST} !important;
}}
.title {{
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: {DARKEST};
}}
.card {{
    background: white;
    padding: 28px 32px;
    border-radius: 16px;
    border-left: 6px solid {MID};
    box-shadow: 0 4px 18px rgba(52, 78, 65, 0.12);
    margin-bottom: 20px;
}}
.card p {{
    font-size: 19px;
    line-height: 1.7;
    color: {DARKEST};
}}
.card b {{
    color: {MID};
}}

/* Sidebar background gradient */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {DARKEST} 0%, {DARK} 100%);
}}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: white !important;
}}
/* Selected radio dot */
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] div:first-child {{
    border-color: {LIGHTEST} !important;
}}

/* Metric cards */
div[data-testid="stMetric"] {{
    background-color: {LIGHTEST}33;
    border: 1px solid {LIGHTEST};
    border-radius: 12px;
    padding: 10px;
}}
div[data-testid="stMetricValue"] {{
    color: {DARKEST};
}}

/* Divider color tweak */
hr {{
    border-color: {LIGHTEST} !important;
}}

/* Info box */
div[data-testid="stAlert"] {{
    background-color: {LIGHTEST}33;
    border-left: 6px solid {MID};
}}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    path = kagglehub.dataset_download("aldinwhyudii/student-depression-and-lifestyle-100k-data")
    csv_file = os.path.join(path, "student_lifestyle_100k.csv")
    return pd.read_csv(csv_file)


df = load_data()

st.sidebar.title("Sections")
section = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA", "Cleaning & Preprocessing", "Model Results"]
)

st.sidebar.divider()
st.sidebar.title("Filters")

gender_options = ["All"] + sorted(df["Gender"].unique().tolist())
selected_gender = st.sidebar.selectbox("Gender", gender_options)

dept_options = ["All"] + sorted(df["Department"].unique().tolist())
selected_dept = st.sidebar.selectbox("Department", dept_options)

age_min = int(df["Age"].min())
age_max = int(df["Age"].max())
selected_age = st.sidebar.slider(
    "Age Range", age_min, age_max, (age_min, age_max)
)

stress_min = int(df["Stress_Level"].min())
stress_max = int(df["Stress_Level"].max())
selected_stress = st.sidebar.slider(
    "Stress Level Range", stress_min, stress_max, (stress_min, stress_max)
)

if selected_gender != "All":
    df = df[df["Gender"] == selected_gender]

if selected_dept != "All":
    df = df[df["Department"] == selected_dept]

df = df[
    (df["Age"] >= selected_age[0]) &
    (df["Age"] <= selected_age[1])
]

df = df[
    (df["Stress_Level"] >= selected_stress[0]) &
    (df["Stress_Level"] <= selected_stress[1])
]

st.markdown("<div class='title'>Student Depression Dashboard</div>", unsafe_allow_html=True)
st.divider()


if section == "Overview":
    st.subheader("Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.write("### First 5 Rows")

    def color_rows(row):
        color = f"{LIGHTEST}55" if row.name % 2 == 0 else f"{MID}22"
        return [f"background-color: {color}; color: {DARKEST}"] * len(row)

    styled_df = df.head().style.apply(color_rows, axis=1)
    st.dataframe(styled_df)

    st.markdown(f"""
    <div class='card'>
    <p>
    This project studies <b>student lifestyle data</b> to understand factors related to depression.
    The goal is to explore how <b>sleep</b>, <b>study hours</b>, <b>stress level</b>, <b>CGPA</b>, <b>gender</b>,
    and <b>department</b> may relate to depression and which patterns show up most strongly
    when comparing students with and without depression.
    </p>
    </div>
    """, unsafe_allow_html=True)


elif section == "EDA":
    st.subheader("EDA Section")

    chart_choice = st.radio(
        "Choose visualization",
        ["Students Sleep", "CGPA", "Stress Level", "Heatmap", "Stress Sleep Relation",
         "All Distributions", "Stress Sleep deprestion"],
        horizontal=True
    )

    bins = st.slider("Change chart bins", 5, 60, 20)

    def smooth_density_chart(data, title, xlabel, color, bins):
        fig, ax = plt.subplots(figsize=(4.5, 2.3))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        # Raised the smoothing floor (2.5 instead of 1.5) so narrow-range,
        # near-discrete columns like Stress Level don't render as separate
        # spikes even at the lowest slider setting.
        bw = np.interp(bins, [5, 60], [2.5, 0.4])
        sns.kdeplot(data, fill=True, color=color, alpha=0.55, linewidth=1.5, ax=ax, bw_adjust=bw, clip=(data.min(), data.max()))

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(DARKEST)

        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel(xlabel, color=DARKEST)
        ax.set_title(title, fontsize=13, fontweight="bold", color=DARKEST)
        ax.tick_params(colors=DARK)

        # use_container_width=False keeps the figure at its true 6x3 aspect
        # instead of Streamlit stretching it to the full column width.
        st.pyplot(fig, transparent=True, use_container_width=False)

    if chart_choice == "Students Sleep":
        smooth_density_chart(df["Sleep_Duration"], "Distribution of Sleep Duration", "Sleep Duration", MID, bins)

    elif chart_choice == "CGPA":
        smooth_density_chart(df["CGPA"], "Distribution of CGPA", "CGPA", LIGHT, bins)

    elif chart_choice == "Stress Level":
        smooth_density_chart(df["Stress_Level"], "Distribution of Stress Level", "Stress Level", DARK, bins)

    elif chart_choice == "Heatmap":
        df_heatmap = df.copy()
        df_heatmap["Depression"] = df_heatmap["Depression"].astype(int)
        numeric_df = df_heatmap.select_dtypes(include=["number"]).drop(columns=["Student_ID"])

        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="Greens", ax=ax, annot_kws={"size": 7})
        ax.set_title("Relationships Between Numerical Features", color=DARKEST)
        st.pyplot(fig, use_container_width=False)

    elif chart_choice == "All Distributions":
        # One figure, three panels -- Sleep / CGPA / Stress side by side
        # so the shapes can be compared directly.
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.6))
        fig.patch.set_alpha(0)
        bw = np.interp(bins, [5, 60], [2.5, 0.4])
        panels = [
            ("Sleep_Duration", "Sleep Duration", MID),
            ("CGPA", "CGPA", LIGHT),
            ("Stress_Level", "Stress Level", DARK),
        ]
        for ax, (col, label, color) in zip(axes, panels):
            ax.set_facecolor("none")
            sns.kdeplot(df[col], fill=True, color=color, alpha=0.55, linewidth=1.5,
                        ax=ax, bw_adjust=bw, clip=(df[col].min(), df[col].max()))
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color(DARKEST)
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.set_xlabel(label, color=DARKEST)
            ax.set_title(label, fontsize=11, fontweight="bold", color=DARKEST)
            ax.tick_params(colors=DARK, labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, transparent=True, use_container_width=False)

    elif chart_choice == "Stress Sleep deprestion":
        # Jittered points instead of a smoothed curve -- shows every
        # student individually, colored by depression status.
        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        sns.stripplot(
            data=df, x="Stress_Level", y="Sleep_Duration", hue="Depression",
            palette=[MID, DARKEST], alpha=0.5, jitter=0.3, size=3, ax=ax
        )

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(DARKEST)
        ax.set_xlabel("Stress Level", color=DARKEST)
        ax.set_ylabel("Sleep Duration", color=DARKEST)
        ax.set_title("Stress Level vs Sleep Duration (Jittered)", fontsize=12, fontweight="bold", color=DARKEST)
        ax.tick_params(colors=DARK)
        legend = ax.legend(title="Depression", labels=["No", "Yes"], frameon=False)
        legend.get_title().set_color(DARKEST)
        for text in legend.get_texts():
            text.set_color(DARKEST)

        st.pyplot(fig, transparent=True, use_container_width=False)

    elif chart_choice == "Stress Sleep Relation":
        temp = df.copy()
        temp["Sleep_Deficit"] = (8 - temp["Sleep_Duration"]).clip(lower=0)
        temp["stress_sleep_relation"] = temp["Stress_Level"] * temp["Sleep_Deficit"]

        temp["bin"] = pd.cut(temp["stress_sleep_relation"], bins=bins)

        chart_df = temp.groupby("bin", observed=True).agg(
            stress_sleep_relation=("stress_sleep_relation", "mean"),
            stress_level=("Stress_Level", "mean"),
            depression_rate=("Depression", "mean")
        ).reset_index()

        chart_df["depression_scaled"] = chart_df["depression_rate"] * 10

        fig, ax = plt.subplots(figsize=(7, 3.2))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        x = chart_df["stress_sleep_relation"]

        ax.fill_between(x, chart_df["stress_level"], color=MID, alpha=0.55, linewidth=0, label="Stress Level")
        ax.plot(x, chart_df["stress_level"], color=MID, linewidth=1.5)

        ax.fill_between(x, chart_df["depression_scaled"], color=LIGHTEST, alpha=0.75, linewidth=0, label="Depression Rate x10")
        ax.plot(x, chart_df["depression_scaled"], color=DARK, linewidth=1.5)

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(DARKEST)

        ax.set_yticks([])
        ax.set_xlabel("Stress Sleep Relation", color=DARKEST)
        ax.set_title("Stress Sleep Relation vs Stress Level and Depression", fontsize=13, fontweight="bold", color=DARKEST)
        ax.legend(frameon=False, loc="upper left")
        ax.tick_params(colors=DARK)

        st.pyplot(fig, transparent=True, use_container_width=False)


elif section == "Cleaning & Preprocessing":
    st.subheader("Cleaning & Preprocessing")

    col1, col2 = st.columns(2)
    col1.metric("Missing Values", df.isnull().sum().sum())
    col2.metric("Duplicate Rows", df.duplicated().sum())

    st.write("### Encoding")

    st.code("""
Gender:
Female → 0
Male   → 1

Depression:
No Depression → 0
Depression    → 1

Department:
One-hot encoding using pd.get_dummies()
""")

    st.write("### Feature Engineering")

    st.code("""
Sleep_Deficit = 8 - Sleep_Duration

stress_sleep_relation = Stress_Level * Sleep_Deficit

Study_sleep_ratio = Study_Hours / Sleep_Duration
""")


elif section == "Model Results":
    st.subheader("Model Results")

    data = df.copy()
    data = data.drop(columns=["Student_ID"])

    data["Sleep_Deficit"] = (8 - data["Sleep_Duration"]).clip(lower=0)
    data["stress_sleep_relation"] = data["Stress_Level"] * data["Sleep_Deficit"]
    data["Study_sleep_ratio"] = data["Study_Hours"] / data["Sleep_Duration"]
    data["Gender"] = data["Gender"].map({"Female": 0, "Male": 1})
    data["Depression"] = data["Depression"].astype(int)
    data = pd.get_dummies(data, columns=["Department"], dtype=int)

    X = data.drop(columns=["Depression"])
    y = data["Depression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model_choice = st.radio(
        "Choose model",
        ["Logistic Regression", "Random Forest", "Logistic Regression with PCA"],
        horizontal=True
    )

    threshold = st.slider("Change decision threshold", 0.1, 0.9, 0.5)

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        model.fit(X_train_scaled, y_train)
        prob = model.predict_proba(X_test_scaled)[:, 1]

    elif model_choice == "Random Forest":
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
        model.fit(X_train_scaled, y_train)
        prob = model.predict_proba(X_test_scaled)[:, 1]

    else:
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        model.fit(X_train_pca, y_train)
        prob = model.predict_proba(X_test_pca)[:, 1]

    pred = (prob >= threshold).astype(int)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", round(accuracy_score(y_test, pred), 3))
    col2.metric("Precision", round(precision_score(y_test, pred), 3))
    col3.metric("Recall", round(recall_score(y_test, pred), 3))
    col4.metric("F1 Score", round(f1_score(y_test, pred), 3))

    # values_format="d" -> whole-number counts instead of long decimals
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        cm,
        display_labels=["No Depression", "Depression"]
    ).plot(ax=ax, values_format="d", cmap="Greens")
    ax.set_title(f"{model_choice} Confusion Matrix", color=DARKEST)
    st.pyplot(fig)

    st.info("""
    Lower threshold usually increases recall, meaning the model catches more depressed students.
    Higher threshold usually increases precision, but may miss more depressed students.
    """)