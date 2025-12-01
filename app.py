# app.py
# Final Streamlit MVP – CMPT 3835 (Machine Learning Modeling & Optimization)
# Beaverhill Bird Observatory – Saw-whet Owl Migration Project

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# BASIC CONFIG
# =========================================================
st.set_page_config(
    page_title="Saw-whet Owl Migration Dashboard",
    layout="wide"
)

DATA_PATH = "data/night_level_bbo.csv"
MODEL_PATH = "models/night_type_model.pkl"


# =========================================================
# LOAD DATA & MODEL (CACHED)
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Overview",
        "Visual Insights",
        "ML Model & SHAP",
        "RAG Chatbot"
    ]
)


# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================
if page == "Overview":
    st.title("Saw-whet Owl Migration – Dashboard Overview")

    st.markdown("""
        This dashboard analyzes **16-sheet Motus detection data** to understand
        Saw-whet Owl migration timing, behaviour types, station activity patterns,
        and ML predictions for **fly-by, intermediate, and linger nights**.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Nights", df["date"].nunique())
    with col2:
        st.metric("Unique Owl Tags", df["motusTagID"].nunique())
    with col3:
        st.metric("Night Types", ", ".join(df["night_type"].unique()))

    st.subheader("Night-level dataset preview")
    st.dataframe(df.head(20))

    st.subheader("Night type counts")
    st.bar_chart(df["night_type"].value_counts())


# =========================================================
# PAGE 2 — VISUAL INSIGHTS
# =========================================================
elif page == "Visual Insights":
    st.title("Visual Insights – Migration Patterns")

    # ------------------- Plot 1 -------------------
    st.subheader("Total detections per day (all tags)")
    daily = df.groupby("date")["n_hits"].sum().reset_index()

    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(daily["date"], daily["n_hits"], marker="o")
    ax1.set_title("Total detections per day")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total hits")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # ------------------- Plot 2 -------------------
    st.subheader("Nightly detections per tag (top 20 tags)")
    top_tags = df.groupby("motusTagID")["n_hits"].sum().nlargest(20).index
    sub = df[df["motusTagID"].isin(top_tags)]
    heat = sub.pivot_table(index="motusTagID", columns="date", values="n_hits",
                           aggfunc="sum", fill_value=0)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    im = ax2.imshow(heat.values, aspect="auto", origin="lower")
    ax2.set_yticks(range(len(heat.index)))
    ax2.set_yticklabels(heat.index)
    ax2.set_title("Nightly detections per tag")
    fig2.colorbar(im, ax=ax2)
    st.pyplot(fig2)

    # ------------------- Plot 3 -------------------
    st.subheader("Migration wave: hits & active tags per night")
    hits = df.groupby("date")["n_hits"].sum()
    tags = df.groupby("date")["motusTagID"].nunique()

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(hits.index, hits.values, label="Total hits")
    ax3.set_ylabel("Total hits")
    ax4 = ax3.twinx()
    ax4.plot(tags.index, tags.values, color="orange", label="Active tags")
    ax4.set_ylabel("Active tags")
    ax3.legend(loc="upper left")
    ax4.legend(loc="upper right")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # ------------------- Plot 4 -------------------
    st.subheader("Hourly detection patterns for top 5 tags")
    top5 = df.groupby("motusTagID")["n_hits"].sum().nlargest(5).index
    sub2 = df[df["motusTagID"].isin(top5)]
    hourly = sub2.groupby(["motusTagID", "hour"])["n_hits"].sum().reset_index()

    fig4, ax4 = plt.subplots(figsize=(12, 5))
    for tag in top5:
        temp = hourly[hourly["motusTagID"] == tag]
        ax4.plot(temp["hour"], temp["n_hits"], marker="o", label=str(tag))
    ax4.set_xlabel("Hour of day")
    ax4.set_ylabel("Hits")
    ax4.legend()
    st.pyplot(fig4)

    # ------------------- Plot 5 -------------------
    st.subheader("Night types: fly-by vs intermediate vs linger")
    fig5, ax5 = plt.subplots(figsize=(7, 5))
    for nt in df["night_type"].unique():
        temp = df[df["night_type"] == nt]
        ax5.scatter(temp["duration_hours"], temp["n_hits"], label=nt, alpha=0.8)
    ax5.set_xlabel("Duration (hours)")
    ax5.set_ylabel("Hits")
    ax5.legend()
    st.pyplot(fig5)

    # ------------------- Plot 6 -------------------
    st.subheader("Detections by antenna port")
    ports = df.groupby("port")["n_hits"].sum()
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    ax6.bar(ports.index.astype(str), ports.values)
    ax6.set_xlabel("Port")
    ax6.set_ylabel("Hits")
    st.pyplot(fig6)

    # ------------------- Plot 7 -------------------
    st.subheader("Detection date ranges for top 10 tags")
    top10 = df.groupby("motusTagID")["n_hits"].sum().nlargest(10).index
    ranges = df.groupby("motusTagID")["date"].agg(["min", "max"]).loc[top10]

    fig7, ax7 = plt.subplots(figsize=(12, 4))
    for i, (tag, row) in enumerate(ranges.iterrows()):
        ax7.plot([row["min"], row["max"]], [i, i], marker="o")
    ax7.set_yticks(range(len(ranges)))
    ax7.set_yticklabels(ranges.index)
    plt.xticks(rotation=45)
    st.pyplot(fig7)

    # ------------------- Plot 8 -------------------
    st.subheader("Nightly activity with anomalies highlighted")
    nightly = df.groupby("date")["n_hits"].sum().reset_index()
    threshold = nightly["n_hits"].mean() + 2 * nightly["n_hits"].std()
    nightly["is_anomaly"] = nightly["n_hits"] > threshold

    fig8, ax8 = plt.subplots(figsize=(12, 4))
    ax8.scatter(nightly["date"],
                nightly["n_hits"],
                c=np.where(nightly["is_anomaly"], "red", "grey"))
    plt.xticks(rotation=45)
    st.pyplot(fig8)

    # ------------------- Plot 9 -------------------
    st.subheader("Approximate probability of detection vs hour")
    prob = df.groupby("hour")["n_hits"].sum()
    prob = prob / prob.sum()

    fig9, ax9 = plt.subplots(figsize=(10, 4))
    ax9.plot(prob.index, prob.values, marker="o")
    ax9.set_xlabel("Hour")
    ax9.set_ylabel("Probability")
    st.pyplot(fig9)


# =========================================================
# PAGE 3 — ML MODEL & SHAP
# =========================================================
elif page == "ML Model & SHAP":

    st.title("Machine Learning Model – Night Type Classification")

    feature_cols = ["n_hours", "n_hits", "duration_hours"]
    X = df[feature_cols]
    y = df["night_type"]

    st.subheader("Model Classification Report")
    preds = model.predict(X)
    st.json(classification_report(y, preds, output_dict=True))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, preds, labels=y.unique())

    fig_cm, ax_cm = plt.subplots()
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks(range(len(y.unique())))
    ax_cm.set_yticks(range(len(y.unique())))
    ax_cm.set_xticklabels(y.unique())
    ax_cm.set_yticklabels(y.unique())
    fig_cm.colorbar(im, ax=ax_cm)
    st.pyplot(fig_cm)

    st.subheader("SHAP Summary Plot (Feature Importance)")

    sample = X.sample(min(200, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    fig_shap = plt.figure(figsize=(8, 4))
    shap.summary_plot(shap_values, sample, feature_names=feature_cols, show=False)
    st.pyplot(fig_shap)

    # --- Prediction playground ---
    st.subheader("Try the model")

    n_hours_in = st.slider("n_hours", 0.0, 24.0, 3.0, 0.5)
    n_hits_in = st.slider("n_hits", 0, 8000, 500, 50)
    dur_in = st.slider("duration_hours", 0.0, 24.0, 3.0, 0.5)

    sample_input = pd.DataFrame([[n_hours_in, n_hits_in, dur_in]], columns=feature_cols)
    pred_class = model.predict(sample_input)[0]

    st.markdown(f"### Predicted night type: **{pred_class}**")

    shap_single = explainer.shap_values(sample_input)
    fig_bar = plt.figure(figsize=(6, 3))
    shap.summary_plot(shap_single, sample_input, feature_names=feature_cols,
                      plot_type="bar", show=False)
    st.pyplot(fig_bar)


# =========================================================
# PAGE 4 — RAG CHATBOT (LOCAL, NO OPENAI)
# =========================================================
elif page == "RAG Chatbot":

    st.title("Ask the Owls – Local RAG Chatbot")

    st.markdown("""
        This chatbot uses a local TF-IDF search engine over all nights & tags.
        No internet, no API keys — **your data only**.
    """)

    # -------- Build text corpus --------
    @st.cache_resource
    def build_corpus(df):
        corpus = []
        for _, r in df.iterrows():
            text = f"""
            On {r['date'].date()},
            tag {r['motusTagID']} had a {r['night_type']} night
            with {r['n_hits']} detections
            lasting {r['duration_hours']} hours
            mainly on port {r['port']}.
            """
            corpus.append(text)

        tfidf = TfidfVectorizer()
        matrix = tfidf.fit_transform(corpus)
        return corpus, tfidf, matrix

    corpus, tfidf, matrix = build_corpus(df)

    # -------- Chat interface --------
    user_q = st.text_input("Ask something (e.g., When were linger nights common?)")

    if user_q:
        q_vec = tfidf.transform([user_q])
        sims = cosine_similarity(q_vec, matrix)[0]
        top_idx = sims.argsort()[::-1][:5]

        st.subheader("Relevant Nights")
        st.write(df.iloc[top_idx][["date", "motusTagID", "night_type",
                                   "n_hits", "duration_hours", "port"]])

        subset = df.iloc[top_idx]
        common = ", ".join(subset["night_type"].value_counts().index)
        avg_hits = int(subset["n_hits"].mean())

        st.subheader("Chatbot Answer")
        st.write(
            f"The most relevant nights show mainly **{common}** behaviour, "
            f"with an average of **{avg_hits} detections**. "
            f"These nights occurred between **{subset['date'].min().date()}** "
            f"and **{subset['date'].max().date()}**."
        )
