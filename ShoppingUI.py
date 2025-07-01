import streamlit as st
import pandas as pd
import joblib
import os

# Paths
model_dir = r'C:\Shopper spectrum Project\models'

# Load models and data
kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
scaler = joblib.load(os.path.join(model_dir, 'rfm_scaler.pkl'))
product_similarity = joblib.load(os.path.join(model_dir, 'product_similarity.pkl'))
product_names = joblib.load(os.path.join(model_dir, 'product_names.pkl'))

# --------------- Product Recommendation Logic ---------------

def get_top_similar_products(product_name, top_n=5):
    # Search for product code
    matched = product_names[product_names.str.lower().str.contains(product_name.lower())]
    if matched.empty:
        return None, None
    
    product_code = matched.index[0]
    similar_scores = product_similarity[product_code].sort_values(ascending=False)
    similar_codes = similar_scores.drop(product_code).head(top_n).index
    return product_code, product_names.loc[similar_codes]

# --------------- Customer Segmentation Logic ----------------

def predict_cluster(recency, frequency, monetary):
    rfm_input = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
    rfm_scaled = scaler.transform(rfm_input)
    cluster = kmeans.predict(rfm_scaled)[0]
    
    # Heuristic interpretation
    if recency < 60 and frequency > 5 and monetary > 500:
        label = "High-Value"
    elif frequency > 3 and monetary > 200:
        label = "Regular"
    elif recency > 180 and frequency < 2:
        label = "At-Risk"
    else:
        label = "Occasional"
    
    return cluster, label

# ---------------------- Streamlit UI ------------------------

st.set_page_config(page_title="Shopper Spectrum", layout="centered")

st.title("🛍️ Shopper Spectrum Dashboard")

tab1, tab2 = st.tabs(["📦 Product Recommendations", "👤 Clustering"])

# ----------------- 📦 Product Recommendations -----------------

with tab1:
    st.header("🔍 Find Similar Products")
    product_input = st.text_input("Enter a product name", placeholder="e.g. white mug")

    if st.button("Get Recommendations"):
        if product_input:
            product_code, recommendations = get_top_similar_products(product_input)
            if recommendations is not None:
                st.success(f"Top 5 products similar to **{product_names[product_code]}**:")
                for i, prod in enumerate(recommendations.values, 1):
                    st.markdown(f"**{i}.** {prod}")
            else:
                st.warning("No similar products found. Try another keyword.")
        else:
            st.warning("Please enter a product name.")

# ------------------ 👤 Customer Segmentation -------------------

with tab2:
    st.header("🎯 Predict Customer Segment")
    recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=60)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, max_value=100, value=5)
    monetary = st.number_input("Monetary (total amount spent)", min_value=0.0, value=200.0)

    if st.button("Predict Cluster"):
        cluster, label = predict_cluster(recency, frequency, monetary)
        st.success(f"🧠 Predicted Segment: **{label}** (Cluster {cluster})")