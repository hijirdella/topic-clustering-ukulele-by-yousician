import streamlit as st
import pandas as pd
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load model dan vectorizer
kmeans_model = joblib.load("UkulelebyYousician_clustering.pkl")
tfidf_vectorizer = joblib.load("UkulelebyYousician_tfidf_vectorizer.pkl")

st.set_page_config(page_title="Clustering Review - Ukulele by Yousician", layout="wide")

st.title("Clustering Review - Ukulele by Yousician")

# Input mode
mode = st.radio("Pilih metode input:", ["📝 Input Manual", "📁 Upload CSV"])

# Fungsi untuk prediksi cluster
def predict_cluster(texts):
    X = tfidf_vectorizer.transform(texts)
    cluster_labels = kmeans_model.predict(X)

    # PCA untuk visualisasi
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    return cluster_labels, X_pca

# Mode 1: Manual
if mode == "📝 Input Manual":
    user_input = st.text_area("Masukkan review (1 atau lebih):", height=200)
    if st.button("Prediksi Cluster"):
        texts = [t.strip() for t in user_input.strip().split("\n") if t.strip()]
        if texts:
            clusters, pca_result = predict_cluster(texts)
            df_result = pd.DataFrame({
                "Review": texts,
                "Cluster": clusters,
                "PCA 1": pca_result[:, 0],
                "PCA 2": pca_result[:, 1]
            })
            st.dataframe(df_result)

            # Visualisasi
            st.subheader("Visualisasi PCA")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_result, x='PCA 1', y='PCA 2', hue='Cluster', palette='Set2', s=80, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Masukkan minimal satu review.")

# Mode 2: Upload CSV
else:
    uploaded_file = st.file_uploader("Upload file CSV dengan kolom 'clean_review':", type='csv')
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'clean_review' not in df.columns:
            st.error("Kolom 'clean_review' tidak ditemukan dalam file.")
        else:
            clusters, pca_result = predict_cluster(df['clean_review'].fillna(""))
            df['Cluster'] = clusters
            df['PCA 1'] = pca_result[:, 0]
            df['PCA 2'] = pca_result[:, 1]
            st.dataframe(df[['clean_review', 'Cluster']])

            # Visualisasi
            st.subheader("Visualisasi PCA")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='PCA 1', y='PCA 2', hue='Cluster', palette='Set2', s=70, ax=ax)
            st.pyplot(fig)

            # Unduh hasil
            csv = df.to_csv(index=False)
            st.download_button("📥 Unduh Hasil", csv, "hasil_klaster.csv", "text/csv")
