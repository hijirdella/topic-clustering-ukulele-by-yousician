import streamlit as st
import pandas as pd
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

# === Load Model dan Vectorizer ===
kmeans_model = joblib.load("UkulelebyYousician_clustering.pkl")
tfidf_vectorizer = joblib.load("UkulelebyYousician_tfidf_vectorizer.pkl")

st.set_page_config(page_title="Topic Clustering - Ukulele by Yousician", layout="wide")
st.title("Topic Clustering - Ukulele by Yousician")

# === Fungsi Pembersihan Review ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_cluster(texts):
    X = tfidf_vectorizer.transform(texts)
    clusters = kmeans_model.predict(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
    return clusters, X_pca

# === MODE 1: INPUT MANUAL ===
mode = st.radio("Pilih metode input:", ["📝 Input Manual", "📁 Upload CSV"])

if mode == "📝 Input Manual":
    name = st.text_input("Nama Pengguna:")
    star_rating = st.selectbox("Rating Bintang:", [1, 2, 3, 4, 5])
    date = st.date_input("Tanggal Ulasan:")
    review = st.text_area("Tulis Review di sini:")

    if st.button("Prediksi Cluster"):
        if review.strip() == "":
            st.warning("Review tidak boleh kosong.")
        else:
            cleaned = clean_text(review)
            cluster, pca_result = predict_cluster([cleaned])
            df_result = pd.DataFrame({
                "Name": [name],
                "Star Rating": [star_rating],
                "Date": [date],
                "Review": [review],
                "Cluster": cluster,
                "PCA 1": pca_result[:, 0],
                "PCA 2": pca_result[:, 1]
            })
            st.dataframe(df_result)

            # Visualisasi
            st.subheader("Visualisasi PCA")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_result, x='PCA 1', y='PCA 2', hue='Cluster', palette='Set2', s=100, ax=ax)
            st.pyplot(fig)

# === MODE 2: UPLOAD CSV ===
else:
    file = st.file_uploader("Upload file CSV dengan kolom: name, star_rating, date, review", type="csv")
    if file:
        df = pd.read_csv(file)

        required_cols = {'name', 'star_rating', 'date', 'review'}
        if not required_cols.issubset(df.columns):
            st.error(f"File harus memiliki kolom: {', '.join(required_cols)}")
        else:
            df['cleaned_review'] = df['review'].fillna("").apply(clean_text)
            clusters, pca_result = predict_cluster(df['cleaned_review'])
            df['Cluster'] = clusters
            df['PCA 1'] = pca_result[:, 0]
            df['PCA 2'] = pca_result[:, 1]

            st.dataframe(df[['name', 'star_rating', 'date', 'review', 'Cluster']])

            st.subheader("Visualisasi PCA")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='PCA 1', y='PCA 2', hue='Cluster', palette='Set2', s=70, ax=ax)
            st.pyplot(fig)

            # Unduh hasil
            st.download_button(
                label="📥 Unduh Hasil Klaster",
                data=df.to_csv(index=False),
                file_name="hasil_klaster_yousician.csv",
                mime="text/csv"
            )
