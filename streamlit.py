import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Marketplace Recommender", 
    page_icon="🛍️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Perapihan CSS
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 1rem; }
    .stImage { margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. STATE MANAGEMENT
# ==========================================
if 'search_clicked' not in st.session_state:
    st.session_state.search_clicked = False

# ==========================================
# 3. FUNGSI ML & DATA
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    if not os.path.exists('Online Sales Data.csv'):
        st.error("File 'Online Sales Data.csv' tidak ditemukan. Pastikan file berada di folder yang sama dengan script.")
        return pd.DataFrame()
        
    df = pd.read_csv('Online Sales Data.csv')
    df = df.drop_duplicates(subset=['Product Name']).reset_index(drop=True)
    df['Combined_Text'] = df['Product Category'] + " " + df['Product Name']
    return df

@st.cache_resource
def build_features(data):
    if data.empty:
        return None, None, None
        
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['Combined_Text']).toarray()
    
    scaler = MinMaxScaler()
    normalized_price = scaler.fit_transform(data[['Unit Price']])
    
    product_features = np.hstack((tfidf_matrix, normalized_price * 2.0))
    return tfidf, scaler, product_features

df = load_and_preprocess_data()

if not df.empty:
    all_categories = sorted(df['Product Category'].unique().tolist())
    tfidf_model, price_scaler, product_features_matrix = build_features(df)
else:
    all_categories = []

def get_recommendations(selected_categories, min_price, max_price, top_n=5):
    mask = (df['Unit Price'] >= min_price) & (df['Unit Price'] <= max_price)
    filtered_df = df[mask]
    
    if filtered_df.empty:
        return pd.DataFrame()
        
    filtered_indices = filtered_df.index.tolist()
    
    user_text = " ".join(selected_categories)
    user_tfidf = tfidf_model.transform([user_text]).toarray()
    
    target_price = (min_price + max_price) / 2
    user_price_norm = price_scaler.transform([[target_price]])
    
    user_vector = np.hstack((user_tfidf, user_price_norm * 2.0))
    
    sub_product_features = product_features_matrix[filtered_indices]
    sim_scores = cosine_similarity(user_vector, sub_product_features)[0]
    
    sub_sim_indices = np.argsort(sim_scores)[::-1]
    top_indices = sub_sim_indices[:top_n]
    actual_df_indices = [filtered_indices[i] for i in top_indices]
    
    result_df = df.iloc[actual_df_indices].copy()
    result_df['Match Score (%)'] = sim_scores[top_indices] * 100
    return result_df

def get_popularity_recommendations(top_n=5):
    return df.sort_values(by='Units Sold', ascending=False).head(top_n)

# ==========================================
# 4. FUNGSI RENDER UI KARTU
# ==========================================
def render_product_card(row, show_score=False):
    with st.container(border=True):
        st.caption(f"🏷️ {row['Product Category']}")
        
        st.markdown(f"""
            <div style="height: 50px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; margin-bottom: 5px;">
                <h4 style="margin: 0; font-size: 16px; color: #2c3e50;">{row['Product Name']}</h4>
            </div>
        """, unsafe_allow_html=True)
        
        st.metric(label="Harga", value=f"${row['Unit Price']:.2f}")
        st.divider() 
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"📦 Terjual: **{row['Units Sold']}**")
        with col_b:
            if show_score:
                st.write(f"🎯 Match: **{row['Match Score (%)']:.1f}%**")

# ==========================================
# 5. SIDEBAR & MAIN APP
# ==========================================
def render_sidebar():
    with st.sidebar:
        if os.path.exists('logo.jpg'):
            st.image('logo.jpg', use_container_width=True)
        else:
            st.markdown("<h1 style='text-align: center; color: #1E4620;'>🛍️ POROS</h1>", unsafe_allow_html=True)
            
        st.caption("Temukan produk terbaik sesuai preferensi kategori dan budget Anda menggunakan algoritma AI.")
        st.divider()
        
        if df.empty:
            st.warning("Data tidak tersedia untuk filter.")
            return [], 0, 0, 0, ""

        st.write("### ⚙️ Filter Pencarian")
        user_cat = st.multiselect("Pilih Kategori:", options=all_categories, default=[all_categories[0]] if all_categories else None)
        min_b = st.number_input("Harga Minimal ($):", min_value=0.0, value=10.0, step=5.0)
        max_b = st.number_input("Harga Maksimal ($):", min_value=0.0, value=150.0, step=5.0)
        top_n = st.number_input("Jumlah Rekomendasi:", min_value=1, max_value=50, value=8, step=1)
        
        st.divider()
        
        st.write("### 🔀 Pengurutan")
        sort_option = st.selectbox("Urutkan Hasil Berdasarkan:", [
            "Kecocokan Tertinggi (Default)", 
            "Harga Terendah", 
            "Harga Tertinggi", 
            "Paling Laris"
        ])
        
        st.divider()
        if st.button("🔍 Terapkan Filter", type="primary", use_container_width=True):
            st.session_state.search_clicked = True
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.caption("👨‍💻 Developer: Fatih")
        
        return user_cat, min_b, max_b, top_n, sort_option

def main():
    if df.empty:
        st.error("Aplikasi tidak dapat dijalankan tanpa data.")
        return

    user_cat, min_b, max_b, top_n, sort_option = render_sidebar()
    
    if os.path.exists('logo.jpg'):
        st.image('logo.jpg', use_container_width=True)
    else:
        st.warning("File 'logo.jpg' tidak ditemukan di folder script. Menggunakan placeholder teks.")
        st.markdown("""
            <div style="background-color: #ff4b4b; padding: 25px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="color: white; margin-top: 0; margin-bottom: 8px;">🛍️ Dashboard Rekomendasi</h2>
                <p style="color: white; margin: 0; font-size: 16px;">Implementasi Content-Based Filtering & Popularity-Based pada Dataset Penjualan</p>
            </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📦 Total Produk Tersedia", f"{len(df)} Barang")
    with col2:
        st.metric("🏷️ Kategori Tersedia", f"{len(all_categories)} Kategori")
    with col3:
        st.metric("🛒 Total Terjual", f"{df['Units Sold'].sum()} Unit")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["🎯 Rekomendasi Personal", "🔥 Sedang Tren", "📊 Evaluasi Model"])
    
    with tab1:
        if st.session_state.search_clicked:
            if not user_cat:
                st.warning("⚠️ Pilih minimal satu kategori di sidebar.")
            elif min_b > max_b:
                st.error("⚠️ Harga minimal tidak boleh lebih besar dari harga maksimal.")
            else:
                hasil_rek = get_recommendations(user_cat, min_b, max_b, top_n=top_n)
                
                if hasil_rek.empty:
                    st.info("📦 Maaf, tidak ada barang yang sesuai dengan budget dan kategori tersebut.")
                else:
                    if sort_option == "Harga Terendah":
                        hasil_rek = hasil_rek.sort_values(by="Unit Price", ascending=True)
                    elif sort_option == "Harga Tertinggi":
                        hasil_rek = hasil_rek.sort_values(by="Unit Price", ascending=False)
                    elif sort_option == "Paling Laris":
                        hasil_rek = hasil_rek.sort_values(by="Units Sold", ascending=False)

                    cols = st.columns(4) 
                    for index, row in hasil_rek.reset_index(drop=True).iterrows():
                        with cols[index % 4]: 
                            render_product_card(row, show_score=True)
        else:
            st.info("👈 Silakan atur filter di sidebar dan klik 'Terapkan Filter' untuk melihat rekomendasi yang dipersonalisasi.")

    with tab2:
        st.subheader("Paling Banyak Dibeli Minggu Ini")
        hasil_populer = get_popularity_recommendations(top_n=top_n if top_n > 0 else 8)
        
        cols_pop = st.columns(4)
        for index, row in hasil_populer.reset_index().iterrows():
            with cols_pop[index % 4]:
                render_product_card(row, show_score=False)
                
    with tab3:
        st.subheader("Evaluasi Sistem (Precision@K)")
        st.write("Evaluasi menggunakan metrik Precision@K untuk mengukur relevansi produk yang dihasilkan.")
        
        eval_cat = st.selectbox("Skenario Kategori Uji:", all_categories)
        K_eval = top_n if top_n > 0 else 5
        
        if st.button(f"Jalankan Uji Precision@{K_eval}"):
            # Menggunakan min_b dan max_b dari input sidebar agar relevan dengan budget user
            df_eval = get_recommendations([eval_cat], min_b, max_b, top_n=K_eval)
            
            if not df_eval.empty:
                match_count = sum(df_eval['Product Category'] == eval_cat)
                precision_val = match_count / len(df_eval)
                
                st.metric(label=f"Nilai Precision@{len(df_eval)}", value=f"{precision_val * 100:.1f}%")
                st.dataframe(df_eval[['Product Name', 'Product Category', 'Unit Price', 'Match Score (%)']])
            else:
                st.warning("⚠️ Tidak ada produk di kategori ini yang sesuai dengan rentang budget di Sidebar. Coba naikkan 'Harga Maksimal'.")

if __name__ == "__main__":
    main()