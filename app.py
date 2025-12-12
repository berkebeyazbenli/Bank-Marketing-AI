import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. AYARLAR VE MODEL YÜKLEME
# ==========================================
st.set_page_config(
    page_title="Bank Deposit AI",
    page_icon="🏦",
    layout="wide"
)

# Başlık
st.title("🏦 Banka Vadeli Mevduat Tahmin Sistemi")
st.markdown("Bu sistem, yapay zeka (SMOTE + MLP) kullanarak müşterilerin **kampanya teklifini kabul edip etmeyeceğini** tahmin eder.")
st.markdown("---")

# Modeli Yükle (Yeni kaydettiğimiz isimle!)
try:
    model = joblib.load('model_smote.pkl')
    st.sidebar.success("✅ Model Başarıyla Yüklendi!")
except FileNotFoundError:
    st.error("🚨 HATA: 'model_smote.pkl' bulunamadı. Lütfen model dosyasının aynı klasörde olduğundan emin olun.")
    st.stop()

# ==========================================
# 2. SIDEBAR - KULLANICI VERİ GİRİŞİ
# ==========================================
st.sidebar.header("📝 Müşteri Bilgileri")

def user_input_features():
    # --- Demografik ---
    st.sidebar.subheader("1. Demografik Bilgiler")
    age = st.sidebar.slider("Yaş", 18, 90, 30)
    job = st.sidebar.selectbox("Meslek", 
        ['admin.', 'blue-collar', 'technician', 'services', 'management', 
         'retired', 'entrepreneur', 'self-employed', 'housemaid', 'unemployed', 'student', 'unknown'])
    marital = st.sidebar.selectbox("Medeni Durum", ['married', 'single', 'divorced', 'unknown'])
    education = st.sidebar.selectbox("Eğitim Seviyesi", 
        ['university.degree', 'high.school', 'basic.9y', 'professional.course', 
         'basic.4y', 'basic.6y', 'unknown', 'illiterate'])
    
    # --- Finansal ---
    st.sidebar.subheader("2. Finansal Durum")
    default = st.sidebar.selectbox("Kredi Temerrüdü Var mı?", ['no', 'unknown', 'yes'])
    housing = st.sidebar.selectbox("Konut Kredisi Var mı?", ['yes', 'no', 'unknown'])
    loan = st.sidebar.selectbox("Bireysel Kredi Var mı?", ['no', 'yes', 'unknown'])

    # --- İletişim ---
    st.sidebar.subheader("3. İletişim Detayları")
    contact = st.sidebar.selectbox("İletişim Türü", ['cellular', 'telephone'])
    month = st.sidebar.selectbox("Son İletişim Ayı", ['may', 'jul', 'aug', 'jun', 'nov', 'apr', 'oct', 'sep', 'mar', 'dec'])
    day_of_week = st.sidebar.selectbox("Son İletişim Günü", ['mon', 'thu', 'wed', 'tue', 'fri'])
    duration = st.sidebar.number_input("Son Görüşme Süresi (Saniye)", 0, 5000, 200)
    campaign = st.sidebar.number_input("Mevcut Kampanya İçin Görüşme Sayısı", 1, 50, 1)

    # --- Geçmiş Veriler ---
    st.sidebar.subheader("4. Geçmiş Kampanyalar")
    pdays = st.sidebar.number_input("Önceki Kampanyadan Beri Geçen Gün (999: Aranmadı)", 0, 999, 999)
    previous = st.sidebar.number_input("Bu Kampanya Öncesi Görüşme Sayısı", 0, 10, 0)
    poutcome = st.sidebar.selectbox("Önceki Kampanya Sonucu", ['nonexistent', 'failure', 'success'])

    # --- Ekonomik Göstergeler ---
    st.sidebar.subheader("5. Ekonomik Göstergeler")
    emp_var_rate = st.sidebar.number_input("İstihdam Değişim Oranı", -4.0, 2.0, -1.8)
    cons_price_idx = st.sidebar.number_input("Tüketici Fiyat Endeksi", 90.0, 95.0, 92.8)
    cons_conf_idx = st.sidebar.number_input("Tüketici Güven Endeksi", -55.0, -25.0, -46.2)
    euribor3m = st.sidebar.number_input("Euribor 3 Ay Oranı", 0.0, 6.0, 1.2)
    nr_employed = st.sidebar.number_input("Çalışan Sayısı", 4900.0, 5300.0, 5099.1)

    # Verileri DataFrame'e dönüştür
    data = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan,
        'contact': contact, 'month': month, 'day_of_week': day_of_week,
        'duration': duration, 'campaign': campaign, 'pdays': pdays,
        'previous': previous, 'poutcome': poutcome,
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# ==========================================
# 3. ANA EKRAN VE TAHMİN
# ==========================================

# Kullanıcı verisini göster
st.subheader("📋 Müşteri Profili")
st.dataframe(input_df)

if st.button('🚀 Analiz Et ve Tahminle'):
    
    # --- KRİTİK: Feature Engineering ---
    # Model eğitimi sırasında yaptığımız manuel işlemleri burada da yapmalıyız!
    
    # 1. 'previously_contacted' özelliğini türet
    input_df['previously_contacted'] = np.where(input_df['pdays'] != 999, 1, 0)
    
    # 2. 'unknown' string'lerini NaN yap (Pipeline'daki Imputer doldursun diye)
    input_df = input_df.replace(['unknown', 'nonexistent'], np.nan)

    # --- TAHMİN ---
    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1] # "Evet" (1) olma ihtimali

        st.markdown("---")
        
       # --- GELİŞMİŞ GÖRSELLEŞTİRME ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tahmin Sonucu")
            
            # DURUM 1: Güçlü EVET (> %60)
            if prediction_proba > 0.60:
                st.success("✅ **EVET (Mevduat Hesabı Açacak)**")
                st.balloons()
                
            # DURUM 2: GRİ ALAN / KARARSIZ (%40 - %60 Arası)
            elif prediction_proba > 0.40:
                st.warning("⚠️ **DİKKAT: Müşteri Kararsız (Sınırda)**")
                st.markdown("*Bu müşteri 'Hayır' diyebilir ancak doğru teklifle ikna edilmeye çok yakın.*")
                
            # DURUM 3: Net HAYIR (< %40)
            else:
                st.error("❌ **HAYIR (Teklifi Reddedecek)**")
        
        with col2:
            st.subheader("Güven Skoru")
            st.info(f"Olasılık: **%{prediction_proba*100:.2f}**")
            st.progress(prediction_proba)
            
            # Detaylı Yorumlama
            if prediction_proba > 0.75:
                st.write("💡 **Yorum:** Çok güçlü bir potansiyel müşteri. Kaçırmayın!")
            elif prediction_proba > 0.60:
                st.write("💡 **Yorum:** Olumlu görünüyor, standart prosedürü uygulayın.")
            elif prediction_proba > 0.40:
                st.write("💡 **Yorum:** **KRİTİK BÖLGE!** Kampanya detayları veya faiz avantajı ile ikna edilebilir.")
            else:
                st.write("💡 **Yorum:** Düşük ihtimal, kaynak harcamaya değmeyebilir.")
                
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")