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
st.title("🏦 Bank Term Deposit Prediction System")
st.markdown("---")

# Modeli Yükle (Yeni kaydettiğimiz isimle!)
try:
    model = joblib.load('bank_marketing_final_model.pkl')
    st.sidebar.success("✅ Model Loaded Successfully!")
except FileNotFoundError:
    st.error("🚨 ERROR: 'bank_marketing_final_model.pkl' not found. Please ensure the model file is in the same folder.")
    st.stop()

# ==========================================
# 2. SIDEBAR - KULLANICI VERİ GİRİŞİ
# ==========================================
st.sidebar.header("📝 Customer Information")

def user_input_features():
    # --- Demografik ---
    st.sidebar.subheader("1. Demographic Information")
    age = st.sidebar.slider("Age", 18, 90, 30)
    job = st.sidebar.selectbox("Job", 
        ['admin.', 'blue-collar', 'technician', 'services', 'management', 
         'retired', 'entrepreneur', 'self-employed', 'housemaid', 'unemployed', 'student', 'unknown'])
    marital = st.sidebar.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
    education = st.sidebar.selectbox("Education Level", 
        ['university.degree', 'high.school', 'basic.9y', 'professional.course', 
         'basic.4y', 'basic.6y', 'unknown', 'illiterate'])
    
    # --- Finansal ---
    st.sidebar.subheader("2. Financial Status")
    default = st.sidebar.selectbox("Has Credit Default?", ['no', 'unknown', 'yes'])
    housing = st.sidebar.selectbox("Has Housing Loan?", ['yes', 'no', 'unknown'])
    loan = st.sidebar.selectbox("Has Personal Loan?", ['no', 'yes', 'unknown'])

    # --- İletişim ---
    st.sidebar.subheader("3. Contact Details")
    contact = st.sidebar.selectbox("Contact Type", ['cellular', 'telephone'])
    month = st.sidebar.selectbox("Last Contact Month", ['may', 'jul', 'aug', 'jun', 'nov', 'apr', 'oct', 'sep', 'mar', 'dec'])
    day_of_week = st.sidebar.selectbox("Last Contact Day", ['mon', 'thu', 'wed', 'tue', 'fri'])
    duration = st.sidebar.number_input("Last Call Duration (Seconds)", 0, 5000, 200)
    campaign = st.sidebar.number_input("Number of Contacts for Current Campaign", 1, 50, 1)

    # --- Geçmiş Veriler ---
    st.sidebar.subheader("4. Previous Campaigns")
    pdays = st.sidebar.number_input("Days Since Last Contact (999: Not Contacted)", 0, 999, 999)
    previous = st.sidebar.number_input("Number of Contacts Before This Campaign", 0, 10, 0)
    poutcome = st.sidebar.selectbox("Previous Campaign Outcome", ['nonexistent', 'failure', 'success'])

    # --- Ekonomik Göstergeler ---
    st.sidebar.subheader("5. Economic Indicators")
    emp_var_rate = st.sidebar.number_input("Employment Variation Rate", -4.0, 2.0, -1.8)
    cons_price_idx = st.sidebar.number_input("Consumer Price Index", 90.0, 95.0, 92.8)
    cons_conf_idx = st.sidebar.number_input("Consumer Confidence Index", -55.0, -25.0, -46.2)
    euribor3m = st.sidebar.number_input("Euribor 3 Month Rate", 0.0, 6.0, 1.2)
    nr_employed = st.sidebar.number_input("Number of Employees", 4900.0, 5300.0, 5099.1)

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
st.subheader("📋 Customer Profile")
st.dataframe(input_df)

if st.button('🚀 Analyze and Predict'):
    
    # --- KRİTİK: Feature Engineering ---
    # Model eğitimi sırasında yaptığımız manuel işlemleri burada da yapmalıyız!
    
    # 1. 'was_contacted' özelliğini türet (pdays != 999 means was contacted before)
    input_df['was_contacted'] = np.where(input_df['pdays'] != 999, 1, 0)
    
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
            st.subheader("Prediction Result")
            
            # DURUM 1: Güçlü EVET (> %60)
            if prediction_proba > 0.60:
                st.success("✅ **YES (Will Open Deposit Account)**")
                st.balloons()
                
            # DURUM 2: GRİ ALAN / KARARSIZ (%40 - %60 Arası)
            elif prediction_proba > 0.40:
                st.warning("⚠️ **WARNING: Customer is Uncertain (Borderline)**")
                st.markdown("*This customer may say 'No' but is very close to being convinced with the right offer.*")
                
            # DURUM 3: Net HAYIR (< %40)
            else:
                st.error("❌ **NO (Will Reject the Offer)**")
        
        with col2:
            st.subheader("Confidence Score")
            st.info(f"Probability: **{prediction_proba*100:.2f}%**")
            st.progress(prediction_proba)
            
            # Detaylı Yorumlama
            if prediction_proba > 0.75:
                st.write("💡 **Comment:** Very strong potential customer. Don't miss out!")
            elif prediction_proba > 0.60:
                st.write("💡 **Comment:** Looks positive, follow standard procedure.")
            elif prediction_proba > 0.40:
                st.write("💡 **Comment:** **CRITICAL ZONE!** Can be convinced with campaign details or interest rate advantage.")
            else:
                st.write("💡 **Comment:** Low probability, may not be worth resource allocation.")
                
    except Exception as e:
        st.error(f"An error occurred: {e}")