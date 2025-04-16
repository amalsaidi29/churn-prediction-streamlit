import joblib
import pandas as pd
import streamlit as st
import shap

# 🎨 -- STYLING FONCTION --
def set_bg_and_style():
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(to right, #f9fafe, #e6ecf9);
            padding: 2rem;
            font-family: 'Segoe UI', sans-serif;
        }

        h1, h3, h2 {
            color: #3b3f8f;
            text-align: center;
            font-weight: 700;
        }

        .stSelectbox, .stSlider, .stNumberInput {
            background-color: #ffffff !important;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .stButton>button {
            background: linear-gradient(to right, #4e54c8, #8f94fb);
            color: white;
            border: none;
            padding: 10px 30px;
            border-radius: 30px;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #8f94fb, #4e54c8);
            transform: scale(1.05);
        }

        .prediction-result {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            margin-top: 2rem;
            color: #444;
        }

        .block-container {
            padding-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

# 🔄 -- PRÉTRAITEMENT --
def preprocess_input(input_data, encoders):
    input_data = input_data.copy()
    for col in input_data.columns:
        if col in encoders:
            encoder = encoders[col]
            input_data[col] = encoder.transform([input_data[col][0]])
        elif input_data[col].dtype == 'object':
            input_data[col] = pd.factorize(input_data[col])[0]
    return input_data

# 🚀 -- CHARGEMENTS --
model = joblib.load("model/gradient_boosting_model_15_features.pkl")
encoders = joblib.load("model/encoder.pkl")
explainer = shap.TreeExplainer(model)

# 🌟 -- UI --
set_bg_and_style()
st.image("https://img.icons8.com/clouds/500/artificial-intelligence.png", width=120)
st.title("📡 Prédiction de Résiliation Client Télécom")
st.markdown("Remplissez les informations du client pour savoir s’il risque de **résilier son abonnement**.")

# 🧾 -- FORMULAIRE --
with st.form("formulaire_churn"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("👤 Sexe", ["Male", "Female"])
        SeniorCitizen = st.selectbox("🎓 Client senior", ["Yes", "No"])
        Partner = st.selectbox("💍 A un(e) partenaire", ["Yes", "No"])
        Dependents = st.selectbox("👶 A des personnes à charge", ["Yes", "No"])
        Tenure = st.slider("📅 Ancienneté (mois)", 0, 72, 12)
        PhoneService = st.selectbox("📞 Service téléphonique", ["Yes", "No"])
        MultipleLines = st.selectbox("📱 Lignes multiples", ["Yes", "No"])
        InternetService = st.selectbox("🌐 Connexion Internet", ["DSL", "Fiber optic", "No"])

    with col2:
        OnlineSecurity = st.selectbox("🔒 Sécurité en ligne", ["Yes", "No"])
        OnlineBackup = st.selectbox("💾 Sauvegarde en ligne", ["Yes", "No"])
        StreamingTV = st.selectbox("📺 Streaming TV", ["Yes", "No"])
        Contract = st.selectbox("📝 Type de contrat", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("📨 Facturation sans papier", ["Yes", "No"])
        PaymentMethod = st.selectbox("💳 Méthode de paiement", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        MonthlyCharges = st.number_input("💸 Facturation mensuelle (€)", 0.0, 500.0, 50.0)

    submitted = st.form_submit_button("🔍 Prédire le risque")

# 🧠 -- PRÉDICTION --
if submitted:
    raw_input = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "Tenure": Tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "StreamingTV": StreamingTV,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges
    }])

    input_encoded = preprocess_input(raw_input, encoders)
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.markdown(f"""
        <div class='prediction-result'>
            <h2 style='color:#d90429;'>⚠️ Risque élevé !</h2>
            <p>Ce client a <strong>{proba * 100:.1f}%</strong> de chance de résilier son abonnement.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='prediction-result'>
            <h2 style='color:#2b9348;'>✅ Client fidèle</h2>
            <p>Ce client a seulement <strong>{(1 - proba) * 100:.1f}%</strong> de risque de résiliation.</p>
        </div>
        """, unsafe_allow_html=True)

    # SHAP
    shap_values = explainer.shap_values(input_encoded)
    st.subheader("🔍 Pourquoi cette prédiction ?")
    st.pyplot(shap.summary_plot(shap_values, input_encoded, plot_type="bar", show=False))
