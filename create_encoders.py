import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import streamlit as st

# Charger les données
df = pd.read_csv("Data4.csv")

# Colonnes à encoder (sauf 'SeniorCitizen' qui est déjà numérique)
categorical_columns = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", 
    "InternetService", "OnlineSecurity", "OnlineBackup", "StreamingTV",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

# Dictionnaire pour stocker les encodeurs
encoders = {}

# Vérification des encodeurs manquants
missing_encoders = []
for col in categorical_columns:
    if col in encoders:
        # Si l'encodeur existe déjà, appliquer la transformation
        df[col] = encoders[col].transform(df[col].astype(str))
    else:
        # Si l'encodeur manque, ajouter à la liste
        missing_encoders.append(col)

# Afficher les colonnes avec encodeurs manquants
if missing_encoders:
    for col in missing_encoders:
        st.warning(f"⚠️ L'encodeur pour la colonne '{col}' est manquant.")
    st.stop()  # Pour éviter que le modèle plante

# Entraîner un encodeur par colonne et stocker dans le dictionnaire
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = df[col].astype(str)  # Au cas où la colonne serait mal formatée
    le.fit(df[col])  # Entraîner l'encodeur
    encoders[col] = le  # Ajouter l'encodeur au dictionnaire

# Sauvegarder le dictionnaire complet d'encodeurs
joblib.dump(encoders, "model/encoder.pkl")
st.success("✅ Dictionnaire d'encodeurs sauvegardé dans 'model/encoder.pkl'")
