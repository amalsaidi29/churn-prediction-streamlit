import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# 🔹 Charger les données
df = pd.read_csv("Data4.csv")  # adapte si ton chemin est différent

# 🔹 Liste des 15 meilleures features (selon toi)
features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract'
]

target = 'Churn'

# 🔹 Encoder les colonnes catégoriques
for col in df[features + [target]].select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# 🔹 Séparer les variables
X = df[features]
y = df[target]

# 🔹 Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Entraîner le modèle
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# 🔹 Créer dossier model/ si besoin
os.makedirs("model", exist_ok=True)

# 🔹 Sauvegarder le modèle
joblib.dump(model, "model/gradient_boosting_model_15_features.pkl")

print("✅ Modèle sauvegardé avec succès.")
