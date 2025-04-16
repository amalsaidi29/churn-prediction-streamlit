import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Charger le fichier CSV
df = pd.read_csv("Data4.csv")

# Extraire la colonne cible
y_train = df["Churn"]

# Créer et entraîner l'encodeur
le = LabelEncoder()
le.fit(y_train)

# Sauvegarder l'encodeur
joblib.dump(le, "model/target_encoder.pkl")

print("✅ Encodeur sauvegardé avec succès dans 'model/target_encoder.pkl'")
