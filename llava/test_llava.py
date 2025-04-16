import replicate
import requests
import json
import os
from dotenv import load_dotenv



# === 1. FONCTION POUR UPLOADER L’IMAGE LOCALE SUR FILE.IO ===
def upload_image_to_fileio(image_path):
    with open(image_path, "rb") as file:
        response = requests.post("https://file.io", files={"file": file})
        
        try:
            data = response.json()
        except Exception as e:
            print("❌ Erreur de décodage JSON :", e)
            print("🔍 Contenu brut de la réponse :", response.text)
            raise

        if response.status_code == 200 and data.get("success"):
            return data["link"]
        else:
            print("❌ Upload échoué. Détails :", data)
            raise Exception("Upload échoué.")


# === 2. DÉFINITION DU PROMPT STRUCTURÉ POUR LE FORMAT DE VISITE CLINIQUE ===
prompt = (
    "Voici un tableau extrait d’un protocole clinique. Extrait et convertis les données en un objet JSON "
    "structuré représentant les visites de l'étude. Chaque entrée du tableau doit être convertie dans ce format :\n\n"
    "{\n"
    "  \"id\": <numéro entier unique et incrémenté à partir de 1>,\n"
    "  \"code\": <nom de la visite comme D1, D8, Screening, etc>,\n"
    "  \"description\": <texte court résumant la visite avec le jour et sa tolérance>,\n"
    "  \"order\": <même valeur que id>,\n"
    "  \"is_scheduled\": true si elle est planifiée, false si non,\n"
    "  \"anchor\": null si c'est la première visite (Screening), sinon l'id de la visite de référence,\n"
    "  \"expected_days\": <nombre de jours depuis la visite Screening>,\n"
    "  \"window_before\": <jours avant>,\n"
    "  \"window_after\": <jours après>\n"
    "}\n\n"
    "Toutes les visites doivent être listées dans une seule clé 'visits'.\n"
    "Réponds uniquement avec un objet JSON valide, sans aucun texte explicatif ou commentaire.\n"
)

# === 3. IMAGE À TRAITER ===
image_path = "./example_de_protocole/ex1/schedule.png"

# Upload de l’image localement
print("📤 Upload de l’image en cours...")
image_url = upload_image_to_fileio(image_path)
print(f"✅ Image uploadée : {image_url}")

# === 4. APPEL À LLAVA VIA REPLICATE ===
print("🤖 Envoi de la requête à LLaVA...")
output = replicate.run(
    "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb",
    input={
        "image": image_url,
        "prompt": prompt,
        "temperature": 0.2,
        "top_p": 1,
        "max_tokens": 2000
    }
)

# === 5. RÉCUPÉRATION & SAUVEGARDE DU RÉSULTAT ===
full_output = ""
for item in output:
    full_output += str(item)

output_path = "./llava/output.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    f.write(full_output)

print(f"✅ Résultat sauvegardé dans : {output_path}")
