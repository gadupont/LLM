import replicate
import requests
import json
import os
from dotenv import load_dotenv



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
image_url = "https://raw.githubusercontent.com/gadupont/LLM/094bbfb70cb3e9130c0a16bb4d45a57ecffd84b5/example_de_protocole/ex1/schedule.png"



# === 4. APPEL À LLAVA VIA REPLICATE ===
print("🤖 Envoi de la requête à LLaVA...")
output = replicate.run(
    "yorickvp/llava-v1.6-mistral-7b:19be067b589d0c46689ffa7cc3ff321447a441986a7694c01225973c2eafc874",
    input={
        "image": image_url,
        "prompt": prompt,
        
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
