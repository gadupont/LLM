import replicate
import requests
import json
import os
from dotenv import load_dotenv



# === 2. DÉFINITION DU PROMPT STRUCTURÉ POUR LE FORMAT DE VISITE CLINIQUE ===
prompt = (
    "Voici un tableau extrait d’un protocole clinique. À partir de ce tableau, génère un objet JSON structuré avec deux clés : `visits` et `dispensing_units`.\n\n"
    "Pour chaque visite dans le tableau, remplis un objet selon cette structure :\n\n"
    "{\n"
    "  \"id\": <entier unique et croissant, à partir de 1>,\n"
    "  \"code\": <code court de la visite, ex: V1, D1, Screening, etc>,\n"
    "  \"description\": <brève description de la visite, ex: Baseline visit or MRI visit>,\n"
    "  \"order\": <même valeur que id>,\n"
    "  \"is_scheduled\": true ou false selon si la visite est prévue ou non planifiée,\n"
    "  \"anchor\": null si c’est la première visite, sinon l’id de la visite précédente sur laquelle celle-ci est calée (ex: une visite à J+1 d’une autre),\n"
    "  \"expected_days\": nombre de jours attendus depuis l’anchor (0 si même jour),\n"
    "  \"window_before\": nombre de jours acceptables avant cette visite,\n"
    "  \"window_after\": nombre de jours acceptables après cette visite\n"
    "}\n\n"
    "Ajoute aussi une clé `dispensing_units`, qui contient une liste des produits administrés pendant l’étude, selon cette structure :\n\n"
    "{\n"
    "  \"code\": <code produit, ex: P03277_0.1>,\n"
    "  \"description\": <texte du dosage ou nom complet du produit, ex: 'P03277 - 0.1 mmol/kg'>\n"
    "}\n\n"
    "Réponds uniquement avec un JSON valide. Ne commente rien, ne résume rien, ne fais aucun texte explicatif. Utilise exactement les noms de champs spécifiés.\n\n"
    "Voici un exemple attendu :\n\n"
    "{\n"
    "  \"visits\": [\n"
    "    {\n"
    "      \"id\": 1,\n"
    "      \"code\": \"V1\",\n"
    "      \"description\": \"Eligibility assessment and consent signature\",\n"
    "      \"order\": 1,\n"
    "      \"is_scheduled\": true,\n"
    "      \"anchor\": null,\n"
    "      \"expected_days\": null,\n"
    "      \"window_before\": null,\n"
    "      \"window_after\": null\n"
    "    },\n"
    "    {\n"
    "      \"id\": 2,\n"
    "      \"code\": \"V2\",\n"
    "      \"description\": \"First MRI visit with contrast agent (P03277 or MultiHance)\",\n"
    "      \"order\": 2,\n"
    "      \"is_scheduled\": true,\n"
    "      \"anchor\": 1,\n"
    "      \"expected_days\": 7,\n"
    "      \"window_before\": 7,\n"
    "      \"window_after\": 0\n"
    "    }\n"
    "  ],\n"
    "  \"dispensing_units\": [\n"
    "    {\n"
    "      \"code\": \"P03277_0.025\",\n"
    "      \"description\": \"P03277 - 0.025 mmol/kg\"\n"
    "    },\n"
    "    {\n"
    "      \"code\": \"MultiHance_0.1\",\n"
    "      \"description\": \"MultiHance - 0.1 mmol/kg\"\n"
    "    }\n"
    "  ]\n"
    "}"
)

# === 3. IMAGE À TRAITER ===
image_url = "https://raw.githubusercontent.com/gadupont/LLM/094bbfb70cb3e9130c0a16bb4d45a57ecffd84b5/example_de_protocole/ex1/schedule.png"



# === 4. APPEL À LLAVA VIA REPLICATE ===
print(" Envoi de la requête à LLaVA...")
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
