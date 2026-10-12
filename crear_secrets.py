"""
crear_secrets.py — Convierte la llave JSON de Google en .streamlit/secrets.toml

Uso (desde la carpeta tablero-seguros):
    python crear_secrets.py

Te pide la ruta del archivo .json y el link de tu Google Sheet, y escribe
.streamlit/secrets.toml (que está en .gitignore y NO se sube a GitHub).
"""

import glob
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))

# 1 · Buscar la llave JSON (por defecto, la más reciente en Descargas)
descargas = os.path.join(os.path.expanduser("~"), "Downloads")
candidatos = sorted(glob.glob(os.path.join(descargas, "*.json")), key=os.path.getmtime, reverse=True)
candidatos = [c for c in candidatos if '"client_email"' in open(c, encoding="utf-8", errors="ignore").read()]

sugerida = candidatos[0] if candidatos else ""
ruta = input(f"Ruta del archivo JSON de Google [{sugerida}]: ").strip().strip('"') or sugerida
with open(ruta, encoding="utf-8") as f:
    cred = json.load(f)

sheet_url = input("Pega el link de tu Google Sheet: ").strip().strip('"')

# 2 · Escribir secrets.toml (json.dumps genera strings válidos en TOML, incluido el \n de la llave)
lineas = [f"sheet_url = {json.dumps(sheet_url)}", "", "[gcp_service_account]"]
lineas += [f"{k} = {json.dumps(v)}" for k, v in cred.items()]

os.makedirs(os.path.join(AQUI, ".streamlit"), exist_ok=True)
destino = os.path.join(AQUI, ".streamlit", "secrets.toml")
with open(destino, "w", encoding="utf-8") as f:
    f.write("\n".join(lineas) + "\n")

print(f"\nListo: se creó {destino}")
print(f"Correo del robot (compártele la hoja como Editor): {cred.get('client_email')}")
