import base64
import io

import joblib
import numpy as np
import pandas as pd
import regex
import requests
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps
from pydantic import BaseModel
from rapidocr_onnxruntime import RapidOCR

#Configurações
CAMINHO_MODELO = "models/modelo_xgboost.joblib"
MAPA_CLASSES = {0: "Invalidos", 1: "RG_Antigo", 2: "RG_novo"};
ERROS_FUZZY = 2     
LADO_MAXIMO = 1600  

# Modelo
modelo = joblib.load(CAMINHO_MODELO)
COLUNAS = list(modelo.feature_names_in_)  # as 26 palavras, na ordem do modelo
ocr = RapidOCR()

#API
app = FastAPI()



class Entrada(BaseModel):
    fonte: str


# Ler a imagem (URL ou base64)
def carregar_imagem(fonte):
    fonte = fonte.strip()

    if fonte.startswith("http://") or fonte.startswith("https://"):
        resposta = requests.get(fonte, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resposta.raise_for_status()
        dados = resposta.content
    else:
        if fonte.startswith("data:"):
            fonte = fonte.split(",", 1)[1]  # remove "data:image/png;base64,"
        fonte = regex.sub(r"\s+", "", fonte)
        fonte += "=" * (-len(fonte) % 4)    # corrige padding que às vezes some
        dados = base64.b64decode(fonte)

    imagem = ImageOps.exif_transpose(Image.open(io.BytesIO(dados))).convert("RGB")
    imagem.thumbnail((LADO_MAXIMO, LADO_MAXIMO))
    return np.array(imagem)



def ler_texto(imagem_rgb):
    bgr = np.ascontiguousarray(imagem_rgb[:, :, ::-1])
    resultado, _ = ocr(bgr)
    if not resultado:
        return ""
    return "\n".join(linha[1] for linha in resultado)

def tem_palavra(texto, palavra):
    termo = r"\s+".join(regex.escape(parte) for parte in palavra.split())
    padrao = rf"(?e)(?<!\w)(?:{termo}){{e<={ERROS_FUZZY}}}(?!\w)"
    return regex.search(padrao, texto, flags=regex.IGNORECASE) is not None


def nome_da_classe(classe):
    return MAPA_CLASSES.get(int(classe), int(classe))


# Rota da API
@app.post("/prever")
def prever(entrada: Entrada):
    try:
        imagem = carregar_imagem(entrada.fonte)
    except Exception as erro:
        # 400 = o erro foi na requisição do cliente
        raise HTTPException(status_code=400, detail=f"Não consegui ler a imagem: {erro}")

    try:
        texto = ler_texto(imagem)
        encontradas = {palavra: int(tem_palavra(texto, palavra)) for palavra in COLUNAS}
        features = pd.DataFrame([encontradas], columns=COLUNAS)
        probabilidades = modelo.predict_proba(features)[0]
    except Exception as erro:
        # 500 = o erro foi do servidor (OCR, modelo...)
        raise HTTPException(status_code=500, detail=f"Erro ao classificar a imagem: {erro}")

    melhor = int(np.argmax(probabilidades))

    return {
        "predicao": nome_da_classe(modelo.classes_[melhor]),
        "confianca": round(float(probabilidades[melhor]), 4),
    }