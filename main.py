"""
SERVIDOR: recebe uma URL ou uma imagem em base64 de um documento
e devolve a classe prevista pelo modelo.

Para rodar:
    pip install -r requirements.txt
    pip install --no-deps rapidocr-onnxruntime==1.4.4
    uvicorn main:app --reload

Depois abra http://127.0.0.1:8000/docs no navegador.
"""

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

# --- Configurações ---------------------------------------------------
CAMINHO_MODELO = "models/modelo_xgboost.joblib"
MAPA_CLASSES = {0: "Invalidos", 1: "RG_Antigo", 2: "RG_novo"};
ERROS_FUZZY = 2     
LADO_MAXIMO = 1600  

# Carrega uma vez, ao iniciar o servidor
modelo = joblib.load(CAMINHO_MODELO)
COLUNAS = list(modelo.feature_names_in_)  # as 26 palavras, na ordem do modelo
ocr = RapidOCR()

# 1) Cria a aplicacao. Ela e o "servidor" em si.
app = FastAPI()


# 2) Formato do que o cliente envia: {"fonte": "..."}
class Entrada(BaseModel):
    fonte: str


# 3) Le a imagem (URL ou base64)
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


# 4) Le o texto da imagem (OCR)
def ler_texto(imagem_rgb):
    bgr = np.ascontiguousarray(imagem_rgb[:, :, ::-1])
    resultado, _ = ocr(bgr)  # lista de [caixa, texto, confiança] ou None
    if not resultado:
        return ""
    return "\n".join(linha[1] for linha in resultado)


# 5) A palavra aparece no texto? (aceita pequenos erros de leitura)
def tem_palavra(texto, palavra):
    termo = r"\s+".join(regex.escape(parte) for parte in palavra.split())
    padrao = rf"(?e)(?<!\w)(?:{termo}){{e<={ERROS_FUZZY}}}(?!\w)"
    return regex.search(padrao, texto, flags=regex.IGNORECASE) is not None


def nome_da_classe(classe):
    return MAPA_CLASSES.get(int(classe), int(classe))


# 6) A rota. O cliente envia um POST para /prever
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

    # 7) O que a API devolve para o cliente
    return {
        "predicao": nome_da_classe(modelo.classes_[melhor]),
        "confianca": round(float(probabilidades[melhor]), 4),
        "probabilidades": {
            str(nome_da_classe(classe)): round(float(p), 4)
            for classe, p in zip(modelo.classes_, probabilidades)
        },
        "palavras_encontradas": [p for p, achou in encontradas.items() if achou],
        "texto_lido": bool(texto.strip()),  # False = o OCR não leu nada
        "shape": list(imagem.shape),        # [altura, largura, canais]
    }