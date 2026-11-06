# Instala as bibliotecas necessárias (rode só se precisar)
# %pip install joblib numpy pandas pillow requests opencv-python-headless matplotlib scikit-learn pytesseract
import base64
import importlib.util
import io
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
import requests
import cv2
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

#Configurações
CAMINHO_MODELO = "models/modelo_xgboost.joblib"
USAR_DETECTOR_OPENCV = True
TIPO_DETECTOR = "rosto"
MAPA_CLASSES = None
MODO_EXTRACAO = "script"
NOME_MODULO_FEATURES = "services/extracao_features"
FORMATO_IMAGEM_EXTRACAO = "numpy_bgr"
VALIDAR_FEATURES = True
COLUNAS_MODELO = ["Identidade","Carteira de Identidade","Registro Geral","CPF","Nome","Filiação","Data de Nascimento","Naturalidade","Nacionalidade","Sexo","Órgão Expedidor","Data de Expedição","SSP","Polegar Direito","Assinatura do Titular","Válida em todo território nacional","Carteira de Identidade Nacional","Identidade Nacional","Validade","República Federativa do Brasil","National Identity Card","Identity Card","Name","Date of Birth","Place of Birth","Nationality"]
IMAGEM_VALIDACAO = "Dataset/Teste/doc_0990.jpg"


# Carregar modelo
modelo = joblib.load(CAMINHO_MODELO)
print("Modelo carregado:", type(modelo).__name__)


n_features_esperado = getattr(modelo, "n_features_in_", None)

nomes_modelo = getattr(modelo, "feature_names_in_", None)

if n_features_esperado is not None:
    print("Número de features esperado:", n_features_esperado)
else:
    print("Este modelo não informa quantas features espera.")

if nomes_modelo is not None:
    print("Nomes das colunas guardados no modelo:", list(nomes_modelo))
else:
    print("Este modelo não guarda nomes de colunas (provavelmente foi treinado com numpy).")

if hasattr(modelo, "classes_"):
    print("Classes que o modelo conhece:", np.asarray(modelo.classes_).tolist())

# Carregar imagem
def eh_url(texto):
    return texto.lower().startswith(("http://", "https://"))


def carregar_imagem(fonte):

    if isinstance(fonte, (bytes, bytearray)):
        dados = bytes(fonte)

    elif isinstance(fonte, str):
        fonte = fonte.strip()

        if eh_url(fonte):
            resposta = requests.get(
                fonte,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            resposta.raise_for_status()
            dados = resposta.content

        elif os.path.exists(fonte):
            with open(fonte, "rb") as arquivo:
                dados = arquivo.read()

        else:
            if fonte.startswith("data:"):
                fonte = fonte.split(",", 1)[1]

            fonte = re.sub(r"\s+", "", fonte)

            try:
                dados = base64.b64decode(fonte, validate=True)
            except Exception:
                raise ValueError(
                    "Não consegui interpretar a entrada. "
                    "Envie um caminho de arquivo válido, uma URL "
                    "ou uma imagem em base64."
                )

    else:
        raise TypeError(
            "A fonte da imagem deve ser texto "
            "(caminho, URL, base64) ou bytes."
        )

    imagem = Image.open(io.BytesIO(dados)).convert("RGB")
    return np.array(imagem)


# Detectar rosto
ARQUIVOS_DETECTORES = {
    "rosto": "haarcascade_frontalface_default.xml",
    "olhos": "haarcascade_eye.xml",
    "corpo": "haarcascade_fullbody.xml",
}

detector = None

if USAR_DETECTOR_OPENCV:

    if TIPO_DETECTOR not in ARQUIVOS_DETECTORES:
        raise ValueError(
            f"Tipo de detector inválido: {TIPO_DETECTOR}"
        )

    caminho_xml = os.path.join(
        cv2.data.haarcascades,
        ARQUIVOS_DETECTORES[TIPO_DETECTOR]
    )

    detector = cv2.CascadeClassifier(caminho_xml)

    if detector.empty():
        raise RuntimeError(
            f"Não consegui carregar o detector em: {caminho_xml}"
        )

    print(f"Detector '{TIPO_DETECTOR}' carregado.")

else:
    print("Detector OpenCV desligado: a imagem inteira será usada.")


def detectar_regioes(imagem_bgr):
    """
    Devolve uma lista de regiões. Cada região contém:
      - caixa: (x, y, largura, altura)
      - recorte: pedaço da imagem dentro da caixa
    """

    altura_img, largura_img = imagem_bgr.shape[:2]

    if detector is None:
        return [{
            "caixa": (0, 0, largura_img, altura_img),
            "recorte": imagem_bgr
        }]

    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)

    caixas = detector.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    regioes = []

    for (x, y, largura, altura) in caixas:
        regioes.append({
            "caixa": (
                int(x),
                int(y),
                int(largura),
                int(altura)
            ),
            "recorte": imagem_bgr[
                y:y + altura,
                x:x + largura
            ]
        })

    return regioes


carregar_imagem(IMAGEM_VALIDACAO)