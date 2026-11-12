# %% [markdown]
# # Inferência: classificador de documentos (arquivo / URL / base64)
# Requisito: salvar o seu script extrator como `extracao_features.py`
# na MESMA pasta deste notebook (ele não precisa ser alterado).

# %%
# Instala as bibliotecas necessárias (rode só se precisar)
# ATENÇÃO: faltavam `xgboost` (joblib.load precisa dele) e `regex` (o extrator usa).
# %pip install joblib numpy pandas pillow requests opencv-python-headless matplotlib scikit-learn xgboost pytesseract regex

# %%
import base64
import importlib
import io
import os
import re
import sys

import poplib
import joblib
import numpy as np
import pandas as pd
import requests
import cv2
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image, ImageOps

# Windows: descomente e ajuste se o Tesseract não estiver no PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# %%
# Configurações
CAMINHO_MODELO = "models/modelo_xgboost.joblib"
USAR_DETECTOR_OPENCV = False   # mantenha False: o modelo usa o texto do documento inteiro
TIPO_DETECTOR = "rosto"
MAPA_CLASSES = {0: "Invalidos", 1: "RG_Antigo", 2: "RG_novo"}
MODO_EXTRACAO = "script"
NOME_MODULO_FEATURES = "extracao_features"
FORMATO_IMAGEM_EXTRACAO = "numpy_bgr"
VALIDAR_FEATURES = True
ERROS_FUZZY = 2                # mesmo valor usado ao gerar o CSV de treino
COLUNAS_MODELO = [
    "Identidade", "Carteira de Identidade", "Registro Geral", "CPF", "Nome",
    "Filiação", "Data de Nascimento", "Naturalidade", "Nacionalidade", "Sexo",
    "Órgão Expedidor", "Data de Expedição", "SSP", "Polegar Direito",
    "Assinatura do Titular", "Válida em todo território nacional",
    "Carteira de Identidade Nacional", "Identidade Nacional", "Validade",
    "República Federativa do Brasil", "National Identity Card", "Identity Card",
    "Name", "Date of Birth", "Place of Birth", "Nationality",
]
IMAGEM_VALIDACAO = "Dataset/Teste/00018418_in.jpg"

# %%
# Carregar modelo
# O .joblib é um DICT (você mesmo imprime modelo.keys() no fim), então o
# estimador está dentro dele. Aqui separamos as peças uma vez só.
def desempacotar_modelo(artefato):
    if not isinstance(artefato, dict):
        return artefato, None, None

    estimador = artefato.get("modelo")
    if estimador is None:
        estimador = next(
            (v for v in artefato.values() if hasattr(v, "predict")), None
        )
    if estimador is None:
        raise KeyError(
            "Não achei um estimador com .predict no arquivo. "
            f"Chaves encontradas: {list(artefato.keys())}"
        )

    encoder = next(
        (artefato[k] for k in ("label_encoder", "encoder", "le") if k in artefato),
        None,
    )
    colunas = next(
        (list(artefato[k]) for k in ("colunas", "features", "feature_names", "columns")
         if k in artefato),
        None,
    )
    return estimador, encoder, colunas


artefato = joblib.load(CAMINHO_MODELO)
modelo, ENCODER, COLUNAS_ARTEFATO = desempacotar_modelo(artefato)

print("Conteúdo do arquivo:",
      list(artefato.keys()) if isinstance(artefato, dict) else type(artefato).__name__)
print("Estimador:", type(modelo).__name__)

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

# Colunas finais (e a ORDEM delas): o que o modelo guardou tem prioridade.
if nomes_modelo is not None:
    COLUNAS_FINAIS = list(nomes_modelo)
elif COLUNAS_ARTEFATO:
    COLUNAS_FINAIS = COLUNAS_ARTEFATO
else:
    COLUNAS_FINAIS = list(COLUNAS_MODELO)

if VALIDAR_FEATURES:
    if n_features_esperado is not None and n_features_esperado != len(COLUNAS_FINAIS):
        raise ValueError(
            f"O modelo espera {n_features_esperado} features, "
            f"mas a lista tem {len(COLUNAS_FINAIS)}."
        )
    if set(COLUNAS_FINAIS) != set(COLUNAS_MODELO):
        print("[AVISO] As colunas do modelo diferem de COLUNAS_MODELO. "
              "Usando as do modelo:", COLUNAS_FINAIS)

# %%
# Tradução da predição (estas funções não existiam no seu código)
def para_python(valor):
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    return valor


def traduzir_predicao(valor):
    valor = para_python(valor)

    if MAPA_CLASSES is not None:
        return MAPA_CLASSES.get(valor, valor)

    if ENCODER is not None:
        try:
            if hasattr(ENCODER, "inverse_transform"):
                return para_python(ENCODER.inverse_transform(np.asarray([valor]))[0])
            return para_python(ENCODER[int(valor)])  # lista/array de nomes
        except Exception:
            return valor

    return valor

# %%
# Carregar imagem: arquivo, URL, data URI, base64 puro, bytes ou numpy
def eh_url(texto):
    return texto.lower().startswith(("http://", "https://"))


def decodificar_base64(texto):
    if texto.startswith("data:"):
        texto = texto.split(",", 1)[1]

    texto = re.sub(r"\s+", "", texto)
    texto = texto.replace("-", "+").replace("_", "/")   # base64 url-safe
    texto += "=" * (-len(texto) % 4)                     # padding que às vezes some

    try:
        return base64.b64decode(texto, validate=True)
    except Exception as erro:
        raise ValueError(
            "Não consegui interpretar a entrada. Envie um caminho de arquivo "
            "existente, uma URL http(s) ou uma imagem em base64. "
            f"(pasta atual: {os.getcwd()})"
        ) from erro


def obter_bytes(fonte):
    if isinstance(fonte, (bytes, bytearray, memoryview)):
        return bytes(fonte)

    if isinstance(fonte, os.PathLike):
        fonte = os.fspath(fonte)

    if not isinstance(fonte, str):
        raise TypeError(
            "A fonte da imagem deve ser texto (caminho, URL, base64), "
            "bytes ou um array numpy."
        )

    fonte = fonte.strip().strip("\"'")

    if eh_url(fonte):
        resposta = requests.get(
            fonte, timeout=15, headers={"User-Agent": "Mozilla/5.0"}
        )
        resposta.raise_for_status()
        return resposta.content

    if fonte.startswith("data:"):
        return decodificar_base64(fonte)

    # base64 gigante nunca é caminho de arquivo; só testa se for curto
    if len(fonte) < 4096 and os.path.isfile(fonte):
        with open(fonte, "rb") as arquivo:
            return arquivo.read()

    return decodificar_base64(fonte)


def decodificar_imagem(dados):
    """Devolve imagem BGR. cv2.imdecode se comporta como o cv2.imread
    usado na geração do CSV (inclusive aplicando a orientação EXIF)."""
    img = cv2.imdecode(np.frombuffer(dados, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        try:
            pil = ImageOps.exif_transpose(Image.open(io.BytesIO(dados))).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as erro:
            raise ValueError(
                "Os dados recebidos não são uma imagem válida "
                "(jpg, png, webp, bmp...)."
            ) from erro

    return img


def carregar_imagem(fonte):
    """Retorna SEMPRE numpy BGR (mesmo formato do treino)."""
    if isinstance(fonte, np.ndarray):
        return fonte
    return decodificar_imagem(obter_bytes(fonte))

# %%
# Detectar rosto (opcional)
ARQUIVOS_DETECTORES = {
    "rosto": "haarcascade_frontalface_default.xml",
    "olhos": "haarcascade_eye.xml",
    "corpo": "haarcascade_fullbody.xml",
}

detector = None

if USAR_DETECTOR_OPENCV:
    if TIPO_DETECTOR not in ARQUIVOS_DETECTORES:
        raise ValueError(f"Tipo de detector inválido: {TIPO_DETECTOR}")

    caminho_xml = os.path.join(cv2.data.haarcascades, ARQUIVOS_DETECTORES[TIPO_DETECTOR])
    detector = cv2.CascadeClassifier(caminho_xml)

    if detector.empty():
        raise RuntimeError(f"Não consegui carregar o detector em: {caminho_xml}")

    print(f"Detector '{TIPO_DETECTOR}' carregado.")
else:
    print("Detector OpenCV desligado: a imagem inteira será usada.")


def detectar_regioes(imagem_bgr):
    altura_img, largura_img = imagem_bgr.shape[:2]
    imagem_inteira = [{
        "caixa": (0, 0, largura_img, altura_img),
        "recorte": imagem_bgr,
    }]

    if detector is None:
        return imagem_inteira

    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    caixas = detector.detectMultiScale(
        cinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    regioes = [{
        "caixa": (int(x), int(y), int(w), int(h)),
        "recorte": imagem_bgr[y:y + h, x:x + w],
    } for (x, y, w, h) in caixas]

    # Nada detectado: usa a imagem inteira em vez de devolver lista vazia
    return regioes or imagem_inteira

# %%
# Extração de features (usa as funções do seu extracao_features.py)
_MODULO_FEATURES = None


def obter_modulo_features():
    global _MODULO_FEATURES

    if _MODULO_FEATURES is None:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())   # era .append(0, ...) -> TypeError

        try:
            _MODULO_FEATURES = importlib.import_module(NOME_MODULO_FEATURES)
        except ModuleNotFoundError as erro:
            if erro.name == NOME_MODULO_FEATURES:
                raise RuntimeError(
                    f"Não encontrei '{NOME_MODULO_FEATURES}.py' em {os.getcwd()}. "
                    "Salve o script extrator com esse nome nessa pasta."
                ) from erro
            raise  # falta uma dependência DO módulo (regex, pytesseract...)

    return _MODULO_FEATURES


def extrair_features(imagem_bgr):
    if MODO_EXTRACAO != "script":
        raise ValueError(f"MODO_EXTRACAO inválido: {MODO_EXTRACAO}")
    if FORMATO_IMAGEM_EXTRACAO != "numpy_bgr":
        raise ValueError(f"FORMATO_IMAGEM_EXTRACAO inválido: {FORMATO_IMAGEM_EXTRACAO}")

    modulo = obter_modulo_features()

    if not hasattr(modulo, "extrair_features"):
        raise RuntimeError(
            f"'{NOME_MODULO_FEATURES}.py' ainda não tem a função extrair_features. "
            "Cole nele o conteúdo de extracao_features_adicionar.py."
        )

    # Toda a extração (orientação, OCR, bag of words) acontece no módulo
    return modulo.extrair_features(
        imagem_bgr, colunas=COLUNAS_FINAIS, erros=ERROS_FUZZY
    )




def preparar_entrada(df):
    # Modelo treinado com DataFrame -> DataFrame (nomes e ordem batendo).
    # Modelo treinado com numpy -> numpy.
    return df if nomes_modelo is not None else df.to_numpy()

# %%
# Predição
def prever(fonte):
    imagem = carregar_imagem(fonte)
    regioes = detectar_regioes(imagem)

    resultados = []

    for regiao in regioes:
        features = preparar_entrada(extrair_features(regiao["recorte"]))

        predicao = modelo.predict(features)[0]

        item = {
            "caixa": regiao["caixa"],
            "predicao_bruta": para_python(predicao),
            "predicao": traduzir_predicao(predicao),
        }

        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(features)[0]
            classes = getattr(modelo, "classes_", range(len(probabilidades)))
            item["probabilidades"] = {
                str(traduzir_predicao(classe)): round(float(p), 4)
                for classe, p in zip(classes, probabilidades)
            }

        resultados.append(item)

    return {"quantidade_regioes": len(resultados), "resultados": resultados}


def mostrar_resultado(fonte, resultado):
    imagem = carregar_imagem(fonte)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))

    rotulos = []
    for r in resultado["resultados"]:
        if USAR_DETECTOR_OPENCV:
            x, y, w, h = r["caixa"]
            plt.gca().add_patch(
                plt.Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2)
            )
        rotulo = str(r["predicao"])
        if "probabilidades" in r:
            rotulo += f" ({max(r['probabilidades'].values()):.0%})"
        rotulos.append(rotulo)

    plt.title(" | ".join(rotulos))
    plt.axis("off")
    plt.show()

# %%
# Uso: a mesma chamada serve para os três formatos
fonte = IMAGEM_VALIDACAO                       # arquivo
# fonte = "https://site.com/documento.jpg"     # URL
# fonte = "data:image/jpeg;base64,/9j/4AAQ..." # base64 (com ou sem o prefixo data:)

imagem = carregar_imagem(fonte)   # decodifica uma vez só (evita baixar a URL 2x)
resultado = prever(imagem)
print(resultado)
mostrar_resultado(imagem, resultado)