"""
SERVIDOR: recebe uma URL ou uma imagem em base64
e devolve a imagem lida e o shape dela.

Para rodar:
    pip install fastapi uvicorn requests pillow numpy
    uvicorn servidor:app --reload

Depois abra http://127.0.0.1:8000/docs no navegador.
O FastAPI cria essa pagina sozinho e voce pode testar a API por la.
"""

import base64
import io

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

# 1) Cria a aplicacao. Ela e o "servidor" em si.
app = FastAPI()


# 2) Define o formato do que o cliente deve enviar.
#    Aqui o cliente manda um JSON assim: {"fonte": "..."}
class Entrada(BaseModel):
    fonte: str


# 3) A funcao que sabe ler a imagem (URL ou base64).
def carregar_imagem(fonte):
    fonte = fonte.strip()

    if fonte.startswith("http://") or fonte.startswith("https://"):
        # E uma URL: baixa a imagem da internet
        resposta = requests.get(fonte, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resposta.raise_for_status()
        dados = resposta.content
    else:
        # Senao, tratamos como base64
        if fonte.startswith("data:"):
            # Remove o prefixo "data:image/png;base64,"
            fonte = fonte.split(",", 1)[1]
        dados = base64.b64decode(fonte)

    imagem = Image.open(io.BytesIO(dados)).convert("RGB")
    return np.array(imagem)


# 4) Cria a rota. O cliente envia um POST para /ler-imagem
@app.post("/ler-imagem")
def ler_imagem(entrada: Entrada):
    try:
        imagem = carregar_imagem(entrada.fonte)
    except Exception as erro:
        # Codigo 400 significa: "o erro foi na requisicao do cliente"
        raise HTTPException(status_code=400, detail=f"Nao consegui ler a imagem: {erro}")

    # Converte o array de volta para uma imagem PNG e depois para base64,
    # porque JSON so transporta texto (nao transporta imagem direto).
    buffer = io.BytesIO()
    Image.fromarray(imagem).save(buffer, format="PNG")
    imagem_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # 5) O que a API devolve para o cliente
    return {
        "shape": list(imagem.shape),  # [altura, largura, canais]
        "imagem_base64": imagem_base64,
    }