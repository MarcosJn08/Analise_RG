import gc
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from extract_features import ExtratorFeatures, carregar_imagem

# Configurações do Modelo
CAMINHO_MODELO = "models/modelo_xgboost.joblib"
MAPA_CLASSES = {0: "Invalidos", 1: "RG_Antigo", 2: "RG_novo"}

# Inicialização na inicialização da aplicação (para não recarregar a cada request)
modelo = joblib.load(CAMINHO_MODELO)
COLUNAS = list(modelo.feature_names_in_)  # 26 palavras-chave esperadas pelo modelo
extrator = ExtratorFeatures(colunas=COLUNAS)

# Aplicação FastAPI
app = FastAPI(
    title="API de Análise e Classificação de RG",
    description="Classifica documentos em Inválidos, RG Antigo ou RG Novo. Otimizado para o plano gratuito do Render.",
    version="2.0.0"
)


class Entrada(BaseModel):
    fonte: str


def nome_da_classe(classe):
    return MAPA_CLASSES.get(int(classe), str(classe))


@app.get("/")
@app.get("/health")
def healthcheck():
    """
    Endpoint leve para verificação de status e keep-alive no Render.
    Pode ser chamado por monitores (ex: UptimeRobot) a cada 10-14 min para evitar suspensão do serviço.
    """
    return {
        "status": "online",
        "modelo": "XGBoost",
        "total_features": len(COLUNAS)
    }


@app.post("/prever")
def prever(entrada: Entrada):
    # 1. Carregar a imagem (URL ou Base64)
    try:
        imagem = carregar_imagem(entrada.fonte)
    except Exception as erro:
        # HTTP 400: erro atribuível à requisição do cliente
        raise HTTPException(status_code=400, detail=f"Não foi possível carregar a imagem: {erro}")

    # 2. Extrair features e classificar
    try:
        features, texto = extrator.extrair(imagem)

        # Libera o array numpy da imagem para desocupar RAM (crítico para os 512MB do Render)
        del imagem

        # Early Exit: se o OCR não detectou nenhum texto na imagem, é Inválido diretamente
        if not texto.strip():
            return {
                "predicao": "Invalidos",
                "confianca": 1.0,
                "probabilidades": {
                    "Invalidos": 1.0,
                    "RG_Antigo": 0.0,
                    "RG_novo": 0.0
                },
                "palavras_encontradas": [],
                "texto_detectado": False
            }

        features_df = pd.DataFrame([features], columns=COLUNAS)
        probabilidades = modelo.predict_proba(features_df)[0]
    except Exception as erro:
        # HTTP 500: erro no processamento do servidor
        raise HTTPException(status_code=500, detail=f"Erro ao classificar a imagem: {erro}")
    finally:
        # Força coleta de lixo pontual para evitar acúmulo de memória
        gc.collect()

    melhor = int(np.argmax(probabilidades))
    classe_predita = modelo.classes_[melhor]

    return {
        "predicao": nome_da_classe(classe_predita),
        "confianca": round(float(probabilidades[melhor]), 4),
        "probabilidades": {
            str(nome_da_classe(classe)): round(float(p), 4)
            for classe, p in zip(modelo.classes_, probabilidades)
        },
        "palavras_encontradas": [p for p, v in features.items() if v == 1],
        "texto_detectado": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
