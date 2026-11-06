from fastapi import FastAPI
from pydantic import BaseModel

# Cria a aplicação (a nossa API)
app = FastAPI()


# Aqui definimos COMO o pedido deve chegar.
# O cliente precisa enviar um dado chamado "nome", do tipo texto.
class Entrada(BaseModel):
    fonte: str
# Uma "rota" é um endereço da API.
@app.post("/predict")
def predict(entrada: Entrada):
    # Aqui acontece o processamento (neste caso, algo bem simples)
    mensagem = f"Olá, {entrada.fonte}! Sua primeira API está funcionando."

    # O que retornamos vira a resposta enviada ao cliente (em formato JSON)
    return {"mensagem": mensagem}


# ------------------------------------------------------------
# SEGUNDA ROTA: recebe um número e devolve ele multiplicado por 2
# ------------------------------------------------------------
# Cada rota pode esperar um tipo de dado diferente.
# Aqui o cliente precisa enviar um campo chamado "numero".
# Usamos float para aceitar tanto inteiros (5) quanto decimais (2.5).
class PedidoNumero(BaseModel):
    numero: float


@app.post("/dobro")
def dobro(pedido: PedidoNumero):
    resultado = pedido.numero * 2
    return {"numero_enviado": pedido.numero, "resultado": resultado}


# Uma rota GET simples, só para testar se o servidor está vivo
@app.get("/")
def inicio():
    return {"status": "servidor funcionando"}
