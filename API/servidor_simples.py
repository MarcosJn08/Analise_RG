from fastapi import FastAPI
from pydantic import BaseModel

# cria a nossa API
app = FastAPI()

# dado esperado na rota /saudacao
class Pedido(BaseModel):
    nome: str

@app.post("/saudacao")
def saudacao(pedido: Pedido):
    msg = f"Olá, {pedido.nome}!"
    return {"mensagem": msg}

# dado esperado na rota /dobro
class PedidoNumero(BaseModel):
    numero: float

@app.post("/dobro")
def dobro(pedido: PedidoNumero):
    return {"resultado": pedido.numero * 2}
