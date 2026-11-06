from fastapi import FastAPI
from pydantic import BaseModel


# Cria a aplicação (a nossa API)
app = FastAPI()


# Aqui definimos COMO o pedido deve chegar.
# O cliente precisa enviar um dado chamado "nome", do tipo texto.
class Pedido(BaseModel):
    nome: str


# Uma "rota" é um endereço da API.
# @app.post significa: "esta função responde a pedidos do tipo POST
# no endereço /saudacao".
@app.post("/saudacao")
def saudacao(pedido: Pedido):
    # Aqui acontece o processamento (neste caso, algo bem simples)
    mensagem = f"Olá, {pedido.nome}! Sua primeira API está funcionando."

    # O que retornamos vira a resposta enviada ao cliente (em formato JSON)
    return {"mensagem": mensagem}
