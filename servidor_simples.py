from fastapi import FastAPI
from pydantic import BaseModel

# Cria a aplicação (a nossa API)
app = FastAPI()


# Aqui definimos COMO o pedido deve chegar.
# O cliente precisa enviar um dado chamado "nome", do tipo texto.
class Entrada(BaseModel):
    fonte: str


@app.post("predict")
def predict(entrada: Entrada):
    mensagem = f"Olá {entrada.fonte}, sua requisição foi recebida com sucesso!"
    
    return {"mensagem": mensagem}
