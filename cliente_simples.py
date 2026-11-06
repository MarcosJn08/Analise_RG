# ============================================================
# PARTE 1: O CLIENTE (o "cliente" do restaurante)
# ============================================================
# O cliente faz o pedido e recebe a resposta.
# Antes de rodar este arquivo, o servidor precisa estar ligado.
#
# Para rodar:  python cliente_simples.py
# ============================================================

import requests

# Endereço da API: onde o servidor está + a rota que queremos usar
url = "http://127.0.0.1:8000/saudacao"

# O dado que vamos enviar (um dicionário Python, que vira JSON)
dados = {"nome": input("Digite seu nome: ")}

# Envia o pedido (POST) e guarda a resposta
resposta = requests.post(url, json=dados)

# Mostra o código de status (200 significa "deu tudo certo")
print("Status:", resposta.status_code)

# Mostra o conteúdo que o servidor devolveu
print("Resposta:", resposta.json())
print("Mensagem:", resposta.json()["mensagem"])


# ============================================================
# TESTANDO A SEGUNDA ROTA: /dobro
# ============================================================
# Mesma ideia, mas agora enviamos um número em vez de um nome.
# O servidor multiplica por 2 e devolve o resultado.

url_dobro = "http://127.0.0.1:8000/dobro"

numero = float(input("\nDigite um número: "))

resposta = requests.post(url_dobro, json={"numero": numero})

print("Status:", resposta.status_code)
print("Resposta:", resposta.json())
print("O dobro de", numero, "é", resposta.json()["resultado"])