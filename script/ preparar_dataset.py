from pathlib import Path
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import messagebox
import random
import shutil


# Pasta principal que contem as pastas do dataset.
BASE = Path(
    "/home/marcos/Área de trabalho/Projetos/Analise_RG/Dataset/DataSet02"
)

# Pasta de destino dos arquivos classificados.
DESTINO = Path("/home/marcos/Área de trabalho/Projetos/Analise_RG/Dataset/Dataset_Filtrado")

random.seed(42)

# Pastas de saida
ANTIGO = DESTINO / "RG_antigo"
NOVO = DESTINO / "RG_novo"
INVALIDO = DESTINO / "Invalido"

for pasta in (ANTIGO, NOVO, INVALIDO):
    pasta.mkdir(parents=True, exist_ok=True)

EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Verifica a pasta principal
if not BASE.is_dir():
    raise FileNotFoundError(
        f"A pasta Dataset nao foi encontrada:\n{BASE}\n\n"
        "Ajuste a variavel BASE para o caminho correto."
    )

# Analisa todas as subpastas e encontra imagens, sem depender
# de nomes especificos de pastas ou do sufixo *_in.jpg.
imagens_por_pasta = {}
for arquivo in sorted(BASE.rglob("*")):
    if (
        arquivo.is_file()
        and arquivo.suffix.lower() in EXTENSOES
        and not arquivo.name.startswith(".")
        and DESTINO not in arquivo.parents
    ):
        pasta_relativa = arquivo.parent.relative_to(BASE)
        nome_pasta = str(pasta_relativa) if str(pasta_relativa) != "." else "(raiz)"
        imagens_por_pasta.setdefault(nome_pasta, []).append(arquivo)

imagens = [
    arquivo
    for arquivos in imagens_por_pasta.values()
    for arquivo in arquivos
]
imagens = sorted(imagens, key=lambda p: str(p).lower())
random.shuffle(imagens)

print(f"Dataset analisado: {BASE}")
print(f"Total de imagens encontradas: {len(imagens)}")
print("Quantidade por pasta:")
for nome_pasta, arquivos in sorted(imagens_por_pasta.items()):
    print(f"  {nome_pasta}: {len(arquivos)} imagem(ns)")

if not imagens:
    raise RuntimeError(
        f"Nenhuma imagem foi encontrada dentro de:\n{BASE}\n"
        "Verifique se as imagens estao em subpastas e se usam formatos "
        "JPG, JPEG, PNG, WEBP, BMP, TIF ou TIFF."
    )

def contar(pasta):
    return sum(
        1 for arquivo in pasta.iterdir()
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES
    )

def destino_unico(arquivo, pasta):
    destino = pasta / arquivo.name
    n = 1
    while destino.exists():
        destino = pasta / f"{arquivo.stem}_{n}{arquivo.suffix}"
        n += 1
    return destino

# Evita repetir imagens ja copiadas para qualquer categoria.
# O caminho completo da origem e registrado em um arquivo de controle.
REGISTRO = DESTINO / "imagens_classificadas.txt"
classificadas = set()
if REGISTRO.exists():
    classificadas = {
        linha.strip()
        for linha in REGISTRO.read_text(encoding="utf-8").splitlines()
        if linha.strip()
    }

pendentes = [p for p in imagens if str(p.resolve()) not in classificadas]
indice = 0
imagem_tk = None
arquivo_atual = None

janela = tk.Tk()
janela.title("Analisador e classificador de imagens do dataset")
janela.geometry("900x760")

titulo = tk.Label(janela, text="Classificador de imagens", font=("Arial", 16, "bold"))
titulo.pack(pady=8)

foto = tk.Label(janela, text="")
foto.pack(pady=8)

nome = tk.Label(janela, text="", font=("Arial", 10), wraplength=850)
nome.pack(pady=5)

contagem = tk.Label(janela, text="", font=("Arial", 11), justify="center")
contagem.pack(pady=8)

info = tk.Label(
    janela,
    text=f"Imagens encontradas: {len(imagens)} | Pendentes nesta execucao: {len(pendentes)}",
    font=("Arial", 10),
    wraplength=850,
)
info.pack(pady=5)

def atualizar_contagem():
    contagem.config(
        text=(
            f"RG antigo: {contar(ANTIGO)}  |  "
            f"RG novo: {contar(NOVO)}  |  "
            f"Inválido: {contar(INVALIDO)}"
        )
    )

def registrar(arquivo):
    classificadas.add(str(arquivo.resolve()))
    REGISTRO.write_text(
        "\n".join(sorted(classificadas)) + "\n",
        encoding="utf-8",
    )

def atualizar():
    global indice, imagem_tk, arquivo_atual

    arquivo_atual = None
    atualizar_contagem()

    while indice < len(pendentes):
        atual = pendentes[indice]
        if str(atual.resolve()) not in classificadas:
            break
        indice += 1

    if indice >= len(pendentes):
        titulo.config(text="Fim das imagens")
        foto.config(image="", text="")
        nome.config(text="")
        info.config(
            text=(
                f"Analise concluida. Imagens encontradas: {len(imagens)}.\n"
                f"Arquivos classificados nesta pasta de destino: {DESTINO}"
            )
        )
        messagebox.showinfo(
            "Concluido",
            "Nao ha mais imagens pendentes nesta execucao.\n\n"
            f"Total encontrado: {len(imagens)}\n"
            f"Destino: {DESTINO}",
        )
        return

    arquivo = pendentes[indice]
    arquivo_atual = arquivo

    try:
        with Image.open(arquivo) as original:
            img = ImageOps.exif_transpose(original).convert("RGB")
            img.thumbnail((820, 480))
            imagem_tk = ImageTk.PhotoImage(img)
        foto.config(image=imagem_tk, text="")
    except Exception as erro:
        print(f"Arquivo nao pode ser aberto: {arquivo} - {erro}")
        registrar(arquivo)
        indice += 1
        atualizar()
        return

    titulo.config(text=f"Imagem {indice + 1} de {len(pendentes)}")
    nome.config(
        text=(
            f"Arquivo: {arquivo.name}\n"
            f"Pasta: {arquivo.parent.relative_to(BASE)}"
        )
    )
    info.config(
        text="Escolha uma categoria. Se nao souber classificar, clique em Pular."
    )

def classificar(pasta):
    global indice

    if arquivo_atual is None:
        return

    arquivo = arquivo_atual
    try:
        shutil.copy2(arquivo, destino_unico(arquivo, pasta))
        registrar(arquivo)
    except OSError as erro:
        messagebox.showerror("Erro ao copiar", str(erro))
        return

    indice += 1
    atualizar()

def pular():
    global indice
    if arquivo_atual is None:
        return
    # O arquivo pulado nao e registrado como classificado.
    indice += 1
    atualizar()

botoes = tk.Frame(janela)
botoes.pack(pady=12)

tk.Button(
    botoes, text="1 - RG ANTIGO", width=18,
    command=lambda: classificar(ANTIGO)
).grid(row=0, column=0, padx=5, pady=5)

tk.Button(
    botoes, text="2 - RG NOVO", width=18,
    command=lambda: classificar(NOVO)
).grid(row=0, column=1, padx=5, pady=5)

tk.Button(
    botoes, text="3 - INVALIDO", width=18,
    command=lambda: classificar(INVALIDO)
).grid(row=0, column=2, padx=5, pady=5)

tk.Button(
    botoes, text="Pular", width=12,
    command=pular
).grid(row=1, column=0, columnspan=3, pady=8)

tk.Label(
    janela,
    text=(
        "Este programa procura imagens em todas as subpastas de Dataset.\n"
        "A classificacao e manual: o programa nao decide automaticamente "
        "se um documento e valido ou invalido.\n"
        "Os arquivos originais sao mantidos; as classificacoes sao copiadas "
        "para a pasta de destino."
    ),
    font=("Arial", 9),
    justify="center",
    wraplength=850,
).pack(pady=10)

if not pendentes:
    messagebox.showinfo(
        "Sem imagens pendentes",
        "Todas as imagens encontradas ja constam no registro de classificacao.\n"
        f"Registro: {REGISTRO}",
    )
    janela.destroy()
else:
    atualizar()
    janela.mainloop()
