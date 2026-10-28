
from pathlib import Path

import cv2
import pytesseract
import pandas as pd
import regex as re


# ============================================================
# CONFIGURAÇÕES
# ============================================================

# Diretório onde este script está localizado:
# /home/marcos/Área de trabalho/Projetos/Analise_RG/Dataset
PROJETO = Path(__file__).resolve().parent

# Pasta com as imagens que serão processadas.
PASTA_IMAGENS = PROJETO / "Dataset_Filtrado" / "DataSetFinal"

# Arquivo CSV de saída.
CSV_SAIDA = PROJETO / "features.csv"

# Classificador Haar Cascade do OpenCV.
CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ============================================================
# CARREGAR IMAGENS
# ============================================================

def carregar_imagens(pasta_imagens):
    caminhos = []

    if not pasta_imagens.is_dir():
        raise FileNotFoundError(
            f"Pasta de imagens não encontrada: {pasta_imagens}"
        )

    for arquivo in sorted(
        pasta_imagens.iterdir(),
        key=lambda item: item.name.lower()
    ):
        if not arquivo.is_file():
            continue

        if arquivo.suffix.lower() not in EXTENSOES:
            continue

        # Verifica se o OpenCV consegue ler a imagem.
        img = cv2.imread(str(arquivo))

        if img is not None:
            print(f"[IMAGEM] {arquivo.name}")
            caminhos.append(arquivo)
        else:
            print(f"[AVISO] Imagem inválida: {arquivo.name}")

    return caminhos


# ============================================================
# DETECTAR ROSTOS
# ============================================================

def detectar_rosto(img, face_cascade):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) > 0:
        # Seleciona o maior rosto detectado.
        x, y, w, h = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        return int(x), int(y), int(w), int(h)

    return None, None, None, None


# ============================================================
# DETECTAR QR CODE
# ============================================================

def detectar_qrcode(img):
    detector = cv2.QRCodeDetector()

    dados, bbox, _ = detector.detectAndDecode(img)

    if dados:
        return 1

    return 0


# ============================================================
# EXTRAIR TEXTO COM TESSERACT
# ============================================================

def extrair_texto(img):
    # Usa português e inglês.
    # Requer os pacotes de idioma por e eng do Tesseract.
    texto = pytesseract.image_to_string(
        img,
        lang="por+eng"
    )

    return texto


# ============================================================
# CONTAR PALAVRAS
# ============================================================

def contar_palavras(texto):
    return len(texto.split())


# ============================================================
# PROCESSAR IMAGEM
# ============================================================

def processar_imagem(caminho, face_cascade):
    img = cv2.imread(str(caminho))

    if img is None:
        print(f"[AVISO] Não foi possível abrir: {caminho.name}")
        return None

    x, y, w, h = detectar_rosto(img, face_cascade)

    texto = extrair_texto(img)

    quantidade_palavras = contar_palavras(texto)

    qrcode = detectar_qrcode(img)

    return {
        "nome_arquivo": caminho.name,
        "x_face": x,
        "y_face": y,
        "w_face": w,
        "h_face": h,
        "qrcode_detectado": qrcode,
        "texto_extraido": texto.strip(),
        "quantidade_palavras": quantidade_palavras
    }


# ============================================================
# BAG OF WORDS
# ============================================================

def obter_bag_palavras():
    return [
        "Identidade",
        "nome",
        "cpf",
        "Registro Geral",
        "Sexo",
        "Nacionalidade",
        "DOC. ORIGEM",
        "name",
        "date of birth",
        "Place of birth",
        "Nationality",
        "Sex"
    ]


# ============================================================
# BUSCA APROXIMADA COM REGEX FUZZY
# ============================================================

def buscar_palavra_aproximada(texto, palavra, erros=2):
    """
    Busca um termo no texto permitindo erros de OCR.

    erros=0: correspondência exata.
    erros=1: permite um erro.
    erros=2: permite até dois erros.

    Retorna as correspondências encontradas.
    """

    partes = palavra.split()


    padrao_termo = r"\s+".join(
        re.escape(parte) for parte in partes
    )

    padrao = (
        rf"(?e)(?<!\w)"
        rf"(?:{padrao_termo})"
        rf"{{e<={erros}}}"
        rf"(?!\w)"
    )

    return list(
        re.finditer(
            padrao,
            texto,
            flags=re.IGNORECASE
        )
    )


def gerar_features_bag_palavras(df, bag_palavras, erros=2):

    for palavra in bag_palavras:
        coluna = f"freq_{palavra}"

        df[coluna] = df["texto_extraido"].apply(
            lambda texto: len(
                buscar_palavra_aproximada(
                    texto or "",
                    palavra,
                    erros=erros
                )
            )
        )

    return df

def gerar_csv(df, nome_arquivo):
    df.to_csv(
        nome_arquivo,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n[OK] CSV gerado: {nome_arquivo}")

def main():
    print("=" * 60)
    print("EXTRAÇÃO DE FEATURES DE DOCUMENTOS")
    print("=" * 60)

    print(f"Projeto: {PROJETO}")
    print(f"Pasta de imagens: {PASTA_IMAGENS}")
    print(f"Saída CSV: {CSV_SAIDA}")

    # Verifica a pasta de imagens.
    if not PASTA_IMAGENS.is_dir():
        raise FileNotFoundError(
            f"\nA pasta não existe:\n{PASTA_IMAGENS}\n"
            "Confira o caminho e o nome da pasta."
        )

    # Carrega o detector facial.
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if face_cascade.empty():
        raise RuntimeError(
            "Não foi possível carregar o Haar Cascade.\n"
            f"Arquivo: {CASCADE_PATH}\n"
            "Verifique a instalação do OpenCV."
        )

    caminhos_imagens = carregar_imagens(PASTA_IMAGENS)

    if not caminhos_imagens:
        print("[AVISO] Nenhuma imagem válida encontrada.")
        return

    resultados = []

    total = len(caminhos_imagens)

    for indice, caminho in enumerate(caminhos_imagens, start=1):
        print(f"\n[{indice}/{total}] Processando {caminho.name}")

        try:
            resultado = processar_imagem(
                caminho,
                face_cascade
            )

            if resultado is not None:
                resultados.append(resultado)

        except Exception as erro:
            print(
                f"[ERRO] Falha em {caminho.name}: {erro}"
            )

    if not resultados:
        print("[AVISO] Nenhuma imagem foi processada com sucesso.")
        return

    df = pd.DataFrame(resultados)

    bag_palavras = obter_bag_palavras()

    print("\nGerando features da bag de palavras...")

    df = gerar_features_bag_palavras(
        df,
        bag_palavras,
        erros=2
    )

    gerar_csv(df, CSV_SAIDA)

    print("\nResumo:")
    print(f"Imagens encontradas: {total}")
    print(f"Imagens processadas: {len(resultados)}")
    print(f"Features geradas: {len(df.columns)}")
    print("\nProcessamento finalizado.")


if __name__ == "__main__":
    main()
