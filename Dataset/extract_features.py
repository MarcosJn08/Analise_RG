
from pathlib import Path

import cv2
import pytesseract
import pandas as pd
import regex as re


PROJETO = Path(__file__).resolve().parent
PASTA_IMAGENS = PROJETO / "Dataset_Filtrado"
CSV_SAIDA = PROJETO / "features.csv"

CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def carregar_imagens(pasta_imagens):
    if not pasta_imagens.is_dir():
        raise FileNotFoundError(
            f"Pasta de imagens não encontrada: {pasta_imagens}"
        )

    caminhos = []

    for arquivo in sorted(
        pasta_imagens.rglob("*"),
        key=lambda item: str(item).lower()
    ):
        if not arquivo.is_file():
            continue

        if arquivo.suffix.lower() not in EXTENSOES:
            continue

        img = cv2.imread(str(arquivo))

        if img is not None:
            caminhos.append(arquivo)
            print(f"[IMAGEM] {arquivo.relative_to(pasta_imagens)}")
        else:
            print(f"[AVISO] Imagem inválida: {arquivo.name}")

    return caminhos


def detectar_rosto(img, face_cascade):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) > 0:
        x, y, w, h = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        return int(x), int(y), int(w), int(h)

    return None, None, None, None


def detectar_qrcode(img):
    detector = cv2.QRCodeDetector()
    dados, bbox, _ = detector.detectAndDecode(img)
    return int(bool(dados))


def extrair_texto(img):
    return pytesseract.image_to_string(
        img,
        lang="por+eng"
    )


def contar_palavras(texto):
    return len(texto.split())


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


def buscar_palavra_aproximada(texto, palavra, erros=2):
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
        df[palavra] = df["texto_extraido"].apply(
            lambda texto: int(
                len(
                    buscar_palavra_aproximada(
                        texto or "",
                        palavra,
                        erros=erros
                    )
                ) > 0
            )
        )

    return df


def processar_imagem(caminho, face_cascade):
    img = cv2.imread(str(caminho))

    if img is None:
        print(f"[AVISO] Não foi possível abrir: {caminho.name}")
        return None

    x, y, w, h = detectar_rosto(img, face_cascade)
    texto = extrair_texto(img)
    qrcode = detectar_qrcode(img)

    return {
        "nome_arquivo": caminho.name,
        "categoria": caminho.parent.name,
        "rosto_detectado": int(x is not None),
        "qrcode_detectado": qrcode,
        "texto_extraido": texto.strip(),
        "quantidade_palavras": contar_palavras(texto)
    }


def main():
    print(f"Projeto: {PROJETO}")
    print(f"Pasta de imagens: {PASTA_IMAGENS}")
    print(f"Arquivo de saída: {CSV_SAIDA}")

    if not PASTA_IMAGENS.is_dir():
        raise FileNotFoundError(
            f"Pasta não encontrada: {PASTA_IMAGENS}"
        )

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if face_cascade.empty():
        raise RuntimeError(
            f"Não foi possível carregar o classificador: {CASCADE_PATH}"
        )

    caminhos_imagens = carregar_imagens(PASTA_IMAGENS)
    resultados = []

    for indice, caminho in enumerate(caminhos_imagens, start=1):
        print(
            f"[{indice}/{len(caminhos_imagens)}] "
            f"Processando {caminho.name}"
        )

        try:
            resultado = processar_imagem(caminho, face_cascade)

            if resultado is not None:
                resultados.append(resultado)

        except Exception as erro:
            print(f"[ERRO] {caminho.name}: {erro}")

    if not resultados:
        print("Nenhuma imagem foi processada.")
        return

    df = pd.DataFrame(resultados)

    df = gerar_features_bag_palavras(
        df,
        obter_bag_palavras(),
        erros=2
    )

    df.to_csv(
        CSV_SAIDA,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"CSV gerado: {CSV_SAIDA}")
    print(f"Imagens processadas: {len(resultados)}")


if __name__ == "__main__":
    main()
