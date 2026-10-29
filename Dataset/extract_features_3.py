from pathlib import Path

import cv2
import numpy as np
import pytesseract
import pandas as pd
import regex as re
from pytesseract import Output


PROJETO = Path(__file__).resolve().parent
PASTA_IMAGENS = PROJETO / "Dataset_Filtrado02"
CSV_SAIDA = PROJETO / "Feature" / "features_6.csv"

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


# ---------------------------------------------------------------------
# Correção de orientação (0/90/180/270) e inclinação (ângulos pequenos)
# ---------------------------------------------------------------------
def _redimensionar(img, largura_max=1000):
    h, w = img.shape[:2]

    if w <= largura_max:
        return img

    escala = largura_max / w
    return cv2.resize(img, (largura_max, int(h * escala)))


def _rotacionar_90(img, graus):
    """Gira a imagem em múltiplos de 90° (sentido horário)."""
    if graus == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if graus == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if graus == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _pontuar_ocr(img):
    """Pontuação = soma da confiança das palavras reconhecidas.
    Imagem de cabeça pra baixo/de lado gera pouco texto legível."""
    dados = pytesseract.image_to_data(
        img,
        lang="por+eng",
        config="--psm 6",
        output_type=Output.DICT
    )

    pontuacao = 0.0

    for texto, conf in zip(dados["text"], dados["conf"]):
        try:
            conf = float(conf)
        except ValueError:
            continue

        if len(texto.strip()) >= 3 and conf >= 60:
            pontuacao += conf

    return pontuacao


def corrigir_orientacao(img):
    """Corrige rotações de 0/90/180/270 graus.

    1) Tenta o OSD do Tesseract (rápido).
    2) Se falhar ou tiver baixa confiança, testa as 4 rotações e
       escolhe a que gera mais texto legível.
    Retorna (imagem_corrigida, graus_aplicados).
    """
    pequena = _redimensionar(img)

    try:
        osd = pytesseract.image_to_osd(
            pequena,
            config="--psm 0",
            output_type=Output.DICT
        )

        if float(osd["orientation_conf"]) >= 2.0:
            graus = int(osd["rotate"])
            return _rotacionar_90(img, graus), graus

    except Exception:
        # OSD costuma falhar em imagens com pouco texto
        # ou sem o arquivo osd.traineddata instalado.
        pass

    melhor_graus = 0
    melhor_pontuacao = -1.0

    for graus in (0, 90, 180, 270):
        pontuacao = _pontuar_ocr(_rotacionar_90(pequena, graus))

        if pontuacao > melhor_pontuacao:
            melhor_pontuacao = pontuacao
            melhor_graus = graus

    return _rotacionar_90(img, melhor_graus), melhor_graus


def estimar_inclinacao(img, limite=15.0, passo=0.5):
    """Estima a inclinação (em graus) pelo método do perfil de projeção:
    o ângulo correto é aquele em que as linhas de texto ficam mais
    "separadas" (maior variância da soma por linha)."""
    gray = cv2.cvtColor(_redimensionar(img, 600), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    binaria = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    h, w = binaria.shape
    centro = (w / 2, h / 2)

    melhor_angulo = 0.0
    melhor_score = -1.0

    for angulo in np.arange(-limite, limite + passo, passo):
        matriz = cv2.getRotationMatrix2D(centro, float(angulo), 1.0)
        rot = cv2.warpAffine(
            binaria, matriz, (w, h),
            flags=cv2.INTER_NEAREST
        )

        perfil = rot.sum(axis=1).astype(np.float64)
        score = perfil.var()

        if score > melhor_score:
            melhor_score = score
            melhor_angulo = float(angulo)

    return melhor_angulo


def corrigir_inclinacao(img, minimo=0.5):
    """Endireita a imagem se estiver levemente torta."""
    angulo = estimar_inclinacao(img)

    if abs(angulo) < minimo:
        return img, 0.0

    h, w = img.shape[:2]
    matriz = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)

    corrigida = cv2.warpAffine(
        img, matriz, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    return corrigida, angulo


def normalizar_imagem(img):
    img, graus = corrigir_orientacao(img)
    img, inclinacao = corrigir_inclinacao(img)
    return img, graus, inclinacao


# ---------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------
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
        "Carteira de Identidade",
        "Registro Geral",
        "CPF",
        "Nome",
        "Filiação",
        "Data de Nascimento",
        "Naturalidade",
        "Nacionalidade",
        "Sexo",
        "Órgão Expedidor",
        "Data de Expedição",
        "SSP",
        "Polegar Direito",
        "Assinatura do Titular",
        "Válida em todo território nacional",
        "Carteira de Identidade Nacional",
        "Identidade Nacional",
        "QR Code",
        "Validade",
        "República Federativa do Brasil",
        "National Identity Card",
        "Identity Card",
        "Name",
        "Date of Birth",
        "Place of Birth",
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


def calcular_features_bag_palavras(texto, bag_palavras, erros=2):
    """Gera as colunas do bag of words a partir do texto,
    sem guardar o texto no resultado."""
    return {
        palavra: int(
            len(buscar_palavra_aproximada(texto, palavra, erros)) > 0
        )
        for palavra in bag_palavras
    }


def processar_imagem(caminho, face_cascade, bag_palavras, erros=2):
    img = cv2.imread(str(caminho))

    if img is None:
        print(f"[AVISO] Não foi possível abrir: {caminho.name}")
        return None

    # Corrige orientação e inclinação ANTES de qualquer extração
    img, graus, inclinacao = normalizar_imagem(img)

    if graus or inclinacao:
        print(
            f"        rotação: {graus}° | inclinação: {inclinacao:.1f}°"
        )

    x, y, w, h = detectar_rosto(img, face_cascade)
    qrcode = detectar_qrcode(img)

    # O texto é usado apenas em memória e descartado em seguida
    texto = extrair_texto(img)

    resultado = {
        "nome_arquivo": caminho.name,
        "categoria": caminho.parent.name,
        "rotacao_corrigida": graus,
        "inclinacao_corrigida": round(inclinacao, 1),
        "rosto_detectado": int(x is not None),
        "qrcode_detectado": qrcode,
        "quantidade_palavras": contar_palavras(texto)
    }

    resultado.update(
        calcular_features_bag_palavras(texto, bag_palavras, erros)
    )

    return resultado


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

    bag_palavras = obter_bag_palavras()
    caminhos_imagens = carregar_imagens(PASTA_IMAGENS)
    resultados = []

    for indice, caminho in enumerate(caminhos_imagens, start=1):
        print(
            f"[{indice}/{len(caminhos_imagens)}] "
            f"Processando {caminho.name}"
        )

        try:
            resultado = processar_imagem(
                caminho, face_cascade, bag_palavras, erros=2
            )

            if resultado is not None:
                resultados.append(resultado)

        except Exception as erro:
            print(f"[ERRO] {caminho.name}: {erro}")

    if not resultados:
        print("Nenhuma imagem foi processada.")
        return

    df = pd.DataFrame(resultados)

    df.to_csv(
        CSV_SAIDA,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"CSV gerado: {CSV_SAIDA}")
    print(f"Imagens processadas: {len(resultados)}")


if __name__ == "__main__":
    main()