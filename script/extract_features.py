import cv2
import pytesseract
import os
import pandas as pd
import regex as re

def carregar_imagens(DataSet_Filtrado):
    caminhos = []
    for arquivo in os.listdir(DataSet_Filtrado):
        print(f"Lendo imagem {arquivo}")
        caminho = os.path.join(DataSet_Filtrado, arquivo)
        if cv2.imread(caminho) is not None:
            caminhos.append(caminho)
        else:
            print(f"Arquivo {arquivo} não é uma imagem válida, pulando")
    return caminhos

def detectar_rosto(img, face_cascade):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return x, y, w, h
    return None, None, None, None

def detectar_qrcode(img):  
    detector = cv2.QRCodeDetector()

    data, bbox, _ = detector.detectAndDecode(img) 

    if dados:
        return 1
    else:
        return 0 


def extrair_texto(img):
    """Extrai texto da imagem usando OCR."""
    return pytesseract.image_to_string(img)


def contar_palavras(texto):
    """Conta o número de palavras em um texto."""
    return len(texto.split())

def processar_imagem(caminho, face_cascade):
    img = cv2.imread(caminho)
    x, y, w, h = detectar_rosto(img, face_cascade)
    texto = extrair_texto(img)
    quantidade_palavras = contar_palavras(texto)
    
    return {
        "nome_arquivo": os.path.basename(caminho),
        "x_face": x,
        "y_face": y,
        "w_face": w,
        "h_face": h,
        "texto_extraido": texto.strip(),
        "quantidade_palavras": quantidade_palavras
    }

def obter_bag_palavras():
    return [
        "Identidade",i´
        "nome",
        "cpf",
        "Registro Geral"
        "cpf",
        "Sexo",
        "Nacionalidade",
        "DOC. ORIGEM",
        "name",
        "date of birth",
        "Place of birth",
        "Nationality",
        "Sex"
    ]


def gerar_features_bag_palavras(df, bag_palavras):
    """Cria colunas de frequência para cada palavra do bag de palavras."""
    for palavra in bag_palavras:
        df[palavra] = df["texto_extraido"].apply(
            lambda texto: texto.lower().split().count(palavra.lower())
        )
    return df


def gerar_csv(resultados, nome_arquivo):
    """Gera um arquivo CSV com os resultados."""
    df = pd.DataFrame(resultados)
    df.to_csv(nome_arquivo, index=False, encoding="utf-8")
    print(f"CSV gerado: {nome_arquivo}")


def main():
    DataSet_Filtrado = "DataSet_Filtrado"
    csv_saida = "features.csv"
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    caminhos_imagens = carregar_imagens(pasta_imagens)
    resultados = []

    for caminho in caminhos_imagens:
        resultado = processar_imagem(caminho, face_cascade)
        resultados.append(resultado)

    df = pd.DataFrame(resultados)
    df["qrcode_detectado"] = df["nome_arquivo"].apply(lambda x: detectar_qrcode(cv2.imread(x)))
    bag_palavras = obter_bag_palavras()
    df = gerar_features_bag_palavras(df, bag_palavras)

    df.to_csv(csv_saida, index=False, encoding="utf-8")
    print(f"CSV gerado com features: {csv_saida}")


if __name__ == "__main__":
    main()