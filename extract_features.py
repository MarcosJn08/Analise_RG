
import cv2
import pytesseract
import os
import pandas as pd


# configuração

projeto = os.path.dirname(os.path.abspath(__file__))
pasta_dataset = os.path.join(projeto, "DataSet")
csv_saida = os.path.join(projeto, "features.csv")

categorias = {
    "RG_antigo": "rg_antigo",
    "RG_novo": "rg_novo",
    "Invalido_antigo": "invalido_antigo",
    "Invalido_novo": "invalido_novo",
}


# carregar imagens

def carregar_imagens(pasta_imagens):
    """Retorna os caminhos das imagens válidas."""

    caminhos = []

    if not os.path.isdir(pasta_imagens):
        print(f"Pasta não encontrada: {pasta_imagens}")
        return caminhos

    for arquivo in sorted(os.listdir(pasta_imagens)):
        caminho = os.path.join(pasta_imagens, arquivo)

        if not os.path.isfile(caminho):
            continue

        imagem = cv2.imread(caminho)

        if imagem is not None:
            print(f"Lendo imagem: {arquivo}")
            caminhos.append(caminho)
        else:
            print(f"Arquivo inválido, ignorado: {arquivo}")

    return caminhos


# detectar rosto

def detectar_rosto(img, face_cascade):
    """Detecta o primeiro rosto e retorna suas coordenadas."""

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) > 0:
        x, y, w, h = faces[0]
        return x, y, w, h

    return None, None, None, None


# extrair texto com ocr

def extrair_texto(img):
    """Extrai texto da imagem usando o Tesseract."""

    return pytesseract.image_to_string(
        img,
        lang="por"
    )


# contar palavras

def contar_palavras(texto):
    """Conta a quantidade de palavras extraídas."""

    return len(texto.split())


# bag of words

def obter_bag_palavras():
    """Retorna as palavras de interesse."""

    return [
        "república",
        "federativa",
        "brasil",
        "estado",
        "secretaria",
        "segurança",
        "pública",
        "identificação",
        "instituto",
        "carteira",
        "identidade",
        "registro",
        "geral",
        "nome",
        "filiação",
        "nascimento",
        "nacionalidade",
        "naturalidade",
        "validade",
        "cpf",
        "documento",
        "ssp",
        "brasileira",
        "nacional",
        "emissão",
        "sexo",
        "lei",
    ]


def gerar_features_bag_palavras(df, bag_palavras):
    """Cria uma coluna para a frequência de cada palavra."""

    textos = (
        df["texto_extraido"]
        .fillna("")
        .str.lower()
        .str.split()
    )

    for palavra in bag_palavras:
        df[palavra] = textos.apply(
            lambda palavras: palavras.count(palavra.lower())
        )

    return df


# processar imagem

def processar_imagem(caminho, face_cascade, categoria):
    """Extrai as características básicas de uma imagem."""

    img = cv2.imread(caminho)

    if img is None:
        print(f"Não foi possível abrir: {caminho}")
        return None

    altura, largura = img.shape[:2]

    x, y, w, h = detectar_rosto(img, face_cascade)
    texto = extrair_texto(img)

    return {
        "nome_arquivo": os.path.basename(caminho),
        "caminho_imagem": os.path.relpath(
            caminho, projeto
        ),
        "categoria": categoria,
        "largura_imagem": largura,
        "altura_imagem": altura,
        "x_face": x,
        "y_face": y,
        "w_face": w,
        "h_face": h,
        "rosto_detectado": x is not None,
        "texto_extraido": texto.strip(),
        "quantidade_palavras": contar_palavras(texto),
    }


# executar programa

def main():
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError(
            "Não foi possível carregar o detector de rostos."
        )

    resultados = []
    bag_palavras = obter_bag_palavras()

    for pasta, categoria in categorias.items():
        pasta_imagens = os.path.join(
            pasta_dataset, pasta
        )

        print(f"\nProcessando categoria: {categoria}")

        caminhos = carregar_imagens(pasta_imagens)

        for caminho in caminhos:
            try:
                resultado = processar_imagem(
                    caminho,
                    face_cascade,
                    categoria
                )

                if resultado is not None:
                    resultados.append(resultado)

            except Exception as erro:
                print(f"Erro ao processar {caminho}: {erro}")

    if not resultados:
        print("Nenhuma imagem foi processada.")
        return

    df = pd.DataFrame(resultados)

    df = gerar_features_bag_palavras(
        df,
        bag_palavras
    )

    df.to_csv(
        csv_saida,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nExtração concluída!")
    print(f"Imagens processadas: {len(df)}")
    print(f"Arquivo CSV: {csv_saida}")

    print("\nQuantidade por categoria:")
    print(df["categoria"].value_counts())


if __name__ == "__main__":
    main()
