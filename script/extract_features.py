
import os
import re
import cv2
import pytesseract
import pandas as pd

# configuração
projeto = os.path.dirname(os.path.abspath(__file__))
pasta_dataset = os.path.join(projeto, "DataSet")
csv_saida = os.path.join(projeto, "features.csv")

pastas_categorias = {
    "RG_antigo": ["RG_antigo"],
    "RG_novo": ["RG_novo"],
    "Invalido": ["Invalido", "Invalido_antigo", "Invalido_novo"]
}

extensoes = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# carregar detector de rosto
cascade_path = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError("Não foi possível carregar o detector de rostos.")

# identificar nacionalidade

# verificar palavra nacionalidade
def identificar_nacionalidade(texto):
    if re.search(r"\bnacionalidade\b", texto, re.IGNORECASE):
        return True
    return False


# extrair características da imagem
def extrair_caracteristicas(caminho, categoria):
    imagem = cv2.imread(caminho)

    if imagem is None:
        print(f"Não foi possível abrir: {os.path.basename(caminho)}")
        return None

    altura, largura = imagem.shape[:2]
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) > 0:
        x_face, y_face, w_face, h_face = max(
            faces,
            key=lambda face: face[2] * face[3]
        )
        rosto_detectado = True
    else:
        x_face = y_face = w_face = h_face = 0
        rosto_detectado = False

    texto = pytesseract.image_to_string(
        cinza,
        lang="por"
    )

    quantidade_letras = sum(
        1 for caractere in texto if caractere.isalpha()
    )

    nacionalidade = identificar_nacionalidade(texto)

    return {
        "nome_arquivo": os.path.basename(caminho),
        "categoria": categoria,
        "largura_imagem": largura,
        "altura_imagem": altura,
        "x_face": int(x_face),
        "y_face": int(y_face),
        "w_face": int(w_face),
        "h_face": int(h_face),
        "rosto_detectado": rosto_detectado,
        "nacionalidade": nacionalidade,
        "quantidade_letras": quantidade_letras
    }

# processar imagens
def main():
    resultados = []

    for categoria, nomes_pastas in pastas_categorias.items():
        for nome_pasta in nomes_pastas:
            pasta = os.path.join(pasta_dataset, nome_pasta)

            if not os.path.isdir(pasta):
                continue

            for nome_arquivo in os.listdir(pasta):
                if not nome_arquivo.lower().endswith(extensoes):
                    continue

                caminho = os.path.join(pasta, nome_arquivo)

                dados = extrair_caracteristicas(caminho, categoria)

                if dados is not None:
                    resultados.append(dados)

            print(f"Categoria processada: {nome_pasta}")

    if not resultados:
        print("Nenhuma imagem foi processada. Verifique as pastas do DataSet.")
        return

    df = pd.DataFrame(resultados)
    df.to_csv(csv_saida, index=False, encoding="utf-8-sig")

    print("\nResumo das categorias:")
    print(df["categoria"].value_counts())

    print(f"\nTotal de imagens: {len(df)}")
    print(f"Arquivo salvo em: {csv_saida}")

if __name__ == "__main__":
    main()
