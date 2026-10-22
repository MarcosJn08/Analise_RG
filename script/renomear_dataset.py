
from pathlib import Path

PROJETO = Path(__file__).resolve().parent.parent
DATASET = PROJETO / "DataSet"

CATEGORIAS = {
    "RG_antigo": "RG_antigo",
    "RG_novo": "RG_novo",
    "Invalido_antigo": "Invalido_antigo",
    "Invalido_novo": "Invalido_novo",
}

EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def renomear_pasta(nome_pasta, prefixo):
    pasta = DATASET / nome_pasta

    if not pasta.is_dir():
        print(f"[ERRO] Pasta não encontrada: {pasta}")
        return

    arquivos = sorted(
        (
            arquivo
            for arquivo in pasta.iterdir()
            if arquivo.is_file()
            and arquivo.suffix.lower() in EXTENSOES
        ),
        key=lambda arquivo: arquivo.name.lower(),
    )

    if not arquivos:
        print(f"[AVISO] Nenhuma imagem em {nome_pasta}")
        return

    # Primeiro, usa nomes temporários para evitar conflitos.
    temporarios = []

    for i, arquivo in enumerate(arquivos, start=1):
        temporario = pasta / (
            f"__temporario_renomeacao_{i:05d}"
            f"{arquivo.suffix.lower()}"
        )

        if temporario.exists():
            raise FileExistsError(
                f"Arquivo temporário já existe: {temporario}"
            )

        arquivo.rename(temporario)
        temporarios.append(temporario)

    # Depois, aplica os nomes definitivos.
    for i, temporario in enumerate(temporarios, start=1):
        novo_nome = f"{prefixo}_{i:03d}{temporario.suffix}"
        destino = pasta / novo_nome

        temporario.rename(destino)
        print(f"{nome_pasta}: {novo_nome}")

    print(f"[OK] {nome_pasta}: {len(arquivos)} imagem(ns)\n")


def main():
    if not DATASET.is_dir():
        raise FileNotFoundError(
            f"Diretório DataSet não encontrado: {DATASET}"
        )

    print(f"Projeto: {PROJETO}")
    print(f"Dataset: {DATASET}\n")

    for nome_pasta, prefixo in CATEGORIAS.items():
        renomear_pasta(nome_pasta, prefixo)

    print("Processo de renomeação finalizado.")


if __name__ == "__main__":
    main()
