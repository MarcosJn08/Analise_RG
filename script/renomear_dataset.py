
from pathlib import Path

DATASET = Path(
    "/home/marcos/Área de trabalho/Projetos/Analise_RG/Dataset/Dataset_Filtrado"
)

CATEGORIAS = {
    "Invalidos": "Invalidos",
    "RG_antigo": "RG_antigo",
    "rg_antigo_invalido": "RG_antigo_invalido",
    "RG_novo": "RG_novo",
    "rg_novo_invalido": "RG_novo_invalido",
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
        key=lambda arquivo: arquivo.name.lower()
    )

    if not arquivos:
        print(f"[AVISO] Nenhuma imagem encontrada em {nome_pasta}")
        return

    print(f"\nProcessando pasta: {nome_pasta}")

    # Primeira etapa: nomes temporários para evitar conflitos.
    temporarios = []

    for i, arquivo in enumerate(arquivos, start=1):
        temporario = pasta / (
            f"__temp_renomeacao_{i:05d}{arquivo.suffix.lower()}"
        )

        if temporario.exists():
            raise FileExistsError(
                f"Arquivo temporário já existe: {temporario}"
            )

        arquivo.rename(temporario)
        temporarios.append(temporario)

    # Segunda etapa: nomes definitivos.
    for i, temporario in enumerate(temporarios, start=1):
        novo_nome = f"{prefixo}_{i:03d}{temporario.suffix.lower()}"
        destino = pasta / novo_nome

        temporario.rename(destino)
        print(f"[OK] {temporario.name} -> {novo_nome}")

    print(f"[FINALIZADO] {len(arquivos)} imagem(ns) renomeada(s).")


def main():
    if not DATASET.is_dir():
        raise FileNotFoundError(
            f"Diretório DataSet não encontrado: {DATASET}"
        )

    print("Pasta encontrada:", DATASET.is_dir())
    print("Caminho:", DATASET)

    for nome_pasta, prefixo in CATEGORIAS.items():
        renomear_pasta(nome_pasta, prefixo)

    print("\nProcesso de renomeação finalizado.")


if __name__ == "__main__":
    main()
