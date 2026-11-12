import base64
import gc
import io
import os
import regex
import requests
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR

# Configurações para economizar CPU e RAM no plano gratuito do Render (512MB RAM / 0.1 vCPU)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 960px é a resolução ideal para RG: economiza mais de 60% de RAM frente a 1600px
# e roda mais que o dobro de rápido no RapidOCR, mantendo legibilidade total dos caracteres.
LADO_MAXIMO_PADRAO = 960
ERROS_FUZZY_PADRAO = 2

# Sessão HTTP reutilizável com keep-alive para evitar novo handshake SSL a cada URL
_sessao_http = requests.Session()
_sessao_http.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})


def carregar_imagem(fonte: str, lado_maximo: int = LADO_MAXIMO_PADRAO) -> np.ndarray:
    """
    Carrega imagem a partir de uma URL ou string Base64.
    Ajusta orientação EXIF, converte para RGB e redimensiona mantendo proporção.
    """
    fonte = fonte.strip()

    if fonte.startswith("http://") or fonte.startswith("https://"):
        resposta = _sessao_http.get(fonte, timeout=12)
        resposta.raise_for_status()
        dados = resposta.content
    else:
        # Se vier com cabeçalho data:image/png;base64,...
        if fonte.startswith("data:"):
            fonte = fonte.split(",", 1)[1]
        fonte = regex.sub(r"\s+", "", fonte)
        fonte += "=" * (-len(fonte) % 4)
        dados = base64.b64decode(fonte)

    with Image.open(io.BytesIO(dados)) as img_raw:
        # Corrige rotação automática por metadados EXIF
        imagem = ImageOps.exif_transpose(img_raw).convert("RGB")
        # Reduz tamanho para não estourar os 512MB de RAM do Render
        imagem.thumbnail((lado_maximo, lado_maximo), Image.Resampling.BILINEAR)
        array_rgb = np.array(imagem)

    return array_rgb


class ExtratorFeatures:
    """
    Extrator de características textuais baseado em RapidOCR e casamento difuso (fuzzy regex).
    Otimizado para ambientes com recursos limitados de CPU e RAM.
    """

    def __init__(
        self,
        colunas: list[str],
        erros_fuzzy: int = ERROS_FUZZY_PADRAO,
        lado_maximo: int = LADO_MAXIMO_PADRAO
    ):
        self.colunas = list(colunas)
        self.erros_fuzzy = erros_fuzzy
        self.lado_maximo = lado_maximo

        # RapidOCR com limite interno de detecção para economizar memória
        self.ocr = RapidOCR(det_limit_side_len=lado_maximo)

        # Pré-compila todas as expressões regulares na inicialização
        # Isso evita compilar 26 regexes a cada requisição
        self.padroes_regex = {
            palavra: regex.compile(
                rf"(?e)(?<!\w)(?:{r'\s+'.join(regex.escape(parte) for parte in palavra.split())}){{e<={self.erros_fuzzy}}}(?!\w)",
                flags=regex.IGNORECASE
            )
            for palavra in self.colunas
        }

    def ler_texto(self, imagem_rgb: np.ndarray) -> str:
        """Executa o RapidOCR e devolve todas as linhas de texto encontradas."""
        # RapidOCR espera BGR
        bgr = np.ascontiguousarray(imagem_rgb[:, :, ::-1])
        resultado, _ = self.ocr(bgr)

        # Ajuda o garbage collector a liberar a cópia BGR
        del bgr

        if not resultado:
            return ""
        return "\n".join(linha[1] for linha in resultado)

    def extrair(self, imagem_rgb: np.ndarray) -> tuple[dict[str, int], str]:
        """
        Extrai as features binárias correspondentes às colunas esperadas pelo modelo.
        Aplica Early Exit e Fast-Path exato antes do fuzzy matching para economizar CPU.
        """
        texto = self.ler_texto(imagem_rgb)

        # Early exit: se a imagem não contém texto, não gasta CPU processando regexes
        if not texto.strip():
            return {palavra: 0 for palavra in self.colunas}, ""

        texto_lower = texto.lower()
        features = {}

        for palavra in self.colunas:
            palavra_lower = palavra.lower()
            # Fast-path: checagem exata ultrarrápida em C (Boyer-Moore)
            if palavra_lower in texto_lower:
                features[palavra] = 1
            else:
                # Fallback: apenas se falhar a busca exata, usa o regex difuso pré-compilado
                padrao = self.padroes_regex[palavra]
                features[palavra] = 1 if padrao.search(texto_lower) is not None else 0

        return features, texto

    def extrair_dataframe(self, imagem_rgb: np.ndarray) -> tuple[pd.DataFrame, str]:
        """Conveniência: retorna as features já empacotadas em DataFrame e o texto lido."""
        features, texto = self.extrair(imagem_rgb)
        df = pd.DataFrame([features], columns=self.colunas)
        return df, texto
