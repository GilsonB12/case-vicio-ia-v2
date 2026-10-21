"""Extração de leis, diplomas, artigos e súmulas citados no texto limpo de uma questão.

Chaves normalizadas (todas em minúsculas):

- ``lei:11340``: lei pelo número, sem pontos e sem o ano ("Lei nº 11.340/06"); ``dl:2848`` para decreto-lei.
- ``diploma:cp``, ``diploma:cpp``, ``diploma:cf``...: código ou estatuto citado por nome ou sigla. Os
  apelidos de leis especiais ("Lei Maria da Penha") apontam para a chave ``lei:`` correspondente.
- ``art:157:cp``, ``art:5:lei:11340``, ``art:147-a:cp``: artigo, com o diploma decidido pelo contexto
  (menção mais próxima nas 80 letras seguintes ou, sem ela, o diploma dominante do texto); ``art:157``
  quando o texto não cita diploma nenhum. Intervalos ("arts. 240 a 245") geram só as pontas.
- ``sumula:v:11`` (vinculante), ``sumula:stf:11``, ``sumula:stj:7``; ``sumula:11`` sem tribunal explícito.

A saída é um ``Counter`` com a contagem de cada chave no texto.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

# Apelido (como aparece no texto) -> chaves. Siglas em maiúsculas casam com caixa exata; nomes casam
# sem distinção de caixa nem de acento. Os nomes mais longos têm precedência ("Código Penal Militar"
# antes de "Código Penal").
DIPLOMAS: dict[str, tuple[str, ...]] = {
    # códigos e Constituição
    "Código Penal": ("diploma:cp",),
    "Estatuto Repressivo": ("diploma:cp",),
    "CP": ("diploma:cp",),
    "Código de Processo Penal": ("diploma:cpp",),
    "Código Processual Penal": ("diploma:cpp",),
    "Estatuto Processual Penal": ("diploma:cpp",),
    "CPP": ("diploma:cpp",),
    "Código Penal Militar": ("diploma:cpm",),
    "CPM": ("diploma:cpm",),
    "Código de Processo Penal Militar": ("diploma:cppm",),
    "CPPM": ("diploma:cppm",),
    "Constituição Federal": ("diploma:cf",),
    "Constituição da República": ("diploma:cf",),
    "Carta Magna": ("diploma:cf",),
    "CF": ("diploma:cf",),
    "CRFB": ("diploma:cf",),
    "CRF": ("diploma:cf",),
    "Código Tributário Nacional": ("diploma:ctn",),
    "CTN": ("diploma:ctn",),
    "Consolidação das Leis do Trabalho": ("diploma:clt",),
    "CLT": ("diploma:clt",),
    "Código Civil": ("diploma:cc",),
    "Código de Processo Civil": ("diploma:cpc",),
    "CPC": ("diploma:cpc",),
    "Código de Defesa do Consumidor": ("diploma:cdc", "lei:8078"),
    "CDC": ("diploma:cdc", "lei:8078"),
    "Estatuto da Criança e do Adolescente": ("diploma:eca", "lei:8069"),
    "ECA": ("diploma:eca", "lei:8069"),
    "Lei de Execução Penal": ("diploma:lep", "lei:7210"),
    "LEP": ("diploma:lep", "lei:7210"),
    # leis especiais por apelido
    "Lei de Drogas": ("lei:11343",),
    "Lei Antidrogas": ("lei:11343",),
    "Lei de Tóxicos": ("lei:11343",),
    "Lei das Contravenções Penais": ("dl:3688",),
    "Lei de Contravenções Penais": ("dl:3688",),
    "LCP": ("dl:3688",),
    "Estatuto do Desarmamento": ("lei:10826",),
    "Lei Maria da Penha": ("lei:11340",),
    "Lei de Lavagem": ("lei:9613",),
    "Lei de Lavagem de Dinheiro": ("lei:9613",),
    "Lei de Abuso de Autoridade": ("lei:13869", "lei:4898"),
    "Lei do Abuso de Autoridade": ("lei:13869", "lei:4898"),
    "Lei dos Crimes Hediondos": ("lei:8072",),
    "Lei de Crimes Hediondos": ("lei:8072",),
    "Lei de Organizações Criminosas": ("lei:12850",),
    "Lei das Organizações Criminosas": ("lei:12850",),
    "Lei do Crime Organizado": ("lei:12850",),
    "Lei dos Juizados Especiais": ("lei:9099",),
    "Juizados Especiais": ("lei:9099",),
    "Juizados Especiais Criminais": ("lei:9099",),
    "Juizado Especial": ("lei:9099",),
    "Estatuto do Idoso": ("lei:10741",),
    "Estatuto da Pessoa Idosa": ("lei:10741",),
    "Lei de Tortura": ("lei:9455",),
    "Lei da Tortura": ("lei:9455",),
    "Lei de Crimes Ambientais": ("lei:9605",),
    "Lei dos Crimes Ambientais": ("lei:9605",),
    "Código de Trânsito": ("lei:9503",),
    "Código de Trânsito Brasileiro": ("lei:9503",),
    "CTB": ("lei:9503",),
    "Lei de Interceptação": ("lei:9296",),
    "Lei de Interceptação Telefônica": ("lei:9296",),
    "Lei de Racismo": ("lei:7716",),
    "Lei do Racismo": ("lei:7716",),
    "Lei Anticrime": ("lei:13964",),
    "Pacote Anticrime": ("lei:13964",),
    "Código Eleitoral": ("lei:4737",),
    "Lei Antiterrorismo": ("lei:13260",),
    "Lei Antiterror": ("lei:13260",),
}

# Diplomas de outras disciplinas: contam no peso de "fora da taxonomia" (a CF só quando é o único
# diploma do texto; ver ``peso_fora``).
CHAVES_FORA = {"diploma:cpm", "diploma:cppm", "diploma:ctn", "diploma:clt", "diploma:cf"}



def _nomes() -> dict[str, str]:
    nomes: dict[str, str] = {}
    for apelido, chaves in DIPLOMAS.items():
        if not apelido.isupper():
            for chave in chaves:
                nomes.setdefault(chave, apelido)
    return nomes


# Nome legível de cada chave de diploma (o primeiro apelido por extenso que aponta para ela).
NOMES: dict[str, str] = _nomes()

# Número do diploma -> sigla usada nas chaves de artigo: "art. 366 do Decreto-Lei 3.689" vira
# ``art:366:cpp``, igual a "art. 366 do CPP".
_SIGLA_DO_NUMERO = {
    "dl:2848": "cp",
    "dl:3689": "cpp",
    "dl:1001": "cpm",
    "dl:1002": "cppm",
    "lei:5172": "ctn",
    "dl:5452": "clt",
    "lei:10406": "cc",
    "lei:13105": "cpc",
    "lei:8078": "cdc",
    "lei:8069": "eca",
    "lei:7210": "lep",
}

# Decretos-leis citados às vezes como "Lei n.º 2.848/1940": a chave fica ``dl:``.
_DECRETOS_LEI = {"2848", "3688", "3689", "1001", "1002", "5452"}

_JANELA = 80  # letras depois do artigo em que se procura o diploma
_JANELA_CURTA = 15  # menção logo depois do artigo ("do CP") vence a menção logo antes ("CP, art.")

# Apelidos que não valem quando seguidos por estes trechos (no texto dobrado): Juizados Federais são a Lei 10.259.
_NAO_SEGUIDO_DE = {
    "juizados especiais": r"\s+federa",
    "juizados especiais criminais": r"\s+federa",
    "juizado especial": r"\s+federa",
    "lei dos juizados especiais": r"\s+federa",
}


def _dobrar(texto: str) -> str:
    """Minúsculas e sem acento, letra a letra, preservando o comprimento (as posições batem)."""
    saida = []
    for letra in texto:
        base = "".join(c for c in unicodedata.normalize("NFD", letra) if not unicodedata.combining(c)).lower()
        saida.append(base if len(base) == 1 else letra)
    return "".join(saida)


def _compilar_nomes() -> tuple[re.Pattern, dict[str, tuple[str, ...]]]:
    por_nome: dict[str, tuple[str, ...]] = {}
    for apelido, chaves in DIPLOMAS.items():
        if not apelido.isupper():
            por_nome[_dobrar(apelido)] = chaves
    alternativas = []
    for nome in sorted(por_nome, key=len, reverse=True):
        padrao = r"\s+".join(re.escape(palavra) for palavra in nome.split())
        if nome in _NAO_SEGUIDO_DE:
            padrao += f"(?!{_NAO_SEGUIDO_DE[nome]})"
        alternativas.append(f"(?:{padrao})")
    return re.compile(r"(?<!\w)(?:" + "|".join(alternativas) + r")(?!\w)"), por_nome


_RE_NOMES, _POR_NOME = _compilar_nomes()
_SIGLAS = {apelido: chaves for apelido, chaves in DIPLOMAS.items() if apelido.isupper()}
_RE_SIGLAS = re.compile(
    r"(?<!\w)(?:" + "|".join(sorted(_SIGLAS, key=len, reverse=True)) + r")(?![\w-])"
)
# "Constituição" sozinha, com maiúscula: CF, salvo a estadual e os usos comuns da palavra.
_RE_CONSTITUICAO = re.compile(
    r"(?<!\w)Constitui[çc][ãa]o(?!\w)"
    r"(?!\s+(?:Federal|federal|da\s+Rep[úu]blica|Estadual|estadual|do\s+Estado|de\s+mil[íi]cia|simulada))"
)

_N = r"(?:n\s?\.?\s?[º°o]?\s?\.?\s*)"  # nº, n.º, n. , n°, no, n o, nº.
_NUMERO_LEI = r"\d{1,3}(?:\.\d{3})+|\d{3,6}"
_RE_LEI = re.compile(
    r"(?<!\w)(?P<tipo>(?i:decreto[\s-]*lei|leis?)|DL)"
    r"(?:\s+(?i:complementar))?(?:\s+(?i:federal))?\s*"
    rf"(?i:{_N})?(?P<num>{_NUMERO_LEI})(?![\d])(?:\s*/\s*\d{{2,4}}(?!\d))?"
)
# Continuação de lista: "Leis n.º 12.850/2013, n.º 13.869/2019 e n.º 9.613/1998". Exige número com
# ponto de milhar ou com ano, para não confundir com prazos e quantidades.
_RE_LEI_CONTINUA = re.compile(
    r"\s*(?:,|;|(?<!\w)e(?!\w))\s*(?:(?i:a\s+)?(?i:lei)\s*)?(?i:{n})?"
    r"(?P<num>\d{{1,3}}(?:\.\d{{3}})+(?:\s*/\s*\d{{2,4}})?|\d{{3,6}}\s*/\s*\d{{2,4}})(?!\d)".format(n=_N)
)

_RE_ARTIGO = re.compile(r"(?<!\w)(?i:art(?:igo)?(?P<plural>s)?)\.?\s*(?P<num>\d{1,3}(?:\.\d{3})+|\d+)")
# Depois do número: ordinal ("5º", "5 o") e letra ("147-A", "3º-B", "158 - A e seguintes").
_RE_SUFIXO = re.compile(
    r"(?:\s?[º°]|\s?o(?!\w))?"
    r"(?:-(?P<letra>[A-Z])(?![\w])"
    r"|\s+-\s+(?P<letra_espacada>[A-Z])(?=\s+e\s+seguintes|\s+a\s+\d|\s*[,;)]|\s*$))?"
)
# O que pode vir entre os números de uma lista de artigos sem encerrá-la.
_RE_CONTINUA = re.compile(
    r"""\s*(?:
      (?P<sep>,|;|(?<!\w)(?:e|a|ou|até)(?!\w))
    | (?P<paragrafo>§§?\s*\d+\.?[º°o]?(?:-[A-Z])?(?:\s*(?:,|e)\s*\d+\.?[º°o]?)*)
    | (?P<paragrafo_extenso>(?i:par[áa]grafos?)\s+(?:(?i:[úu]nico)|\d+\.?[º°o]?))
    | (?P<inciso>(?i:incisos?)\s+[IVXLCDM]+(?:\s*(?:,|e)\s*[IVXLCDM]+)*(?!\w))
    | (?P<romano>[IVXLCDM]+(?!\w))
    | (?P<alinea>(?i:al[íi]neas?)\s+["“'‘]?[a-z]["”'’]?(?!\w))
    | (?P<caput>(?i:caput)(?!\w))
    | (?P<num>\d{1,3}(?:\.\d{3})+|\d+)
    )""",
    re.X,
)
# "do mesmo diploma", "da mesma lei", "do referido Estatuto": o diploma é o da menção anterior.
_RE_REFERENCIA = re.compile(r"(?<!\w)(?:mesm[oa]|referid[oa]|citad[oa])\s+(?:diploma|lei|codigo|estatuto)(?!\w)")

_RE_SUMULA = re.compile(
    r"(?<!\w)(?i:s[úu]mula)\s+(?P<vinculante>(?i:vinculante)\s+)?(?i:{n})?(?P<num>\d+)(?!\d)".format(n=_N)
)
_RE_TRIBUNAL = re.compile(
    r"\s*,?\s*(?:/\s*|(?i:d[oa])\s+)"
    r"(?P<tribunal>STF|STJ|TST|TSE|STM"
    r"|(?i:supremo\s+tribunal\s+federal)|(?i:superior\s+tribunal\s+de\s+justi[çc]a)"
    r"|(?i:superior\s+tribunal\s+militar)|(?i:tribunal\s+superior\s+do\s+trabalho)"
    r"|(?i:tribunal\s+superior\s+eleitoral))"
)
_TRIBUNAIS = {
    "supremo tribunal federal": "stf",
    "superior tribunal de justica": "stj",
    "superior tribunal militar": "stm",
    "tribunal superior do trabalho": "tst",
    "tribunal superior eleitoral": "tse",
}


@dataclass(frozen=True)
class _Mencao:
    """Uma citação de diploma ou de lei no texto, com as chaves que ela gera."""

    inicio: int
    fim: int
    chaves: tuple[str, ...]

    @property
    def sufixo(self) -> str:
        return sufixo_artigo(self.chaves[0])


def normalizar_lei(numero: str | int) -> str:
    """Número de lei sem pontos e sem ano: "Lei nº 8.072/90" -> "8072"."""
    texto = str(numero)
    achado = re.search(r"\d{1,3}(?:\.\d{3})+|\d+", texto)
    if not achado:
        raise ValueError(f"número de lei não encontrado em {numero!r}")
    return achado.group(0).replace(".", "").lstrip("0") or "0"


def sufixo_artigo(chave: str) -> str:
    """Parte da chave de artigo que identifica o diploma: diploma:cp -> cp; dl:2848 -> cp; lei:9613 -> lei:9613."""
    if chave.startswith("diploma:"):
        return chave.split(":", 1)[1]
    return _SIGLA_DO_NUMERO.get(chave, chave)


def _chave_lei(tipo: str, numero: str) -> str:
    numero = normalizar_lei(numero)
    decreto = tipo.lower().startswith("decreto") or tipo == "DL" or numero in _DECRETOS_LEI
    return f"{'dl' if decreto else 'lei'}:{numero}"


def _mencoes(texto: str, dobrado: str) -> list[_Mencao]:
    mencoes: list[_Mencao] = []
    consumido = 0  # fim da última lista de leis: o que já entrou como continuação não conta de novo
    for achado in _RE_LEI.finditer(texto):
        if achado.start() < consumido:
            continue
        tipo = achado.group("tipo")
        mencoes.append(_Mencao(achado.start(), achado.end(), (_chave_lei(tipo, achado.group("num")),)))
        posicao = achado.end()
        while (continua := _RE_LEI_CONTINUA.match(texto, posicao)) is not None:
            mencoes.append(
                _Mencao(continua.start("num"), continua.end(), (_chave_lei(tipo, continua.group("num")),))
            )
            posicao = continua.end()
        consumido = posicao
    for achado in _RE_NOMES.finditer(dobrado):
        nome = re.sub(r"\s+", " ", achado.group(0))
        mencoes.append(_Mencao(achado.start(), achado.end(), _POR_NOME[nome]))
    for achado in _RE_SIGLAS.finditer(texto):
        mencoes.append(_Mencao(achado.start(), achado.end(), _SIGLAS[achado.group(0)]))
    for achado in _RE_CONSTITUICAO.finditer(texto):
        mencoes.append(_Mencao(achado.start(), achado.end(), ("diploma:cf",)))
    return sorted(mencoes, key=lambda mencao: (mencao.inicio, mencao.fim))


def _dominante(mencoes: list[_Mencao]) -> str | None:
    """Diploma mais citado no texto; empate fica com o que aparece primeiro."""
    if not mencoes:
        return None
    return Counter(mencao.sufixo for mencao in mencoes).most_common(1)[0][0]


def _numero_artigo(numero: str, sufixo: re.Match) -> tuple[str, int]:
    letra = sufixo.group("letra") or sufixo.group("letra_espacada")
    numero = numero.replace(".", "")
    return (f"{numero}-{letra.lower()}" if letra else numero), sufixo.end()


def _artigos(texto: str) -> Iterable[tuple[int, int, list[str]]]:
    """(início, fim, números) de cada citação de artigo; listas e intervalos só no plural."""
    for achado in _RE_ARTIGO.finditer(texto):
        numero, fim = _numero_artigo(achado.group("num"), _RE_SUFIXO.match(texto, achado.end()))
        numeros = [numero]
        if achado.group("plural"):
            posicao, depois_de_separador = fim, False
            while (parte := _RE_CONTINUA.match(texto, posicao)) is not None and parte.end() > posicao:
                if parte.group("num") is not None:
                    if not depois_de_separador:
                        break
                    numero, posicao = _numero_artigo(parte.group("num"), _RE_SUFIXO.match(texto, parte.end()))
                    numeros.append(numero)
                    fim, depois_de_separador = posicao, False
                    continue
                depois_de_separador = parte.group("sep") is not None
                posicao = parte.end()
        yield achado.start(), fim, numeros


def _diploma_do_artigo(
    inicio: int, fim: int, mencoes: list[_Mencao], dobrado: str, dominante: str | None
) -> str | None:
    seguintes = [mencao for mencao in mencoes if fim <= mencao.inicio < fim + _JANELA]
    proxima = seguintes[0] if seguintes else None
    if proxima is not None and proxima.inicio < fim + _JANELA_CURTA:
        return proxima.sufixo
    anteriores = [mencao for mencao in mencoes if mencao.fim <= inicio]
    if anteriores and re.fullmatch(r"\s*,\s*", dobrado[anteriores[-1].fim:inicio]):
        return anteriores[-1].sufixo  # "(CP, art. 159)"
    limite = proxima.inicio if proxima is not None else fim + _JANELA
    if anteriores and _RE_REFERENCIA.search(dobrado, fim, limite):
        return anteriores[-1].sufixo  # "do Art. 35 da mesma lei"
    if proxima is not None:
        return proxima.sufixo
    return dominante


def _sumulas(texto: str) -> Iterable[str]:
    for achado in _RE_SUMULA.finditer(texto):
        numero = achado.group("num").lstrip("0") or "0"
        if achado.group("vinculante"):
            yield f"sumula:v:{numero}"
            continue
        tribunal = _RE_TRIBUNAL.match(texto, achado.end())
        if tribunal is None:
            yield f"sumula:{numero}"
            continue
        nome = re.sub(r"\s+", " ", _dobrar(tribunal.group("tribunal")))
        yield f"sumula:{_TRIBUNAIS.get(nome, nome)}:{numero}"


def extrair_chaves(texto: str | None, diploma_padrao: str | None = None) -> Counter[str]:
    """Chaves de leis, diplomas, artigos e súmulas do texto, com contagem.

    ``diploma_padrao`` (por exemplo ``"cp"`` ou ``"lei:11343"``) vale para artigos sem diploma quando o
    texto não cita diploma nenhum; serve ao dicionário, que extrai artigos de listas curtas dos temas.
    """
    chaves: Counter[str] = Counter()
    if not texto:
        return chaves
    texto = unicodedata.normalize("NFC", texto).replace("\xa0", " ")
    texto = re.sub(r"(?:\\[nt]|\s)+", " ", texto)
    dobrado = _dobrar(texto)
    mencoes = _mencoes(texto, dobrado)
    for mencao in mencoes:
        chaves.update(mencao.chaves)
    dominante = _dominante(mencoes) or diploma_padrao
    for inicio, fim, numeros in _artigos(texto):
        diploma = _diploma_do_artigo(inicio, fim, mencoes, dobrado, dominante)
        for numero in numeros:
            chaves[f"art:{numero}:{diploma}" if diploma else f"art:{numero}"] += 1
    chaves.update(_sumulas(texto))
    return chaves


def peso_fora(chaves: Counter[str], fora: Iterable[str] = CHAVES_FORA) -> float:
    """Soma das contagens das chaves de outras disciplinas.

    A CF só conta quando é o único diploma do texto: questão penal cita a Constituição o tempo todo.
    """
    fora = set(fora)
    outro_diploma = any(
        n > 0 and chave != "diploma:cf" and chave.startswith(("diploma:", "lei:", "dl:"))
        for chave, n in chaves.items()
    )
    total = 0.0
    for chave, n in chaves.items():
        if chave in fora and n > 0 and not (chave == "diploma:cf" and outro_diploma):
            total += n
    return total
