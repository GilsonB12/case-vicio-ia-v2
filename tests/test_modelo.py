import numpy as np
import pandas as pd
import pytest

from apoio_contexto import (
    DROGAS,
    EXTORSAO,
    FURTO,
    MARIA_DA_PENHA,
    PATRIMONIO,
    ROUBO,
    montar,
    montar_maior,
)
from sistema import modelo as modulo_modelo
from sistema.arvore import ArvoreTemas
from sistema.caracteristicas import CARACTERISTICAS
from sistema.modelo import (
    CARACTERISTICAS_FORA,
    COLUNAS_FORA,
    GRADE_C,
    DetectorFora,
    ModeloPares,
    agregar_ramo,
    chaves_fora,
    gatilhos_fora,
    rotulos_pares,
    tabela_questoes,
    temas_positivos,
    termos_encontrados,
)


@pytest.fixture(scope="module")
def arvore():
    return ArvoreTemas.carregar()


@pytest.fixture
def ctx(arvore):
    return montar(arvore)


def tabela_de_pares(n_questoes=30, temas_por_questao=6, semente=0):
    """Pares sintéticos: o primeiro tema de cada questão é o positivo e tem `cos_tema` e `bm25` mais altos."""
    gerador = np.random.default_rng(semente)
    linhas, rotulos = [], []
    for q in range(n_questoes):
        for t in range(temas_por_questao):
            positivo = t == 0
            linha = {"id": f"Q{q:04d}", "tema_id": 1000 + t}
            for nome in CARACTERISTICAS:
                linha[nome] = float(gerador.random())
            linha["cos_tema"] = float(0.7 + 0.2 * gerador.random()) if positivo else float(0.5 * gerador.random())
            linha["bm25"] = float(0.6 + 0.4 * gerador.random()) if positivo else float(0.6 * gerador.random())
            linhas.append(linha)
            rotulos.append(int(positivo))
    return pd.DataFrame(linhas), np.array(rotulos), np.ones(len(linhas))


# --- modelo de pares ---------------------------------------------------------------------------------------


def test_probabilidades_normalizadas_somam_1_por_questao():
    tabela, rotulos, pesos = tabela_de_pares()
    modelo = ModeloPares().treinar(tabela, rotulos, pesos)
    prob = modelo.probabilidades(tabela)
    assert list(prob.columns) == ["id", "tema_id", "s", "p"]
    assert len(prob) == len(tabela)
    assert prob["s"].between(0, 1).all() and prob["p"].between(0, 1).all()
    np.testing.assert_allclose(prob.groupby("id")["p"].sum().to_numpy(), 1.0)
    # p é s dividido pela soma dos s da questão
    primeira = prob[prob["id"] == "Q0000"]
    np.testing.assert_allclose(primeira["p"], primeira["s"] / primeira["s"].sum())
    # o positivo construído é o de maior p em quase todas as questões
    melhores = prob.loc[prob.groupby("id")["p"].idxmax(), "tema_id"]
    assert (melhores == 1000).mean() > 0.9


def test_c_escolhido_na_grade_pela_menor_log_loss_e_determinismo():
    tabela, rotulos, pesos = tabela_de_pares()
    um = ModeloPares().treinar(tabela, rotulos, pesos)
    dois = ModeloPares().treinar(tabela, rotulos, pesos)
    assert set(um.perdas_c) == set(GRADE_C)
    assert um.C == min(um.perdas_c, key=um.perdas_c.get)
    assert um.C == dois.C
    np.testing.assert_array_equal(um.probabilidades(tabela)["s"], dois.probabilidades(tabela)["s"])
    # o escalador fica guardado, ajustado no treino
    np.testing.assert_allclose(um.escalador.mean_, tabela[CARACTERISTICAS].mean().to_numpy())


def test_validacao_interna_agrupa_os_pares_por_questao(monkeypatch):
    divisoes = []
    original = modulo_modelo.GroupKFold

    class Espiao(original):
        def split(self, X, y=None, groups=None):
            for treino, validacao in super().split(X, y, groups):
                divisoes.append((set(np.asarray(groups)[treino]), set(np.asarray(groups)[validacao])))
                yield treino, validacao

    monkeypatch.setattr(modulo_modelo, "GroupKFold", Espiao)
    tabela, rotulos, pesos = tabela_de_pares()
    ModeloPares().treinar(tabela, rotulos, pesos)
    assert len(divisoes) == 3
    for treino, validacao in divisoes:
        assert not treino & validacao


def test_pesos_legiveis_por_caracteristica():
    tabela, rotulos, pesos = tabela_de_pares()
    pesos_modelo = ModeloPares().treinar(tabela, rotulos, pesos).pesos_legiveis()
    assert list(pesos_modelo) == CARACTERISTICAS
    assert all(isinstance(valor, float) for valor in pesos_modelo.values())
    assert pesos_modelo["cos_tema"] > 0 and pesos_modelo["bm25"] > 0
    ruido = [abs(pesos_modelo[c]) for c in CARACTERISTICAS if c not in ("cos_tema", "bm25")]
    assert pesos_modelo["cos_tema"] > max(ruido)


def test_uma_classe_so_vira_constante_e_p_uniforme():
    tabela, _, pesos = tabela_de_pares(n_questoes=4, temas_por_questao=4)
    modelo = ModeloPares().treinar(tabela, np.zeros(len(tabela)), pesos)
    prob = modelo.probabilidades(tabela)
    np.testing.assert_allclose(prob["p"], 0.25)
    assert modelo.pesos_legiveis() == {c: 0.0 for c in CARACTERISTICAS}


def test_tamanhos_diferentes_levantam_erro():
    tabela, rotulos, pesos = tabela_de_pares(n_questoes=3)
    with pytest.raises(ValueError):
        ModeloPares().treinar(tabela, rotulos[:-1], pesos)


def test_probabilidades_de_tabela_vazia(ctx):
    tabela, rotulos, pesos = tabela_de_pares(n_questoes=6)
    modelo = ModeloPares().treinar(tabela, rotulos, pesos)
    vazia = modelo.probabilidades(tabela.iloc[0:0])
    assert vazia.empty and list(vazia.columns) == ["id", "tema_id", "s", "p"]


# --- agregação pelo pai ------------------------------------------------------------------------------------


def test_agregar_ramo_soma_o_pai_e_os_filhos_candidatos(arvore):
    prob = pd.DataFrame([
        ("Q1", PATRIMONIO, 0.2), ("Q1", FURTO, 0.3), ("Q1", ROUBO, 0.1), ("Q1", MARIA_DA_PENHA, 0.4),
        ("Q2", FURTO, 0.7), ("Q2", EXTORSAO, 0.3),
    ], columns=["id", "tema_id", "p"])
    ramos = agregar_ramo(prob, arvore)
    assert list(ramos.columns) == ["id", "tema_id", "p_ramo"]
    por_par = {(i, t): p for i, t, p in ramos.itertuples(index=False)}
    assert por_par == {("Q1", PATRIMONIO): pytest.approx(0.6), ("Q1", MARIA_DA_PENHA): pytest.approx(0.4)}
    # Q2 só tem subtemas: o pai não é candidato e não ganha p_ramo
    assert not (ramos["id"] == "Q2").any()


# --- rótulos -----------------------------------------------------------------------------------------------


def test_positivos_sao_central_e_extras_nunca_pai(arvore):
    ctx = montar_maior(arvore)
    assert ctx.linha_folha("Q0007")["extras_ids"] == [FURTO]
    assert temas_positivos(ctx, "Q0007") == {ROUBO, FURTO}  # central e extra; o pai (5157) não
    assert temas_positivos(ctx, "Q0001") == {FURTO}
    assert temas_positivos(ctx, "Q0036") == set()  # fora
    assert temas_positivos(ctx, "S0001") == {PATRIMONIO}
    assert temas_positivos(ctx, "S0028") == set()  # exemplo fora
    assert temas_positivos(ctx, "C0002") == {FURTO}
    pares = pd.DataFrame({"id": ["Q0007"] * 3, "tema_id": [ROUBO, PATRIMONIO, FURTO]})
    assert rotulos_pares(pares, {"Q0007": temas_positivos(ctx, "Q0007")}).tolist() == [1, 0, 1]


# --- características da questão e detector -----------------------------------------------------------------


def test_termos_encontrados_sem_acento_e_palavra_inteira():
    texto = "Crime MILITAR julgado pela Justiça Militar; o militarismo não conta. Direito Tributário também."
    achados = termos_encontrados(texto, ["militar", "justiça militar", "direito tributário", "civil"])
    assert achados == {"militar": 2, "justiça militar": 1, "direito tributário": 1}


def test_tabela_questoes(ctx):
    vizinhos = {"Q0004": [("X1", None, 0.9), ("X2", FURTO, 0.8), ("X3", None, 0.7), ("X4", DROGAS, 0.1)],
                "S0001": []}
    prob = pd.DataFrame({"id": ["Q0004", "Q0004"], "tema_id": [FURTO, DROGAS], "s": [0.3, 0.1], "p": [0.75, 0.25]})
    pares = pd.DataFrame({"id": ["Q0004", "Q0004"], "tema_id": [FURTO, DROGAS], "cos_tema": [0.2, 0.4],
                          "knn_max": [0.8, 0.1]})
    tabela = tabela_questoes(["Q0004", "S0001"], prob, pares, vizinhos, ctx).set_index("id")
    assert list(tabela.reset_index().columns) == COLUNAS_FORA
    q = tabela.loc["Q0004"]
    assert (q["p_max"], q["s_max"], q["cos_max"], q["knn_max"]) == pytest.approx((0.75, 0.3, 0.4, 0.8))
    assert q["fracao_vizinhos_fora"] == pytest.approx(0.5)
    assert q["peso_fora"] == pytest.approx(1.0)  # diploma:cpm
    assert q["termos_fora"] == pytest.approx(2 / 5)  # "militar" duas vezes
    assert q["disciplina_fora"] == 1.0  # Direito Penal Militar
    s = tabela.loc["S0001"]  # exemplo sem candidato, sem vizinho, sem disciplina
    assert s.tolist() == [0.0] * len(CARACTERISTICAS_FORA)


def test_termos_saturam_em_1(ctx):
    ctx.termos_fora[:] = ["militar", "crime", "deserção", "código penal militar", "penal"]  # 6 ocorrências
    tabela = tabela_questoes(["Q0004"], pd.DataFrame(columns=["id", "tema_id", "s", "p"]),
                             pd.DataFrame(columns=["id", "tema_id", "cos_tema", "knn_max"]), {}, ctx)
    assert tabela.loc[0, "termos_fora"] == 1.0


def test_gatilhos_e_chaves_fora(ctx):
    assert chaves_fora(ctx.chaves("Q0004"), ctx.dicionario) == ["diploma:cpm"]
    assert gatilhos_fora(ctx, "Q0004") == [
        "chaves diploma:cpm", "termos militar", "disciplina de origem Direito Penal Militar"]
    assert gatilhos_fora(ctx, "Q0001") == []


def tabela_fora(n_pos, n_neg, semente=0):
    gerador = np.random.default_rng(semente)
    linhas = []
    for i in range(n_pos + n_neg):
        fora = i < n_pos
        linhas.append({
            "id": f"X{i:03d}",
            "p_max": float(gerador.uniform(0.1, 0.4) if fora else gerador.uniform(0.5, 1.0)),
            "s_max": float(gerador.uniform(0.0, 0.3) if fora else gerador.uniform(0.4, 1.0)),
            "cos_max": float(gerador.uniform(0.1, 0.4) if fora else gerador.uniform(0.4, 0.9)),
            "knn_max": float(gerador.uniform(0.1, 0.5) if fora else gerador.uniform(0.4, 0.9)),
            "fracao_vizinhos_fora": float(gerador.uniform(0.3, 1.0) if fora else gerador.uniform(0.0, 0.3)),
            "peso_fora": float(gerador.integers(0, 3) if fora else 0),
            "termos_fora": float(gerador.uniform(0.2, 1.0) if fora else gerador.uniform(0.0, 0.2)),
            "disciplina_fora": float(fora and gerador.random() < 0.5),
        })
    return pd.DataFrame(linhas, columns=COLUNAS_FORA), np.array([1] * n_pos + [0] * n_neg)


def test_detector_calibrado_separa_fora():
    tabela, rotulos = tabela_fora(12, 60)
    detector = DetectorFora().treinar(tabela, rotulos, np.ones(len(tabela)))
    assert detector.calibracao == "platt"
    p_fora = detector.p_fora(tabela)
    assert list(p_fora.index) == list(tabela["id"]) and p_fora.between(0, 1).all()
    assert p_fora[rotulos == 1].mean() > 0.5 > p_fora[rotulos == 0].mean()
    assert list(detector.pesos_legiveis()) == CARACTERISTICAS_FORA
    assert detector.pesos_legiveis()["termos_fora"] > 0


@pytest.mark.parametrize("n_pos", [1, 2, 5])
def test_detector_com_poucos_positivos_nao_quebra(n_pos):
    tabela, rotulos = tabela_fora(n_pos, 40)
    detector = DetectorFora().treinar(tabela, rotulos, np.ones(len(tabela)))
    assert detector.calibracao == "sem calibração"
    assert detector.p_fora(tabela).between(0, 1).all()


def test_detector_sem_positivos_e_constante():
    tabela, rotulos = tabela_fora(0, 10)
    detector = DetectorFora().treinar(tabela, rotulos, np.ones(len(tabela)))
    assert detector.calibracao == "constante"
    assert (detector.p_fora(tabela) == 0.0).all()


def test_detector_nao_treinado_levanta_erro():
    with pytest.raises(RuntimeError):
        DetectorFora().p_fora(tabela_fora(1, 1)[0])


# --- segundo estágio: opções e confiança --------------------------------------------------------------------


def _tabelas_opcoes():
    """Q0001: top-1 Furto (nível 2) com o pai candidato; Q0002: top-1 Patrimônio (nível 1); Q0003: empate no p."""
    prob = pd.DataFrame([
        ("Q0001", FURTO, 0.30, 0.40), ("Q0001", ROUBO, 0.20, 0.25), ("Q0001", PATRIMONIO, 0.10, 0.15),
        ("Q0001", DROGAS, 0.15, 0.2),
        ("Q0002", PATRIMONIO, 0.5, 0.6), ("Q0002", FURTO, 0.3, 0.4),
        ("Q0003", ROUBO, 0.4, 0.5), ("Q0003", FURTO, 0.4, 0.5),
    ], columns=["id", "tema_id", "s", "p"])
    pares = pd.DataFrame([
        ("Q0001", FURTO, 3, 4, 0.81), ("Q0001", ROUBO, 0, 4, 0.5), ("Q0001", PATRIMONIO, 1, 4, 0.0),
        ("Q0001", DROGAS, 0, 0, 0.2),
        ("Q0002", PATRIMONIO, 2, 3, 0.7), ("Q0002", FURTO, 1, 3, 0.6),
        ("Q0003", ROUBO, 1, 2, 0.4), ("Q0003", FURTO, 1, 2, 0.3),
    ], columns=["id", "tema_id", "votos", "votos_ramo", "knn_folha_max"])
    return prob, pares


def test_tabela_opcoes_tema_e_ramo(arvore, ctx):
    from sistema.modelo import COLUNAS_OPCOES, tabela_opcoes

    prob, pares = _tabelas_opcoes()
    ramos = agregar_ramo(prob, arvore)
    opcoes = tabela_opcoes(prob, ramos, pares, ctx)
    assert list(opcoes.columns) == COLUNAS_OPCOES
    linhas = {(r["id"], r["opcao"]): r for r in opcoes.to_dict("records")}
    assert set(linhas) == {("Q0001", "tema"), ("Q0001", "ramo"), ("Q0002", "tema"), ("Q0003", "tema")}
    tema = linhas[("Q0001", "tema")]
    assert (tema["tema_id"], tema["p"], tema["s"], tema["votos"], tema["votos_ramo"], tema["nivel2_opcao"],
            tema["knn_folha_max"]) == (FURTO, 0.40, 0.30, 3, 4, 1, 0.81)
    ramo = linhas[("Q0001", "ramo")]
    # p_ramo do pai = 0,15 + 0,40 + 0,25; s e knn_folha_max continuam os do top-1; votos são os do par do pai
    assert ramo["tema_id"] == PATRIMONIO and ramo["p"] == pytest.approx(0.80)
    assert (ramo["s"], ramo["votos"], ramo["votos_ramo"], ramo["nivel2_opcao"], ramo["knn_folha_max"]) == \
        (0.30, 1, 4, 0, 0.81)
    assert linhas[("Q0002", "tema")]["nivel2_opcao"] == 0  # top-1 de nível 1: sem opção de ramo
    assert linhas[("Q0003", "tema")]["tema_id"] == min(ROUBO, FURTO)  # empate no p: menor id
    # tem_lei vem das chaves da questão no contexto: só a Q0001 cita lei e artigo
    assert opcoes.groupby("id")["tem_lei"].max().to_dict() == {"Q0001": 1.0, "Q0002": 0.0, "Q0003": 0.0}


def test_tem_lei(ctx):
    from sistema.modelo import tem_lei

    assert tem_lei(ctx, "Q0001")  # art. 155 do CP e Lei 11.340
    assert not tem_lei(ctx, "Q0002")
    assert not tem_lei(ctx, "Q0004")  # só o CPM, que não aponta tema


def opcoes_sinteticas(n=400, semente=0):
    """Opções em que a chance de acerto cresce com os votos; o resto é ruído."""
    from sistema.modelo import CARACTERISTICAS_CONFIANCA

    gerador = np.random.default_rng(semente)
    votos = gerador.integers(0, 5, size=n)
    tabela = pd.DataFrame({nome: gerador.random(n) for nome in CARACTERISTICAS_CONFIANCA})
    tabela["votos"] = votos.astype(float)
    tabela["votos_ramo"] = np.minimum(4, votos + gerador.integers(0, 2, size=n)).astype(float)
    chance = np.array([0.35, 0.5, 0.75, 0.92, 0.98])[votos]
    return tabela, (gerador.random(n) < chance).astype(int)


def test_modelo_confianca_separa_opcoes_com_mais_votos():
    from sistema.modelo import CARACTERISTICAS_CONFIANCA, ModeloConfianca

    tabela, rotulos = opcoes_sinteticas()
    modelo = ModeloConfianca().treinar(tabela, rotulos)
    assert modelo.treinado and modelo.regressao.C == 1.0 and modelo.regressao.class_weight is None
    teste = pd.DataFrame({nome: [0.5] * 5 for nome in CARACTERISTICAS_CONFIANCA})
    teste["votos"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    teste["votos_ramo"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    confianca = modelo.confianca(teste)
    assert np.all(np.diff(confianca) > 0) and confianca[0] < 0.5 < confianca[-1]
    pesos = modelo.pesos_legiveis()
    assert list(pesos) == CARACTERISTICAS_CONFIANCA
    assert pesos["votos"] + pesos["votos_ramo"] > max(abs(pesos[c]) for c in ("p", "s", "tem_lei"))
    descricao = modelo.descrever()
    assert descricao["n_opcoes"] == len(tabela) and descricao["n_certas"] == int(rotulos.sum())
    # determinístico
    np.testing.assert_allclose(ModeloConfianca().treinar(tabela, rotulos).confianca(teste), confianca)


def test_modelo_confianca_com_uma_classe_so_e_constante():
    from sistema.modelo import ModeloConfianca

    tabela, rotulos = opcoes_sinteticas(n=20)
    modelo = ModeloConfianca().treinar(tabela, np.ones(len(tabela), dtype=int))
    assert modelo.constante == 1.0 and modelo.confianca(tabela).tolist() == [1.0] * len(tabela)
    vazio = ModeloConfianca().treinar(tabela.iloc[:0], [])
    assert vazio.constante == 0.0 and vazio.confianca(tabela.iloc[:0]).size == 0
    with pytest.raises(RuntimeError):
        ModeloConfianca().confianca(tabela)
