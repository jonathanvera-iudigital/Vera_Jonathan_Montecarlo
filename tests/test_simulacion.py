import random

import numpy as np
import pytest

import simulacion_01 as sim


# ------------------------------ puntuación ------------------------------
@pytest.mark.parametrize("categoria,dados,esperado", [
    ("ones", [1, 1, 1, 4, 5], 3), ("sixes", [6, 6, 2, 3, 4], 12), ("fives", [1, 2, 3, 4, 6], 0),
    ("threeOfKind", [3, 3, 3, 1, 2], 12), ("threeOfKind", [3, 3, 1, 1, 2], 0),
    ("fourOfKind", [2, 2, 2, 2, 6], 14), ("fourOfKind", [2, 2, 2, 6, 6], 0),
    ("fullHouse", [2, 2, 3, 3, 3], 25), ("fullHouse", [4, 4, 4, 4, 4], 0),
    ("smallStraight", [1, 2, 3, 4, 6], 30), ("smallStraight", [1, 2, 2, 3, 4], 30),
    ("smallStraight", [1, 2, 3, 5, 6], 0),
    ("largeStraight", [2, 3, 4, 5, 6], 40), ("largeStraight", [1, 2, 3, 4, 6], 0),
    ("yahtzee", [5, 5, 5, 5, 5], 50), ("yahtzee", [5, 5, 5, 5, 4], 0),
    ("chance", [1, 2, 3, 4, 6], 16),
])
def test_score_hand(categoria, dados, esperado):
    assert sim.score_hand(categoria, dados) == esperado


def test_best_category_score_respeta_disponibles():
    assert sim.best_category_score([6, 6, 6, 6, 6], sim.CATEGORIES_ORDER) == ("yahtzee", 50)
    assert sim.best_category_score([6, 6, 6, 6, 6], ["ones", "chance"]) == ("chance", 30)


def test_bono_superior():
    j = sim.PlayerState("x")
    for cat, pts in zip(sim.CATEGORIES_ORDER[:6], [3, 6, 9, 12, 15, 18]):   # 63 exactos
        j.set_score(cat, pts)
    assert j.upper_subtotal == 63 and j.bonus == 35 and j.total == 98
    j.set_score("ones", 0)
    assert j.bonus == 0


# ------------------------------- dados ----------------------------------
def test_roll_dice_devuelve_solo_indices_lanzados():
    random.seed(1)
    valores = [6, 6, 6, 6, 6]
    rolled = sim.roll_dice(valores, [True, True, False, False, True])
    assert rolled == [2, 3]
    assert valores[0] == valores[1] == valores[4] == 6


def test_dado_uniforme():
    random.seed(7)
    n = 300_000
    conteos = np.bincount([sim.roll_single_die() - 1 for _ in range(n)], minlength=6)
    chi2 = ((conteos - n / 6) ** 2 / (n / 6)).sum()
    assert chi2 < 20.5   # valor crítico al 0,1 % con 5 grados de libertad


# ---------------------------- exactos y Montecarlo ----------------------
def test_distribucion_inicial_suma_uno_y_yahtzee_en_una_tirada():
    p = sim.initial_distribution()
    assert p.sum() == pytest.approx(1.0)
    assert sim.exact_category_probability(p, "yahtzee") == pytest.approx(6 / 7776)


def test_probabilidad_exacta_de_yahtzee_en_un_turno_es_4_6_por_ciento():
    dist = sim.exact_turn_distribution(sim.choose_keep_mask_naive)
    assert dist.sum() == pytest.approx(1.0)
    assert sim.exact_category_probability(dist, "yahtzee") == pytest.approx(0.0460, abs=0.0005)


def test_ev_exacto_de_una_mascara_conocida():
    # Conservar cuatro 6 y relanzar uno, con solo "yahtzee" disponible: P = 1/6, EV = 50/6
    assert sim.exact_ev_after_keeps([6, 6, 6, 6, 1], [True] * 4 + [False], ["yahtzee"]) == \
        pytest.approx(50 / 6)


def test_montecarlo_converge_al_valor_exacto():
    random.seed(3)
    vals, cats, mask = [3, 3, 5, 2, 6], ["threes", "threeOfKind", "fullHouse", "chance"], \
        [True, True, False, False, False]
    exacto = sim.exact_ev_after_keeps(vals, mask, cats)
    est = sim.simulate_expected_value_after_keeps(vals, mask, 1, cats, 100_000)
    assert est == pytest.approx(exacto, abs=0.15)


def test_error_montecarlo_baja_con_n():
    random.seed(11)
    vals, cats, mask = [3, 3, 5, 2, 6], ["threes", "threeOfKind", "fullHouse", "chance"], \
        [True, True, False, False, False]
    exacto = sim.exact_ev_after_keeps(vals, mask, cats)

    def error(n):
        return np.mean([abs(sim.simulate_expected_value_after_keeps(vals, mask, 1, cats, n) - exacto)
                        for _ in range(60)])
    assert error(5_000) < error(50) / 3


# ------------------------- correcciones de la guía ----------------------
def test_ultima_tirada_ya_no_relanza_todo():
    """Fallo 1 de la guía: en la 3.ª tirada siempre se relanzaban los 5 dados."""
    random.seed(0)
    todas = sim.CATEGORIES_ORDER
    assert all(sim.choose_keep_mask_montecarlo([6, 6, 6, 6, 6], 1, todas, sims_per_mask=50))
    assert all(sim.choose_keep_mask_exact([6, 6, 6, 6, 6], 1, todas))
    mask = sim.choose_keep_mask_exact([6, 6, 6, 6, 1], 1, todas)
    assert mask[:4] == [True] * 4


def test_sims_per_mask_se_lee_al_llamar():
    """Fallo 2 de la guía: el valor por defecto se fijaba al definir la función."""
    random.seed(0)
    original = sim.SIMS_PER_MASK
    try:
        sim.SIMS_PER_MASK = 1
        antes = sim.choose_keep_mask_montecarlo([1, 2, 3, 4, 6], 2, sim.CATEGORIES_ORDER)
        assert antes is not None   # con 1 sim por máscara corre rápido y no usa el 300 original
    finally:
        sim.SIMS_PER_MASK = original


def test_estadistica_cuenta_solo_dados_lanzados():
    random.seed(5)
    estado = sim.play_game(strategies=(sim.choose_keep_mask_naive,) * 2, silent=True)
    s = estado.stats
    assert sum(s.face_hist.values()) == s.dice_rolled
    assert s.dice_rolled < sum(s.face_hist_all.values())   # el conteo de la guía inflaba
    assert s.turns == 26


# ------------------------------- partida --------------------------------
def test_partida_completa_y_reproducible():
    def jugar(seed):
        random.seed(seed)
        e = sim.montecarlo_strategy(20)
        return sim.play_game(strategies=(e, e), silent=True)
    a, b = jugar(9), jugar(9)
    assert [p.total for p in a.players] == [p.total for p in b.players]
    for p in a.players:
        assert p.available_categories() == []
    assert sim.winner_index(a) in (-1, 0, 1)


def test_turno_que_se_planta_hace_una_sola_tirada():
    random.seed(0)
    estado = sim.GameState(players=[sim.PlayerState("a"), sim.PlayerState("b")])
    sim.play_one_turn(estado, verbose=False, strategy=lambda v, r, d: [True] * 5)
    assert estado.stats.total_rolls == 1
