# -*- coding: utf-8 -*-
"""Yahtzee con el método de Montecarlo — Actividad didáctica 2-M1.

Este es el archivo de trabajo. Parte de la guía base `simulacion_01` (copia intacta en
original/simulacion_01_original.py) y conserva su estructura, nombres y orden. Cada cambio está
marcado en el código con una de estas etiquetas:

    # MODIFICA: cambio sobre código que ya existía en la guía (corrige o mejora algo)
    # CREA:     código nuevo que no existía en la guía

Uso:  python simulacion_01.py <comando> [--rapido] [--seed N] [--salida carpeta]
Comandos: demo, comparativa, dados, probabilidades, convergencia, decisiones, partidas,
          diagrama, todo.
"""
# MODIFICA: imports adicionales (itertools, lru_cache, argparse...) para los aportes nuevos.
import argparse
import itertools
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ===============================
# Parámetros Montecarlo
# ===============================
SIMS_PER_MASK = 300  # nº de simulaciones por máscara de bloqueo (32 máx). Ajusta si quieres más/menos precisión.
SEED = 42            # fija semilla para reproducibilidad; pon None para aleatorio
OUT_DIR = "resultados"  # CREA: carpeta donde se guardan CSV y figuras

if SEED is not None:
    random.seed(SEED)

# ===============================
# Utilidades de dados
# ===============================
def roll_single_die() -> int:
    """Devuelve un entero uniforme en [1,6]."""
    return random.randint(1, 6)

def roll_dice(values: List[int], locked: List[bool]) -> List[int]:
    """Lanza dados no bloqueados, actualizando 'values' in place."""
    # MODIFICA: ahora devuelve los índices realmente lanzados, para que la estadística
    # cuente solo los dados nuevos y no los bloqueados.
    rolled = []
    for i, is_locked in enumerate(locked):
        if not is_locked:
            values[i] = roll_single_die()
            rolled.append(i)
    return rolled

def has_sequence(values: List[int], needed_len: int) -> bool:
    """¿Existe una secuencia consecutiva de longitud 'needed_len'? (ignora duplicados)."""
    uniq = sorted(set(values))
    if not uniq:
        return False
    longest = curr = 1
    for i in range(1, len(uniq)):
        if uniq[i] == uniq[i-1] + 1:
            curr += 1
            longest = max(longest, curr)
        else:
            curr = 1
    return longest >= needed_len

# ===============================
# Puntuación
# ===============================
CATEGORIES_ORDER = [
    "ones","twos","threes","fours","fives","sixes",
    "threeOfKind","fourOfKind","fullHouse","smallStraight",
    "largeStraight","yahtzee","chance"
]
UPPER_CATEGORIES = CATEGORIES_ORDER[:6]   # CREA: sección superior (para el bono)
BONUS_THRESHOLD, BONUS_POINTS = 63, 35    # CREA: bono clásico de Yahtzee

# MODIFICA: la puntuación se cachea por mano ordenada. Solo hay 252 manos distintas, y el
# cálculo se repetía millones de veces dentro de la simulación. El cuerpo es el de la guía.
@lru_cache(maxsize=None)
def _score_sorted(category: str, dice: tuple) -> int:
    counts = Counter(dice)
    freqs = list(counts.values())
    total = sum(dice)
    if category in ["ones","twos","threes","fours","fives","sixes"]:
        num = {"ones":1, "twos":2, "threes":3, "fours":4, "fives":5, "sixes":6}[category]
        return counts.get(num, 0) * num
    if category == "threeOfKind":
        return total if any(f >= 3 for f in freqs) else 0
    if category == "fourOfKind":
        return total if any(f >= 4 for f in freqs) else 0
    if category == "fullHouse":
        return 25 if (2 in freqs and 3 in freqs) else 0
    if category == "smallStraight":
        return 30 if has_sequence(dice, 4) else 0
    if category == "largeStraight":
        return 40 if has_sequence(dice, 5) else 0
    if category == "yahtzee":
        return 50 if 5 in freqs else 0
    if category == "chance":
        return total
    return 0

def score_hand(category: str, dice: List[int]) -> int:
    return _score_sorted(category, tuple(sorted(dice)))

@lru_cache(maxsize=500_000)
def _best_cached(dice: tuple, available: tuple) -> Tuple[str, int]:
    best_cat, best = None, -1
    for cat in available:
        sc = _score_sorted(cat, dice)
        if sc > best:
            best = sc
            best_cat = cat
    return best_cat, best

def best_category_score(dice: List[int], available: List[str]) -> Tuple[str, int]:
    """Devuelve (mejor_categoria, puntuación) para los dados dados."""
    return _best_cached(tuple(sorted(dice)), tuple(available))

# ===============================
# Estructuras de juego
# ===============================
@dataclass
class PlayerState:
    name: str
    scores: Dict[str, Optional[int]] = field(default_factory=lambda: {c: None for c in CATEGORIES_ORDER})
    total: int = 0

    def available_categories(self) -> List[str]:
        return [c for c,v in self.scores.items() if v is None]

    # CREA: subtotal de la sección superior y bono de 35 puntos si llega a 63
    @property
    def upper_subtotal(self) -> int:
        return sum(self.scores[c] or 0 for c in UPPER_CATEGORIES)

    @property
    def bonus(self) -> int:
        return BONUS_POINTS if self.upper_subtotal >= BONUS_THRESHOLD else 0

    def set_score(self, category: str, points: int) -> None:
        self.scores[category] = points
        # MODIFICA: el total incluye el bono de la sección superior
        self.total = sum(v for v in self.scores.values() if v is not None) + self.bonus

@dataclass
class GameStats:
    # MODIFICA: en la guía `total_rolls` y `face_hist` contaban los 5 dados en CADA tirada,
    # incluidos los bloqueados que no se relanzaban: muestras repetidas y no independientes.
    total_rolls: int = 0      # número de tiradas
    dice_rolled: int = 0      # dados realmente lanzados
    face_hist: Counter = field(default_factory=Counter)      # solo dados lanzados (correcto)
    face_hist_all: Counter = field(default_factory=Counter)  # conteo de la guía (con bloqueados)
    # CREA: manos finales de cada turno, para estimar probabilidades de combinaciones
    turns: int = 0
    yahtzees: int = 0
    large_straights: int = 0
    full_houses: int = 0

    def update_from_roll(self, values: List[int], rolled_idx: Optional[List[int]] = None) -> None:
        if rolled_idx is None:
            rolled_idx = range(len(values))
        self.total_rolls += 1
        self.dice_rolled += len(rolled_idx)
        self.face_hist.update(values[i] for i in rolled_idx)
        self.face_hist_all.update(values)

    def update_final_hand(self, values: List[int]) -> None:   # CREA
        self.turns += 1
        self.yahtzees += score_hand("yahtzee", values) > 0
        self.large_straights += score_hand("largeStraight", values) > 0
        self.full_houses += score_hand("fullHouse", values) > 0

    def merge(self, other: "GameStats") -> None:               # CREA
        self.total_rolls += other.total_rolls
        self.dice_rolled += other.dice_rolled
        self.face_hist.update(other.face_hist)
        self.face_hist_all.update(other.face_hist_all)
        self.turns += other.turns
        self.yahtzees += other.yahtzees
        self.large_straights += other.large_straights
        self.full_houses += other.full_houses

@dataclass
class GameState:
    players: List[PlayerState]
    current_player_idx: int = 0
    turn_in_round: int = 0  # 0..12 (13 turnos)
    dice_values: List[int] = field(default_factory=lambda: [0]*5)
    locked: List[bool] = field(default_factory=lambda: [False]*5)
    rolls_left: int = 3
    stats: GameStats = field(default_factory=GameStats)

    def reset_turn(self):
        self.dice_values = [0]*5
        self.locked = [False]*5
        self.rolls_left = 3

    @property
    def current_player(self) -> PlayerState:
        return self.players[self.current_player_idx]

    def advance_player(self):
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        if self.current_player_idx == 0:
            self.turn_in_round += 1

# ===============================
# Montecarlo para decidir bloqueos
# ===============================
def all_keep_masks() -> List[List[bool]]:
    """Genera las 32 máscaras posibles de bloqueo para 5 dados."""
    masks = []
    for m in range(32):  # 0..31
        mask = [(m >> i) & 1 == 1 for i in range(5)]
        masks.append(mask)
    return masks

def simulate_expected_value_after_keeps(
    current_values: List[int],
    keep_mask: List[bool],
    rolls_remaining: int,
    available_categories: List[str],
    sims: int
) -> float:
    """
    Estima el valor esperado de la mejor categoría si:
    - conservas (keep_mask) algunos dados ahora,
    - y después solo rerolléas los no conservados el número de veces restante,
    - sin volver a cambiar la máscara (aprox. Montecarlo de 1 paso).
    """
    if rolls_remaining <= 0:
        # Sin tiradas: solo puntuar ahora
        _, sc = best_category_score(current_values, available_categories)
        return float(sc)

    kept = [v if keep_mask[i] else None for i, v in enumerate(current_values)]
    unlocked_idx = [i for i,b in enumerate(keep_mask) if not b]

    total_score = 0.0
    for _ in range(sims):
        # Copia de valores
        vals = kept[:]
        # Rerolls: tirar los no conservados 'rolls_remaining' veces, quedándose con la última
        temp = current_values[:]  # por si keep None
        for _r in range(rolls_remaining):
            for i in unlocked_idx:
                temp[i] = roll_single_die()
        # Construir resultado final
        for i in range(5):
            vals[i] = vals[i] if vals[i] is not None else temp[i]

        # Mejor categoría al final
        _, sc = best_category_score(vals, available_categories)
        total_score += sc

    return total_score / sims if sims > 0 else 0.0

def choose_keep_mask_montecarlo(
    current_values: List[int],
    rolls_remaining: int,
    available_categories: List[str],
    sims_per_mask: Optional[int] = None
) -> List[bool]:
    """
    Elige la máscara de bloqueo que maximiza el valor esperado estimado al final del turno.
    Nota: aproximación de 1 paso (no recalcula bloqueos tras la siguiente tirada durante la simulación).
    """
    # MODIFICA (fallo 2): en la guía el valor por defecto era `sims_per_mask=SIMS_PER_MASK`, que
    # Python fija al DEFINIR la función; cambiar SIMS_PER_MASK después no tenía ningún efecto.
    # Ahora se lee al llamar.
    if sims_per_mask is None:
        sims_per_mask = SIMS_PER_MASK
    best_mask = None
    best_ev = -1.0
    for mask in all_keep_masks():
        # Si el jugador decide "plantarse" ahora, la máscara sería keep todo (no cambia nada).
        # MODIFICA (fallo 1): la guía pasaba `rolls_remaining - 1`. En la última tirada valía 0,
        # el valor esperado no dependía de la máscara y ganaba siempre la primera (relanzar los
        # 5 dados, aunque ya hubiera un Yahtzee). Cada máscara se evalúa con UN relanzamiento.
        ev = simulate_expected_value_after_keeps(
            current_values, mask, 1, available_categories, sims_per_mask
        )
        # MODIFICA: en empate se prefiere conservar más dados (antes ganaba la primera máscara)
        if ev > best_ev or (ev == best_ev and sum(mask) > sum(best_mask)):
            best_ev = ev
            best_mask = mask
    return best_mask

# ===============================
# CREA: otras estrategias para comparar con la de Montecarlo
# ===============================
def choose_keep_mask_naive(current_values, rolls_remaining=2, available_categories=None):
    """Va por Yahtzee: conserva la cara más repetida (empate: la más alta)."""
    counts = Counter(current_values)
    face = max(counts, key=lambda f: (counts[f], f))
    return [v == face for v in current_values]

def choose_keep_mask_random(current_values, rolls_remaining=2, available_categories=None):
    """Conserva cada dado con probabilidad 1/2 (línea base)."""
    return [random.random() < 0.5 for _ in current_values]

def exact_ev_after_keeps(current_values, keep_mask, available_categories) -> float:
    """Valor esperado EXACTO de lo mismo que estima `simulate_expected_value_after_keeps`:
    se enumeran los 6^k resultados posibles de los k dados libres (k <= 5, máx. 7776)."""
    free = [i for i, b in enumerate(keep_mask) if not b]
    vals = list(current_values)
    total = 0
    n = 0
    for combo in itertools.product(range(1, 7), repeat=len(free)):
        for i, v in zip(free, combo):
            vals[i] = v
        total += best_category_score(vals, available_categories)[1]
        n += 1
    return total / n

def exact_evs_all_masks(current_values, available_categories) -> List[float]:
    return [exact_ev_after_keeps(current_values, m, available_categories) for m in all_keep_masks()]

def choose_keep_mask_exact(current_values, rolls_remaining=2, available_categories=None):
    """Igual que la de Montecarlo, pero con el valor esperado exacto (N -> infinito)."""
    best_mask, best_ev = None, -1.0
    for mask in all_keep_masks():
        ev = exact_ev_after_keeps(current_values, mask, available_categories)
        if ev > best_ev + 1e-12 or (abs(ev - best_ev) <= 1e-12 and sum(mask) > sum(best_mask)):
            best_ev, best_mask = ev, mask
    return best_mask

def montecarlo_strategy(sims_per_mask: int):
    """Estrategia de Montecarlo con un número fijo de simulaciones por máscara."""
    def strategy(current_values, rolls_remaining, available_categories):
        return choose_keep_mask_montecarlo(current_values, rolls_remaining,
                                           available_categories, sims_per_mask)
    strategy.__name__ = f"montecarlo({sims_per_mask})"
    return strategy

# ===============================
# Flujo del juego
# ===============================
def play_one_turn(state: GameState, verbose: bool = True, strategy=None):
    """Un turno completo para el jugador actual: hasta 3 tiradas + elección de categoría."""
    # CREA: `strategy` permite que cada jugador use una estrategia distinta; por defecto es la
    # de Montecarlo de la guía.
    if strategy is None:
        strategy = choose_keep_mask_montecarlo
    player = state.current_player
    state.reset_turn()

    # Tirada inicial
    rolled = roll_dice(state.dice_values, state.locked)
    state.stats.update_from_roll(state.dice_values, rolled)   # MODIFICA: solo los lanzados
    state.rolls_left -= 1
    if verbose:
        print(f"\n{player.name} — Tirada 1:", state.dice_values)

    # Hasta 2 rerolls con estrategia MC
    while state.rolls_left > 0:
        keep_mask = strategy(
            state.dice_values, state.rolls_left, player.available_categories()
        )

        # Si la máscara "keep_mask" implica mantener todo tal cual (keep 5), podemos decidir plantarnos:
        if all(keep_mask):
            if verbose:
                print(f"{player.name} decide plantarse con:", state.dice_values)
            break

        # Bloquear según la máscara
        state.locked = keep_mask[:]
        # Reroll
        rolled = roll_dice(state.dice_values, state.locked)
        state.stats.update_from_roll(state.dice_values, rolled)   # MODIFICA: solo los lanzados
        state.rolls_left -= 1
        if verbose:
            tirada_num = 3 - state.rolls_left
            print(f"{player.name} — Tirada {tirada_num}:", state.dice_values, "| keep:", keep_mask)

    # Elegir mejor categoría disponible para la mano final
    cat, sc = best_category_score(state.dice_values, player.available_categories())
    player.set_score(cat, sc)
    state.stats.update_final_hand(state.dice_values)   # CREA
    if verbose:
        print(f"→ {player.name} elige categoría '{cat}' y anota {sc} puntos. Total = {player.total}")

def play_game(verbose_each_turn: bool = False, strategies=None, silent: bool = False) -> GameState:
    # CREA: `strategies` = (estrategia J1, estrategia J2); `silent` apaga los mensajes de inicio
    # y fin, necesario al simular miles de partidas.
    if strategies is None:
        strategies = (None, None)
    players = [PlayerState("Jugador 1"), PlayerState("Jugador 2")]
    state = GameState(players=players)
    total_rounds = 13

    if not silent:
        print("🎲 Iniciando simulación de Montecarlo (Motor de IA tomando decisiones)...")
    for r in range(total_rounds):
        for p in range(len(players)):
            play_one_turn(state, verbose=verbose_each_turn, strategy=strategies[p])
            state.advance_player()
    if not silent:
        print("✅ Simulación completada.\n")
    return state

def winner_index(state: GameState) -> int:   # CREA
    """0 = jugador 1, 1 = jugador 2, -1 = empate."""
    a, b = (p.total for p in state.players)
    return 0 if a > b else 1 if b > a else -1

# ===============================
# Mostrar resultados
# ===============================
def print_scoreboard(state: GameState):
    print("\n\n===== MARCADOR FINAL =====")
    for p in state.players:
        print(f"\n{p.name}: {p.total} puntos (bono superior: {p.bonus})")
        for cat in CATEGORIES_ORDER:
            v = p.scores[cat]
            print(f"  - {cat:13s}: {v if v is not None else 0}")
    # Ganador
    if state.players[0].total > state.players[1].total:
        winner = state.players[0].name
    elif state.players[1].total > state.players[0].total:
        winner = state.players[1].name
    else:
        winner = "Empate"
    print(f"\n🏆 Ganador: {winner}")

def print_stats(state: GameState):
    print("\n===== ESTADÍSTICAS =====")
    # MODIFICA: se informan los dados realmente lanzados (antes se contaban también los bloqueados)
    print(f"Tiradas: {state.stats.total_rolls} | dados realmente lanzados: {state.stats.dice_rolled}")
    total_faces = sum(state.stats.face_hist.values())
    for face in range(1,7):
        c = state.stats.face_hist.get(face, 0)
        p = (c/total_faces)*100 if total_faces else 0.0
        print(f" Cara {face}: {c} veces ({p:.2f}%)")

# =========================================================
# CREA: utilidades de los experimentos
# =========================================================
Z95 = 1.96
CHI2_CRIT_5 = 11.0705   # chi-cuadrado, 5 grados de libertad, alfa = 0.05

def _reseed(offset: int) -> None:
    """Cada experimento arranca con su propia semilla derivada de SEED (resultados repetibles)."""
    if SEED is not None:
        random.seed(SEED + offset)

def _reps(n: int, budget: int = 3_000_000, lo: int = 10, hi: int = 30) -> int:
    """Réplicas por tamaño N: más réplicas para N pequeño, menos para N grande (tiempo acotado)."""
    return max(lo, min(hi, budget // n))

def _save_fig(fig, name: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150)
    plt.close(fig)

def _save_csv(df: pd.DataFrame, name: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, name), index=False)

def loglog_slope(ns, errors) -> Optional[float]:
    """Pendiente de log(error) frente a log(N). Si el error ~ 1/sqrt(N), vale cerca de -0,5."""
    ns, errors = np.asarray(ns, float), np.asarray(errors, float)
    ok = errors > 0
    if ok.sum() < 3:
        return None
    return float(np.polyfit(np.log(ns[ok]), np.log(errors[ok]), 1)[0])

def _ref_line(ax, ns, errors) -> None:
    ns, errors = np.asarray(ns, float), np.asarray(errors, float)
    c = np.median(errors * np.sqrt(ns))
    ax.plot(ns, c / np.sqrt(ns), "k--", lw=1, label="referencia $1/\\sqrt{N}$")

# =========================================================
# CREA — Cadena de Markov: probabilidades EXACTAS de un turno
# =========================================================
# La mano que queda tras cada tirada solo depende de la anterior (si la estrategia es fija),
# así que un turno es una cadena de Markov sobre las 252 manos sin orden. Con esto se calcula
# la "verdad" contra la que se mide el error de la simulación (en la guía, el "valor teórico"
# era el último resultado simulado).
MULTISETS = list(itertools.combinations_with_replacement(range(1, 7), 5))   # 252 manos
MS_INDEX = {ms: i for i, ms in enumerate(MULTISETS)}

def initial_distribution() -> np.ndarray:
    """P(cada mano sin orden) en una tirada de 5 dados (multinomial)."""
    p = np.zeros(len(MULTISETS))
    for i, ms in enumerate(MULTISETS):
        orderings = math.factorial(5) // math.prod(math.factorial(c) for c in Counter(ms).values())
        p[i] = orderings / 6 ** 5
    return p

def transition_matrix(strategy) -> np.ndarray:
    """T[i, j] = P(pasar de la mano i a la j en una tirada) con una estrategia determinista."""
    n = len(MULTISETS)
    T = np.zeros((n, n))
    for s, ms in enumerate(MULTISETS):
        mask = strategy(list(ms), 2, CATEGORIES_ORDER)
        free = [i for i, keep in enumerate(mask) if not keep]
        outcomes = 0
        for combo in itertools.product(range(1, 7), repeat=len(free)):
            vals = list(ms)
            for i, v in zip(free, combo):
                vals[i] = v
            T[s, MS_INDEX[tuple(sorted(vals))]] += 1
            outcomes += 1
        T[s] /= outcomes
    return T

def exact_turn_distribution(strategy, rolls: int = 3) -> np.ndarray:
    """Distribución de la mano final de un turno de `rolls` tiradas."""
    T = transition_matrix(strategy)
    p = initial_distribution()
    for _ in range(rolls - 1):
        p = p @ T
    return p

def exact_category_probability(dist: np.ndarray, category: str) -> float:
    return float(sum(p for p, ms in zip(dist, MULTISETS) if score_hand(category, list(ms)) > 0))

# =========================================================
# CREA — Experimento 1: ¿el dado es uniforme?
# =========================================================
def exp1_dados(ns=(10, 100, 1_000, 10_000, 100_000, 1_000_000)):
    """Frecuencia de cada cara frente a 1/6 al crecer N (solo dados realmente lanzados)."""
    _reseed(1)
    rows, one_run = [], {}
    for n in ns:
        devs, chis = [], []
        for r in range(_reps(n)):
            counts = np.bincount([roll_single_die() - 1 for _ in range(n)], minlength=6)
            if r == 0:
                one_run[n] = counts / n
            devs.append(np.abs(counts / n - 1 / 6).max())
            chis.append(((counts - n / 6) ** 2 / (n / 6)).sum())
        chis = np.array(chis)
        rows.append({"N": n, "replicas": len(devs),
                     "desviacion_max_media": float(np.mean(devs)),
                     "chi2_medio": float(chis.mean()),
                     "rechazo_alfa_5pct": float((chis > CHI2_CRIT_5).mean())})
    df = pd.DataFrame(rows)
    _save_csv(df, "exp1_dados.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    for cara in range(6):
        a.plot(ns, [one_run[n][cara] for n in ns], marker="o", label=f"cara {cara + 1}")
    a.axhline(1 / 6, color="k", ls="--", label="teórico 1/6")
    a.set_xscale("log")
    a.set_xlabel("N (dados lanzados)")
    a.set_ylabel("frecuencia relativa")
    a.set_title("Frecuencia de cada cara (una corrida)")
    a.legend(fontsize=7, ncol=2)
    b.loglog(ns, df["desviacion_max_media"], "o-", label="desviación máxima media")
    _ref_line(b, ns, df["desviacion_max_media"])
    b.set_xlabel("N")
    b.set_ylabel("|frecuencia − 1/6| máxima")
    b.set_title("La desviación cae como $1/\\sqrt{N}$")
    b.legend()
    _save_fig(fig, "exp1_dados.png")
    return {"tabla": df, "pendiente": loglog_slope(ns, df["desviacion_max_media"])}

# =========================================================
# CREA — Experimento 2: probabilidad de Yahtzee, full y escaleras en un turno
# =========================================================
def simulate_turn_naive() -> List[int]:
    """Un turno completo (3 tiradas) con la estrategia ingenua; devuelve la mano final."""
    dice = [roll_single_die() for _ in range(5)]
    for _ in range(2):
        mask = choose_keep_mask_naive(dice)
        if all(mask):
            break
        for i, keep in enumerate(mask):
            if not keep:
                dice[i] = roll_single_die()
    return dice

def exp2_probabilidades(ns=(10, 100, 1_000, 10_000, 100_000, 1_000_000)):
    _reseed(2)
    cats = ["yahtzee", "largeStraight", "smallStraight", "fullHouse"]
    dist = exact_turn_distribution(choose_keep_mask_naive)
    exact = {c: exact_category_probability(dist, c) for c in cats}

    est = {(c, n): [] for c in cats for n in ns}
    for n in ns:
        for _ in range(_reps(n)):
            hits = Counter()
            for _t in range(n):
                hand = simulate_turn_naive()
                for c in cats:
                    hits[c] += score_hand(c, hand) > 0
            for c in cats:
                est[(c, n)].append(hits[c] / n)

    rows, errors, slopes = [], {c: [] for c in cats}, {}
    for c in cats:
        for n in ns:
            e = np.array(est[(c, n)])
            err = float(np.abs(e - exact[c]).mean())
            errors[c].append(err)
            rows.append({"categoria": c, "N": n, "replicas": len(e), "exacto": exact[c],
                         "estimacion_media": float(e.mean()),
                         "desv_est_replicas": float(e.std(ddof=1)), "error_abs_medio": err,
                         "semiancho_ic95_teorico": Z95 * math.sqrt(exact[c] * (1 - exact[c]) / n)})
    _save_csv(pd.DataFrame(rows), "exp2_probabilidades.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    for n in ns:
        a.plot([n] * len(est[("yahtzee", n)]), est[("yahtzee", n)], "o", color="tab:blue",
               alpha=0.35, ms=3)
    a.plot(ns, [np.mean(est[("yahtzee", n)]) for n in ns], "o-", color="tab:blue",
           label="media de las réplicas")
    a.axhline(exact["yahtzee"], color="r", ls="--", label=f"exacto = {exact['yahtzee']:.4f}")
    a.set_xscale("log")
    a.set_xlabel("N (turnos simulados)")
    a.set_ylabel("P(Yahtzee en un turno)")
    a.set_title("Convergencia de la probabilidad de Yahtzee")
    a.legend()
    for c in cats:
        # la pendiente solo tiene sentido donde el evento se observa >= 30 veces en promedio
        ok = [i for i, n in enumerate(ns) if n * exact[c] >= 30]
        slopes[c] = loglog_slope([ns[i] for i in ok], [errors[c][i] for i in ok])
        label = c if slopes[c] is None else f"{c} (pendiente {slopes[c]:.2f})"
        b.loglog(ns, errors[c], "o-", label=label)
    _ref_line(b, ns, errors["yahtzee"])
    b.set_xlabel("N")
    b.set_ylabel("error absoluto medio")
    b.set_title("Error frente al valor exacto (log-log)")
    b.legend(fontsize=8)
    _save_fig(fig, "exp2_probabilidades.png")
    return {"exacto": exact, "pendientes": slopes}

# =========================================================
# Estudio de convergencia (Ley de los Grandes Números)
# =========================================================
def reporte_convergencia_lln(valores_iteraciones=(10, 30, 100, 300, 1_000, 3_000, 10_000, 100_000)):
    # MODIFICA: misma mano y máscara de la guía, pero (1) el valor de referencia es el EXACTO
    # (enumeración), no el último resultado simulado; (2) hay varias réplicas por N, con
    # dispersión; (3) se mide la pendiente log-log del error (debe ser ~ -0,5).
    _reseed(3)
    dados_iniciales = [3, 3, 5, 2, 6]
    categorias_disponibles = ["threes", "fullHouse", "threeOfKind", "chance"]
    mascara_keep = [True, True, False, False, False]
    exacto = exact_ev_after_keeps(dados_iniciales, mascara_keep, categorias_disponibles)

    print("\n⏳ Iniciando experimento de convergencia...")
    rows = []
    for sims in valores_iteraciones:
        reps = _reps(sims, budget=2_000_000, lo=20, hi=200)
        estimaciones, t0 = [], time.perf_counter()
        for _ in range(reps):
            estimaciones.append(simulate_expected_value_after_keeps(
                dados_iniciales, mascara_keep, 1, categorias_disponibles, sims))
        t_est = (time.perf_counter() - t0) / reps
        e = np.array(estimaciones)
        rows.append({"Iteraciones": sims, "Replicas": reps, "EV exacto": exacto,
                     "EV estimado (media)": float(e.mean()),
                     "Desv. est. replicas": float(e.std(ddof=1)),
                     "Error abs. medio": float(np.abs(e - exacto).mean()),
                     "Tiempo por estimacion (s)": t_est})
        print(f"   Iteraciones: {sims:6d} | EV medio: {e.mean():.4f} | error medio: "
              f"{rows[-1]['Error abs. medio']:.4f} | {t_est:.4f}s")
    df = pd.DataFrame(rows)
    _save_csv(df, "exp3_convergencia_ev.csv")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.8))
    ns = df["Iteraciones"].to_numpy()
    media, desv = df["EV estimado (media)"].to_numpy(), df["Desv. est. replicas"].to_numpy()
    ax1.plot(ns, media, "o-", color="b", label="media de las réplicas")
    ax1.fill_between(ns, media - Z95 * desv, media + Z95 * desv, alpha=0.25,
                     label="±1,96 desv. est. (réplicas)")
    ax1.axhline(exacto, color="r", ls="--", label=f"EV exacto = {exacto:.3f}")
    ax1.set_xscale("log")
    ax1.set_title("Convergencia del valor esperado (EV)\n(Ley de los Grandes Números)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Iteraciones (escala logarítmica)")
    ax1.set_ylabel("Valor esperado (EV)")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", ls="--")
    ax2.loglog(ns, df["Error abs. medio"], "o-", label="error absoluto medio")
    _ref_line(ax2, ns, df["Error abs. medio"])
    ax2.set_title("El error decrece como $1/\\sqrt{N}$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Iteraciones")
    ax2.set_ylabel("|estimado − exacto|")
    ax2.legend()
    ax2.grid(True, which="both", ls="--")
    ax3.loglog(ns, df["Tiempo por estimacion (s)"], "s-", color="g")
    ax3.set_title("Costo computacional vs iteraciones", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Iteraciones")
    ax3.set_ylabel("Tiempo por estimación (s)")
    ax3.grid(True, which="both", ls="--")
    _save_fig(fig, "exp3_convergencia_ev.png")
    return {"exacto": exacto, "pendiente": loglog_slope(ns, df["Error abs. medio"]), "tabla": df}

# =========================================================
# CREA — Experimento 3b: calidad de la decisión según SIMS_PER_MASK
# =========================================================
def exp3b_calidad_decision(sims_list=(1, 3, 10, 30, 100, 300, 1_000, 3_000), manos=60):
    """Reemplaza la comparación 10/1.000/100.000 de la guía (que no tenía efecto): mide qué tan
    buena es la máscara elegida frente a la mejor según el valor esperado exacto."""
    _reseed(4)
    casos = []
    for _ in range(manos):
        values = [roll_single_die() for _ in range(5)]
        k = random.randint(3, 13)
        elegidas = set(random.sample(CATEGORIES_ORDER, k))
        disponibles = [c for c in CATEGORIES_ORDER if c in elegidas]
        casos.append((values, disponibles, exact_evs_all_masks(values, disponibles)))

    rows = []
    for s in sims_list:
        t0 = time.perf_counter()
        regrets = []
        for values, disponibles, evs in casos:
            mask = choose_keep_mask_montecarlo(values, 2, disponibles, sims_per_mask=s)
            idx = sum(1 << i for i, b in enumerate(mask) if b)
            regrets.append(max(evs) - evs[idx])
        t = (time.perf_counter() - t0) / len(casos)
        regrets = np.array(regrets)
        rows.append({"sims_per_mask": s, "arrepentimiento_medio": float(regrets.mean()),
                     "porcentaje_decision_optima": float(100 * (regrets < 1e-9).mean()),
                     "segundos_por_decision": t})
    df = pd.DataFrame(rows)
    _save_csv(df, "exp3b_calidad_decision.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    a.plot(df["sims_per_mask"], df["porcentaje_decision_optima"], "o-")
    a.set_xscale("log")
    a.set_xlabel("simulaciones por máscara (SIMS_PER_MASK)")
    a.set_ylabel("% de decisiones óptimas (de 1 paso)")
    a.set_title("Calidad de la decisión de Montecarlo")
    b.loglog(df["sims_per_mask"], df["segundos_por_decision"], "s-", color="g")
    b.set_xlabel("simulaciones por máscara (SIMS_PER_MASK)")
    b.set_ylabel("segundos por decisión")
    b.set_title("Costo computacional")
    _save_fig(fig, "exp3b_calidad_decision.png")
    return {"tabla": df}

# =========================================================
# Comparación de SIMS_PER_MASK en partidas completas
# =========================================================
def comparar_sims_per_mask(escenarios_sims=(10, 100, 1_000), partidas=3):
    # MODIFICA: la guía comparaba 10, 1000 y 100000, pero (fallo 2) el parámetro no se aplicaba.
    # Ahora sí se aplica; 100000 se cambia por 100 porque costaría ~20 min por partida en Python.
    # Además se promedian varias partidas por escenario (una sola partida no dice nada).
    _reseed(5)
    global SIMS_PER_MASK
    original = SIMS_PER_MASK
    resultados_comparativa = []
    try:
        for sims in escenarios_sims:
            print(f"\n⏳ Ejecutando {partidas} partidas para SIMS_PER_MASK = {sims}...")
            SIMS_PER_MASK = sims
            t0 = time.time()
            estados = [play_game(verbose_each_turn=False, silent=True) for _ in range(partidas)]
            elapsed = (time.time() - t0) / partidas
            puntos = [p.total for e in estados for p in e.players]
            resultados_comparativa.append({
                "Sims por Máscara": sims,
                "Tiempo por partida (s)": round(elapsed, 3),
                "Puntaje medio por jugador": round(float(np.mean(puntos)), 1),
                "Yahtzees por turno": round(sum(e.stats.yahtzees for e in estados) /
                                            sum(e.stats.turns for e in estados), 4),
            })
    finally:
        SIMS_PER_MASK = original

    df_comparativo = pd.DataFrame(resultados_comparativa)
    print("\n=== TABLA COMPARATIVA DE ESCENARIOS ===")
    print(df_comparativo.to_string(index=False))
    _save_csv(df_comparativo, "comparativa_sims.csv")

    fig = plt.figure(figsize=(8, 4))
    sns.barplot(x="Sims por Máscara", y="Tiempo por partida (s)", hue="Sims por Máscara",
                data=df_comparativo, palette="viridis", legend=False)
    plt.title("Tiempo de ejecución según SIMS_PER_MASK", fontsize=12, fontweight="bold")
    plt.ylabel("Tiempo por partida (segundos)")
    plt.xlabel("Simulaciones por máscara")
    _save_fig(fig, "comparativa_sims.png")
    return df_comparativo

# =========================================================
# CREA — Experimento 4: partidas completas de 2 jugadores
# =========================================================
def exp4_partidas(n_partidas, strat1, strat2, etiqueta):
    n = n_partidas
    totales = np.zeros((n, 2))
    ganadores = np.zeros(n, dtype=int)
    acumulado = GameStats()
    t0 = time.perf_counter()
    for g in range(n):
        estado = play_game(strategies=(strat1, strat2), silent=True)
        totales[g] = [p.total for p in estado.players]
        ganadores[g] = winner_index(estado)
        acumulado.merge(estado.stats)
    segundos = time.perf_counter() - t0

    k = np.arange(1, n + 1)
    g1, g2 = np.cumsum(ganadores == 0), np.cumsum(ganadores == 1)
    decididas = np.maximum(g1 + g2, 1)
    cuota1 = g1 / decididas
    ic_cuota = Z95 * np.sqrt(cuota1 * (1 - cuota1) / decididas)
    medias = np.cumsum(totales, axis=0) / k[:, None]
    varianzas = np.maximum(np.cumsum(totales ** 2, axis=0) / k[:, None] - medias ** 2, 0)
    ic_media = Z95 * np.sqrt(varianzas / k[:, None])

    puntos = sorted({int(x) for x in np.geomspace(1, n, 25).astype(int)})
    _save_csv(pd.DataFrame([{
        "partidas": p, "cuota_victorias_j1": cuota1[p - 1], "ic95_cuota": ic_cuota[p - 1],
        "media_j1": medias[p - 1, 0], "media_j2": medias[p - 1, 1],
        "ic95_media_j1": ic_media[p - 1, 0], "ic95_media_j2": ic_media[p - 1, 1],
    } for p in puntos]), f"exp4_{etiqueta}.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    a.plot(k, cuota1, label="jugador 1")
    a.fill_between(k, cuota1 - ic_cuota, cuota1 + ic_cuota, alpha=0.25, label="IC 95 %")
    a.axhline(0.5, color="r", ls="--", label="0,5 (jugadores idénticos)")
    a.set_xscale("log")
    a.set_ylim(0, 1)
    a.set_xlabel("partidas jugadas")
    a.set_ylabel("fracción de victorias del J1 (sin empates)")
    a.set_title(f"Victorias acumuladas — {etiqueta}")
    a.legend()
    for j, color in enumerate(("tab:blue", "tab:orange")):
        b.plot(k, medias[:, j], color=color, label=f"jugador {j + 1}")
        b.fill_between(k, medias[:, j] - ic_media[:, j], medias[:, j] + ic_media[:, j],
                       color=color, alpha=0.2)
    b.set_xscale("log")
    b.set_xlabel("partidas jugadas")
    b.set_ylabel("puntaje medio acumulado")
    b.set_title("El puntaje medio se estabiliza")
    b.legend()
    _save_fig(fig, f"exp4_{etiqueta}.png")

    # Sesgo del conteo de la guía (con dados bloqueados) frente al conteo corregido
    ok = np.array([acumulado.face_hist[c] for c in range(1, 7)], float)
    orig = np.array([acumulado.face_hist_all[c] for c in range(1, 7)], float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(1, 7)
    ax.bar(x - 0.2, ok / ok.sum(), 0.4, label="corregido: solo dados lanzados")
    ax.bar(x + 0.2, orig / orig.sum(), 0.4, label="guía: cuenta también los bloqueados")
    ax.axhline(1 / 6, color="r", ls="--", label="1/6")
    ax.set_ylim(0, 0.24)
    ax.set_xlabel("cara")
    ax.set_ylabel("frecuencia relativa")
    ax.set_title(f"Frecuencia de caras en las partidas — {etiqueta}")
    ax.legend(fontsize=8)
    _save_fig(fig, f"exp4_caras_{etiqueta}.png")

    return {"etiqueta": etiqueta, "partidas": n, "segundos": round(segundos, 1),
            "victorias_j1": int(g1[-1]), "victorias_j2": int(g2[-1]),
            "empates": int((ganadores == -1).sum()),
            "media_j1": round(float(medias[-1, 0]), 2), "media_j2": round(float(medias[-1, 1]), 2),
            "yahtzee_por_turno": round(acumulado.yahtzees / acumulado.turns, 4)}

def exp4_todo(rapido: bool = False, sims: int = 30):
    """Tres duelos: Montecarlo vs Montecarlo, ingenua vs ingenua, Montecarlo vs ingenua."""
    n_mc, n_ing, n_duelo = (30, 2_000, 30) if rapido else (1_000, 100_000, 1_000)
    mc = montecarlo_strategy(sims)
    filas = []
    for i, (etiqueta, e1, e2, n) in enumerate([
        ("mc_vs_mc", mc, mc, n_mc),
        ("ingenua_vs_ingenua", choose_keep_mask_naive, choose_keep_mask_naive, n_ing),
        ("mc_vs_ingenua", mc, choose_keep_mask_naive, n_duelo),
    ]):
        _reseed(10 + i)
        filas.append(exp4_partidas(n, e1, e2, etiqueta))
    _save_csv(pd.DataFrame(filas), "exp4_resumen.csv")
    return filas

# =========================================================
# Diagrama de flujo (Graphviz)
# =========================================================
def generar_diagrama_flujo():
    # MODIFICA: la guía dibujaba una arista hacia un nodo inexistente ('mejor_mask_decision')
    # y el diagrama no reflejaba el algoritmo corregido. Si falta Graphviz solo se guarda el .gv.
    try:
        from graphviz import Digraph
    except ImportError:
        print("Falta el paquete 'graphviz' (pip install graphviz); se omite el diagrama.")
        return None
    dot = Digraph(comment='Algoritmo Yahtzee Montecarlo', format='png')
    dot.attr(rankdir='TB', size='8,10', dpi='200')

    # Estilos de nodos
    dot.attr('node', shape='box', style='filled,rounded', color='black', fillcolor='#E3F2FD', fontname='Arial', fontsize='10')

    # Definir nodos
    dot.node('inicio', 'Inicio de Turno\n(Tirada Inicial: 5 Dados, tirada = 1)', fillcolor='#C8E6C9', shape='ellipse')
    dot.node('eval_tiros', '¿tirada < 3?', shape='diamond', fillcolor='#FFF9C4')
    dot.node('eval_mascaras', 'Evaluar las 32 Máscaras\nPosibles de Bloqueo (Keep Masks)', fillcolor='#E1BEE7')
    dot.node('montecarlo', 'Para cada máscara:\nsimular N relanzamientos de los dados libres\ny promediar el mejor puntaje (EV)', fillcolor='#BBDEFB')
    dot.node('mejor_mascara', 'Seleccionar máscara con máximo EV\n(empate: conservar más dados)', fillcolor='#BBDEFB')
    dot.node('plantarse', '¿La mejor máscara es\nmantener todos los dados?', shape='diamond', fillcolor='#FFF9C4')
    dot.node('actualizar_dados', 'Lanzar solo los dados no bloqueados\ntirada = tirada + 1\n(estadística: solo dados lanzados)', fillcolor='#FFE0B2')
    dot.node('anotar', 'Anotar en la mejor categoría\ndisponible en la hoja', fillcolor='#F8BBD0')
    dot.node('fin', 'Fin del Turno', fillcolor='#FFCDD2', shape='ellipse')

    # Conexiones
    dot.edge('inicio', 'eval_tiros')
    dot.edge('eval_tiros', 'eval_mascaras', label='Sí')
    dot.edge('eval_tiros', 'anotar', label='No')
    dot.edge('eval_mascaras', 'montecarlo')
    dot.edge('montecarlo', 'mejor_mascara')
    dot.edge('mejor_mascara', 'plantarse')
    dot.edge('plantarse', 'anotar', label='Sí')
    dot.edge('plantarse', 'actualizar_dados', label='No')
    dot.edge('actualizar_dados', 'eval_tiros')
    dot.edge('anotar', 'fin')

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, 'diagrama_flujo_yahtzee')
    dot.save(base + '.gv')
    try:
        dot.render(base, cleanup=True)
        print(f"✅ Diagrama de flujo generado y guardado como '{base}.png'")
    except Exception as exc:  # falta el programa `dot` de Graphviz
        print(f"No se pudo generar el PNG ({exc.__class__.__name__}); instala Graphviz "
              f"(sudo apt-get install graphviz). Se guardó {base}.gv")
    return dot

# =========================================================
# ANÁLISIS Y VISUALIZACIÓN DE UNA PARTIDA
# =========================================================
def analisis_didactico_resultados(state: GameState):
    """Presenta los resultados de la simulación de forma visual y pedagógica."""

    # Configuración visual de Seaborn
    sns.set_theme(style="whitegrid")

    # --- 1. RESUMEN DEL PARTIDO (TABLA VISUAL) ---
    print("="*50)
    print(" 📊 ANÁLISIS DE LA PARTIDA Y TOMA DE DECISIONES")
    print("="*50)

    # Determinamos al ganador
    p1, p2 = state.players
    if p1.total > p2.total:
        ganador = p1.name
    elif p2.total > p1.total:
        ganador = p2.name
    else:
        ganador = "Empate"

    print(f"\n🏆 Resultado de la Simulación: ¡El ganador es {ganador}!\n")

    # Construimos un DataFrame de Pandas para que los estudiantes vean una tabla estructurada
    df_scores = pd.DataFrame({
        "Categoría": CATEGORIES_ORDER,
        p1.name: [p1.scores[c] if p1.scores[c] is not None else 0 for c in CATEGORIES_ORDER],
        p2.name: [p2.scores[c] if p2.scores[c] is not None else 0 for c in CATEGORIES_ORDER]
    })

    # Mostramos la tabla formateada
    print(df_scores.to_string(index=False))
    print("-" * 30)
    print(f"PUNTUACIÓN TOTAL | {p1.name}: {p1.total} | {p2.name}: {p2.total}")

    # --- GRÁFICO 1: COMPARACIÓN DE PUNTUACIONES ---
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(x="Categoría", y="Valor", hue="Jugador",
                data=pd.melt(df_scores, id_vars="Categoría", var_name="Jugador", value_name="Valor"),
                palette="muted")
    plt.title("Comparación de Puntuaciones por Categoría", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Puntos Obtenidos")
    plt.xlabel("Categoría de Yahtzee")
    _save_fig(fig, "demo_puntuaciones.png")   # MODIFICA: se guarda en vez de plt.show()

    # --- 2. ANÁLISIS DE PROBABILIDAD Y LA LEY DE LOS GRANDES NÚMEROS ---
    print("\n" + "="*50)
    print(" 🎲 ANÁLISIS DEL MOTOR DE DADOS (Validación de Montecarlo)")
    print("="*50)
    print("\nEn una simulación de Montecarlo, es vital comprobar si nuestro generador aleatorio es justo.")
    print(f"Teóricamente, cada cara de un dado perfecto tiene una probabilidad de: 1/6 = {100/6:.2f}%")
    print(f"Veamos cómo se comportaron nuestros dados tras {state.stats.dice_rolled} dados lanzados:\n")

    total_faces = sum(state.stats.face_hist.values())
    caras = list(range(1, 7))
    frecuencias = [state.stats.face_hist.get(face, 0) for face in caras]
    probabilidades_empiricas = [(frec / total_faces) * 100 if total_faces else 0 for frec in frecuencias]

    # Imprimimos los resultados teóricos vs simulados
    for face, frec, prob in zip(caras, frecuencias, probabilidades_empiricas):
        error_marginal = abs(prob - 100 / 6)   # MODIFICA: 1/6 exacto en vez de 16.67
        print(f" Cara {face}: Salió {frec} veces -> {prob:.2f}% (Diferencia vs Teoría: {error_marginal:.2f}%)")

    # --- GRÁFICO 2: DISTRIBUCIÓN DE PROBABILIDAD ---
    fig = plt.figure(figsize=(8, 5))

    # Graficamos los datos empíricos
    bar_plot = sns.barplot(x=caras, y=probabilidades_empiricas, color="skyblue", label="Probabilidad Simulada")

    # Agregamos la línea teórica
    plt.axhline(y=100 / 6, color='red', linestyle='--', linewidth=2, label='Probabilidad Teórica (16.67%)')

    plt.title(f"Distribución de Caras tras {state.stats.dice_rolled} dados lanzados", fontsize=14, fontweight="bold")
    plt.xlabel("Cara del Dado")
    plt.ylabel("Frecuencia Relativa (%)")
    plt.ylim(0, max(probabilidades_empiricas) + 5) # Damos espacio arriba
    plt.legend()

    # Añadir los porcentajes encima de las barras para máxima claridad
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.1f') + '%',
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha = 'center', va = 'center',
                          xytext = (0, 9),
                          textcoords = 'offset points')

    _save_fig(fig, "demo_distribucion_caras.png")   # MODIFICA: se guarda en vez de plt.show()

    print("\n💡 REFLEXIÓN DIDÁCTICA:")
    print("Con una sola partida hay pocos dados, y las barras se alejan de la línea roja. Gracias a la")
    print("'Ley de los Grandes Números', al aumentar el número de lanzamientos las barras se acercan a ella")
    print("(ver el experimento 'dados'). El análisis con miles de partidas está en los comandos siguientes.")

# =========================================================
# Programa principal
# =========================================================
def main() -> None:
    global SEED, OUT_DIR
    # MODIFICA: la guía ejecutaba todo al importar el archivo; ahora hay comandos.
    p = argparse.ArgumentParser(description="Yahtzee con el método de Montecarlo")
    p.add_argument("comando", choices=["demo", "comparativa", "dados", "probabilidades",
                                       "convergencia", "decisiones", "partidas", "diagrama", "todo"])
    p.add_argument("--seed", type=int, default=SEED, help="semilla (42 por defecto)")
    p.add_argument("--salida", default=OUT_DIR, help="carpeta de CSV y figuras")
    p.add_argument("--rapido", action="store_true", help="tamaños pequeños, para probar")
    a = p.parse_args()
    SEED, OUT_DIR = a.seed, a.salida
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # tildes y emojis también en Windows
    plt.switch_backend("Agg")                      # sin pantalla (Codespaces, servidores)
    if SEED is not None:
        random.seed(SEED)

    ns = (10, 100, 1_000, 10_000, 100_000) if a.rapido else (10, 100, 1_000, 10_000, 100_000, 1_000_000)
    todo = a.comando == "todo"

    if a.comando == "demo":
        state = play_game(verbose_each_turn=True)
        print_scoreboard(state)
        print_stats(state)
        analisis_didactico_resultados(state)
    if a.comando in ("comparativa", "todo"):
        comparar_sims_per_mask(partidas=1 if a.rapido else 3)
    if a.comando in ("dados", "todo"):
        r = exp1_dados(ns)
        print("Exp 1 (dados): pendiente log-log del error =", round(r["pendiente"], 3))
    if a.comando in ("probabilidades", "todo"):
        r = exp2_probabilidades(ns)
        print("Exp 2 (probabilidades exactas):", {k: round(v, 5) for k, v in r["exacto"].items()})
        print("       pendientes log-log:", {k: None if v is None else round(v, 3)
                                             for k, v in r["pendientes"].items()})
    if a.comando in ("convergencia", "todo"):
        r = reporte_convergencia_lln()
        print(f"Exp 3: EV exacto = {r['exacto']:.4f}; pendiente log-log del error = {r['pendiente']:.3f}")
    if a.comando in ("decisiones", "todo"):
        r = exp3b_calidad_decision(manos=20 if a.rapido else 60)
        print(r["tabla"].to_string(index=False))
    if a.comando in ("partidas", "todo"):
        for fila in exp4_todo(rapido=a.rapido):
            print("Exp 4:", fila)
    if a.comando in ("diagrama", "todo"):
        generar_diagrama_flujo()

if __name__ == "__main__":
    main()
