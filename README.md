# Yahtzee con el método de Montecarlo

Actividad didáctica 2-M1: simulación del Yahtzee clásico de 2 jugadores (hasta 3 tiradas de 5 dados
por turno, distribución uniforme) y análisis de la **Ley de los Grandes Números** con Montecarlo.

Se trabajó **sobre el mismo archivo de la guía**: [`simulacion_01.py`](simulacion_01.py) conserva la
estructura, los nombres y el orden del original ([`original/simulacion_01_original.py`](original/simulacion_01_original.py),
sin modificar). Cada cambio está marcado en el código:

- `# MODIFICA:` cambio sobre código que ya existía (corrige o mejora algo).
- `# CREA:` código nuevo que no existía en la guía.

Para ver exactamente qué cambió: `git diff --no-index original/simulacion_01_original.py simulacion_01.py`
o buscar esas etiquetas en el archivo.

Documentación: el informe técnico está en [`docs/Vera_Jonathan_Montecarlo.docx`](docs/Vera_Jonathan_Montecarlo.docx) y el guion del video en [`docs/guion_video.md`](docs/guion_video.md).

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
sudo apt-get install -y graphviz       # opcional, solo para el PNG del diagrama de flujo
python -m pytest -q                    # 31 pruebas
python simulacion_01.py demo           # una partida narrada, tirada por tirada + análisis
python simulacion_01.py todo           # todos los experimentos (varios minutos); --rapido para probar
```

Comandos: `demo`, `comparativa`, `dados`, `probabilidades`, `convergencia`, `decisiones`, `partidas`,
`diagrama`, `todo`. Opciones: `--seed 42` (por defecto), `--salida resultados`, `--rapido`.
Con la misma semilla los resultados se repiten. CSV y figuras quedan en [`resultados/`](resultados/).

## Modelo de probabilidad

Cada dado es una variable uniforme discreta: P(X = k) = 1/6 para k = 1..6, con dados independientes
(`random.randint(1, 6)`). Un turno es: tirar 5 dados, elegir cuáles conservar (32 máscaras), relanzar
los demás (hasta 2 veces) y anotar en una categoría libre.

**Decisión por Montecarlo.** Para cada máscara se simulan `SIMS_PER_MASK` relanzamientos de los dados
libres, se toma el mejor puntaje disponible de cada mano resultante y se promedia: es una estimación del
valor esperado `E[mejor puntaje | máscara]`. Se elige la máscara de mayor promedio (conservar los 5 dados
equivale a plantarse).

## USA → MODIFICA → CREA

### USA
Se ejecutó la guía tal cual: dos jugadores con IA Montecarlo, una partida completa, comparación de
`SIMS_PER_MASK`, estudio de convergencia y gráficas de frecuencias.

### MODIFICA (errores encontrados en la guía y corregidos)

| # | Fallo en la guía | Corrección |
|---|---|---|
| 1 | En la **3.ª tirada siempre relanzaba los 5 dados**: `choose_keep_mask_montecarlo` pasaba `rolls_remaining - 1`, que valía 0; el valor esperado dejaba de depender de la máscara y ganaba la primera (relanzar todo), aun con un Yahtzee en la mano | Cada máscara se evalúa con un relanzamiento. Prueba: `test_ultima_tirada_ya_no_relanza_todo` |
| 2 | `sims_per_mask: int = SIMS_PER_MASK` se fija **al definir** la función: la comparación 10 / 1.000 / 100.000 corría siempre con 300 | El valor se lee al llamar. La comparación ahora es real. Prueba: `test_sims_per_mask_se_lee_al_llamar` |
| 3 | La estadística de caras contaba los 5 dados en cada tirada, también los **bloqueados** (muestras repetidas, no independientes, sesgadas por la estrategia) | `roll_dice` devuelve los índices lanzados y `GameStats.face_hist` cuenta solo esos. Se conserva `face_hist_all` (conteo de la guía) para mostrar el sesgo |
| 4 | El "valor teórico" del estudio de convergencia era el último resultado simulado (circular), con una sola réplica por N | Valor **exacto** por enumeración y varias réplicas por N (`reporte_convergencia_lln`) |
| 5 | La comparación usaba **una sola partida** por escenario y 100.000 simulaciones (~20 min por partida) | Se promedian varias partidas y se usa 10 / 100 / 1.000 (`comparar_sims_per_mask`) |
| 6 | El diagrama de flujo tenía una arista hacia un nodo inexistente (`mejor_mask_decision`) y no reflejaba el algoritmo corregido | `generar_diagrama_flujo` corregido |
| 7 | Todo se ejecutaba al importar; celda duplicada; gráficas con `plt.show()`; 16.67 escrito a mano; en empate de valor esperado ganaba la primera máscara (descartando dados) | Comandos con `argparse`, figuras con `savefig`, 1/6 exacto, el empate conserva más dados |
| 8 | `score_hand` recalculaba la puntuación millones de veces | Cache por mano ordenada (`lru_cache`): solo hay 252 manos distintas |

### CREA (aportes propios)

- **Cálculo exacto**: `exact_ev_after_keeps` enumera los 6^k resultados posibles y da el valor esperado
  verdadero de cada máscara; así se mide el error *real* de Montecarlo.
- **Cadena de Markov de 252 estados** (`transition_matrix`, `exact_turn_distribution`): probabilidades
  exactas de Yahtzee, full y escaleras en un turno.
- **Estrategias intercambiables**: Montecarlo, exacta, ingenua (conservar la cara más repetida) y aleatoria.
  `play_game(strategies=(...))` permite que cada jugador use la suya.
- **Bono de 35 puntos** si la sección superior suma ≥ 63.
- **Estadísticas de turno**: Yahtzees, escaleras grandes y fulls obtenidos.
- **Cinco experimentos** con réplicas, intervalos de confianza y semilla propia por experimento (abajo).
- **31 pruebas** con `pytest`.

## Experimentos y resultados (semilla 42)

Figuras y tablas completas en `resultados/`. Salida de consola en `resultados/consola.txt`.

**Exp. 1 — ¿Es uniforme el dado?** (`dados`, `exp1_dados.*`). Con N dados lanzados, la desviación máxima de
las frecuencias respecto a 1/6 baja de 0,226 (N = 10) a 0,00056 (N = 10⁶), con pendiente log-log **−0,52**
(teoría: −0,5). El χ² medio (5 grados de libertad) oscila alrededor de 5. El rechazo al 5 % es ruidoso con
pocas réplicas (0 % a 23 %).

**Exp. 2 — Probabilidades en un turno** (`probabilidades`, `exp2_probabilidades.*`). Con la estrategia
ingenua, la cadena de Markov da: P(Yahtzee) = **0,04603** (la guía decía ≈ 4,6 %), P(full) = 0,0988,
P(escalera pequeña) = 0,0248, P(escalera grande) = 0,00016. La simulación converge a esos valores y el error
absoluto cae con pendientes **−0,50** (Yahtzee), −0,54 (escalera pequeña) y −0,44 (full). La escalera grande
es tan rara que con N < 10⁵ casi todas las réplicas valen 0 (hace falta N ≫ 1/p ≈ 6.000), por eso no se
calcula su pendiente.

**Exp. 3 — Valor esperado Montecarlo frente al exacto** (`convergencia`, `exp3_convergencia_ev.*`; mano
3-3-5-2-6 conservando los dos 3, la misma de la guía). Valor exacto = 17,3009. El error medio baja de 0,99
(N = 10) a 0,0085 (N = 10⁵), con pendiente **−0,51**. El costo crece linealmente con N.

**Exp. 3b — Calidad de la decisión según `SIMS_PER_MASK`** (`decisiones`, `exp3b_calidad_decision.*`; 60
manos aleatorias con categorías libres aleatorias). Al pasar de 1 a 3.000 simulaciones por máscara, las
decisiones óptimas (de un paso, respecto al valor exacto) suben de 20 % a 95 %, el arrepentimiento medio baja
de 3,7 a 0,007 puntos y el costo sube de 0,12 a 160 ms por decisión.

**Comparación de partidas por `SIMS_PER_MASK`** (`comparativa`, `comparativa_sims.*`): 3 partidas por
escenario, así que sirve para el costo (0,04 s, 0,28 s y 2,8 s por partida con 10, 100 y 1.000), no para
comparar puntajes (muestra muy pequeña); para eso están el Exp. 3b y el Exp. 4.

**Exp. 4 — Partidas completas** (`partidas`, `exp4_*.png`, `exp4_resumen.csv`):

| Duelo | Partidas | Victorias J1 / J2 / empates | Puntaje medio J1 / J2 |
|---|---|---|---|
| Montecarlo (30 sims) vs Montecarlo (30 sims) | 1.000 | 492 / 499 / 9 | 201,1 / 202,4 |
| Ingenua vs ingenua | 100.000 | 49.443 / 49.771 / 786 | 144,7 / 144,8 |
| Montecarlo (30 sims) vs ingenua | 1.000 | 863 / 135 / 2 | 200,9 / 144,2 |

Con jugadores idénticos, la fracción de victorias tiende a 0,5 y los puntajes medios se igualan; la banda de
confianza se estrecha al crecer el número de partidas (las dos primeras filas). La estrategia de Montecarlo
supera claramente a la ingenua. Las figuras `exp4_caras_*.png` muestran que el conteo de la guía (con dados
bloqueados) se aleja de 1/6, y el corregido no.

**Observación.** La estrategia Montecarlo maximiza el puntaje del turno, no la probabilidad de Yahtzee: saca
~1,8 % de Yahtzees por turno frente a 4,6 % de la ingenua, pero puntúa ~57 puntos más por partida porque
aprovecha todas las categorías.

**Límites de este trabajo.** La decisión de Montecarlo mira un solo paso (no replanifica la tercera tirada
al decidir la segunda), y la elección de categoría es voraz (la que más puntos da en ese instante, sin
reservar categorías ni buscar el bono). No es la estrategia óptima de Yahtzee, ni pretende serlo.

## Conclusión (Ley de los Grandes Números)

En todos los experimentos, el promedio muestral se acerca al valor teórico al crecer N, y el error absoluto
decrece como 1/√N (pendientes log-log entre −0,44 y −0,54; teoría: −0,5). La verificación se hace contra
valores **exactos** (enumeración y cadena de Markov), no contra otra simulación. Con jugadores idénticos, la
fracción de victorias y el puntaje medio se estabilizan en 0,5 y en un mismo valor al aumentar el número de
partidas.

## Referencias

- Metropolis, N. y Ulam, S. (1949). The Monte Carlo Method. *Journal of the American Statistical Association*, 44(247), 335-341.
- Ross, S. M. (2013). *Simulation* (5.ª ed.). Academic Press.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Glenn, J. (2006). *An Optimal Strategy for Yahtzee*. Informe técnico, Loyola College in Maryland.
- Guía de la actividad: «Semana 2 - Laboratorio Yahtzee» (código base `simulacion_01`).
