# 📘 Documento Técnico: DeepMidi-LSTM — Generación Musical Simbólica mediante Redes Neuronales Recurrentes (LSTM) y Tokenización REMI

- **Universidad:** Universidad de Guadalajara — Centro Universitario de Guadalajara (CUGDL)
- **Plan de Estudios:** Licenciatura en Inteligencia Artificial y Ciencia de Datos
- **Curso:** Diseño, Visualización y Comprensión de Deep Neural Networks
- **Profesor:** Heriberto Zapata Solano
- **Autor:** Roberto Alejandro Chagra Martinez (Código: `208447128`)
- **Cuaderno Principal:** [`DeepMidi_LSTM_Proyecto_Integrador.ipynb`](../DeepMidi_LSTM_Proyecto_Integrador.ipynb)

---

## 1. Introducción y Planteamiento del Problema

La generación automática de música mediante aprendizaje profundo aborda dos grandes paradigmas:

1. **Audio en Crudo (*Raw Audio*):** Modela directamente formas de onda a $22.05\text{–}48\text{ kHz}$, exigiendo decenas de miles de pasos continuos por segundo y un altísimo costo computacional para capturar coherencia armónica de largo plazo.
2. **Música Simbólica (*Symbolic Music*):** Representa la composición como una secuencia de eventos discretos de alto nivel —altura tonal (*pitch*), inicio, duración, velocidad dinámica, compás y tempo— formalizados mediante el estándar **MIDI**.

**DeepMidi-LSTM** formula la composición simbólica como un problema de **Modelado de Lenguaje Autorregresivo (*Next-Token Prediction*)** implementado estrictamente con una red neuronal recurrente **LSTM (*Long Short-Term Memory*)**. Al serializar cada partitura en un flujo causal de eventos **REMI (*Revamped MIDI*)**, el sistema aprende simultáneamente dependencias locales (acordes, conducciones de voz, dinámica) y estructura métrica de mediano plazo (compases, frases y cadencias), culminando en síntesis acústica nativa WAV y herramientas interactivas de visualización.

---

## 2. Representación Simbólica: Piano Roll vs. Tokenización REMI

### 2.1 Limitaciones de las Matrices Piano Roll
En una matriz bidimensional $M \in \mathbb{R}^{T \times 128}$:
- **Alta dispersión (*sparsity*):** Más del 95% de las celdas son ceros, desperdiciando capacidad de memoria en los estados ocultos de una red recurrente.
- **Ausencia de reloj métrico explícito:** La red debe inferir fronteras de compás y subdivisiones rítmicas únicamente contando filas.
- **Reconstrucción ambigua:** Convertir salidas probabilísticas continuas de vuelta a MIDI suele introducir notas fragmentadas o duraciones incoherentes.

### 2.2 El Paradigma Causal REMI (MidiTok v3.1.0)
Siguiendo a Huang & Yang (2020) y Fradet et al. (2021), **REMI** serializa la polifonía en una gramática secuencial explícita:

$$\text{Bar\_None} \longrightarrow \text{TimeSig\_4/4} \longrightarrow \text{Position}_k \longrightarrow \text{Tempo}_v \longrightarrow \text{Pitch}_p \longrightarrow \text{Velocity}_d \longrightarrow \text{Duration}_l$$

| Categoría de Token | Función Musical | Configuración en DeepMidi-LSTM |
|---|---|---|
| `Bar_None` / `TimeSig_*` | Delimitador explícito de compás y métrica | Habilitado (`use_time_signatures=True`) |
| `Position_k` | Posición rítmica dentro del compás | `beat_res={(0, 4): 8, (4, 12): 4}` (8 sub-pulsos/negra) |
| `Pitch_p` | Altura cromática MIDI | Rango piano de 88 teclas `(21, 109)` ($A_0$–$C_8$) |
| `Velocity_d` | Intensidad / dinámica interpretativa | 32 niveles cuantizados (`num_velocities=32`) |
| `Duration_l` | Duración exacta hasta *Note-Off* | Cuantización adaptativa por compás |
| `Chord_*` / `Rest_*` / `Tempo_*` | Acordes, silencios y cambios de tempo | `use_chords=True`, `use_rests=True`, `use_tempos=True` |
| Especiales | Control de secuencia y relleno | `PAD_None = 0`, `BOS_None`, `EOS_None`, `MASK_None` |

Esta configuración genera un vocabulario compacto de **$V = 435$ tokens únicos** y garantiza reconstrucción simbólica reversible (`tokenizer.decode()` reconstruyó las 1,047 notas originales de `clean_Caruso.mid` sin pérdida).

---

## 3. Adquisición, Auditoría y Limpieza del Corpus (Partes 1 a 3)

### 3.1 Estructura del Corpus (`examples/`)
El dataset se organiza jerárquicamente en dos niveles (`[Género]/[Artista]/[archivo.mid]`), con descarga automática desde Google Drive mediante `gdown` en entornos sin archivos locales. La exploración recursiva descubrió **748 archivos MIDI** distribuidos en **11 géneros** y **81 subcarpetas de artistas**:

| Género | Archivos | Género | Archivos | Género | Archivos |
|---|---:|---|---:|---|---:|
| Pop | 323 | Classical | 19 | Soundtracks | 10 |
| Rock | 257 | R&B - Soul | 14 | Latin | 9 |
| Electronic | 47 | Folk Rock | 13 | Hip-Hop | 7 |
| New Age | 37 | Reggae | 12 | **Total** | **748** |

### 3.2 Auditoría Acústica y Simbólica con `symusic` (Parte 2)
Sobre las 748 obras exploradas se verificó:
- **Integridad:** **739 archivos legibles** y **9 corruptos/ilegibles**.
- **Volumen musical:** **3,803,835 notas totales** (media de **5,147.27 notas/obra**); duración media de **238.55 s** (mediana **230.36 s**, rango **16.33–1,548.53 s**).
- **Complejidad instrumental:** **732 archivos multipista** frente a **7 monopista**; **708 obras con más de un instrumento melódico**; **674 piezas con canales de percusión** (91.2%). Instrumentos más frecuentes: *Acoustic Grand Piano* (347), *Electric Bass (finger)* (262) y *String Ensemble 1* (253).
- **Ajuste al registro de piano (MIDI 21–108):** Media de **99.92%** de notas dentro del registro estándar de 88 teclas ($A_0 = 27.5\text{ Hz}$ a $C_8 = 4186.0\text{ Hz}$).

### 3.3 Algoritmo de Limpieza y Selección de Pista Principal (Parte 3)
Para aislar una única pista melódico-armónica limpia por obra se aplican cuatro reglas secuenciales:
1. **Exclusión de percusión:** Descarte de canales con `is_drum == True`, dado que sus códigos de nota representan piezas de batería y no alturas tonales.
2. **Selección de pista instrumental:** Priorización de teclados/pianos (programas General MIDI `0`–`7`, aplicada en **504 obras**, 71.7%); en su ausencia, se selecciona la pista no percusiva con mayor cantidad de notas (**199 obras**, 28.3%).
3. **Umbrales de calidad:** Rechazo de piezas con `< 50` notas, duración `< 5.0` segundos o `> 5.0%` de notas fuera del rango `[21, 108]`.
4. **Normalización:** Exportación monopista en `dataset_midi_clean/clean_<nombre>.mid` conservando *ticks per quarter*, tempos y armaduras de compás.

**Balance de retención verificado:** **703 obras conservadas (94.0%)** y **45 descartadas (6.0%)** — 33 por cantidad insuficiente de notas, 9 por archivo corrupto/ilegible y 3 por exceder el 5% fuera del rango tonal 21–108.

---

## 4. Estadística de Secuencias y Preparación Autorregresiva (Partes 4 a 6)

### 4.1 Distribución de Longitudes REMI (Parte 5)
La tokenización de las 703 obras limpias produjo las siguientes estadísticas:
- **Mínimo:** `227` tokens | **Máximo:** `25,195` tokens
- **Promedio:** `3,653.85` tokens | **Mediana:** `3,094.0` tokens
- **Desviación estándar:** `2,662.31` tokens | **Percentil 95:** `8,334.5` tokens
- **Viabilidad (`>= 64` tokens):** **703 de 703 secuencias utilizables (100%)**.

### 4.2 Partición sin Fuga de Datos y Ventaneo Deslizante (Parte 6)
1. **División a nivel de obra completa (*Zero Data Leakage*):** 80% entrenamiento (**562 canciones**) y 20% validación (**141 canciones**) mediante `SEED = 42`.
2. **Segmentación por ventanas deslizantes (*Sliding Window*):** Ventanas de longitud $L = 256$ con solapamiento del 50% ($\text{stride} = 128$), generando **15,608 ventanas de entrenamiento** y **4,104 ventanas de validación** (**19,712 muestras efectivas** en ejecución CUDA; 19,712 totales en todas las plataformas).
3. **Tensores autorregresivos:** Cada ventana conforma el par desplazado $X = [t_0, \dots, t_{254}]$ y $Y = [t_1, \dots, t_{255}]$ (con relleno `pad_id = 0`), empaquetado en `DataLoader` de forma verificada **`[Batch=32, Seq_Len=255]`**, tipo `torch.int64`, cero valores `NaN` y rango acotado a `[0, 434]`.

---

## 5. Arquitectura del Modelo `MusicLM` e Hiperparámetros (Parte 7)

### 5.1 Formulación Matemática de la Celda LSTM
En cada paso temporal $t$, cada capa LSTM actualiza el estado de celda aditivo $C_t$ y el estado oculto $h_t$ mediante tres compuertas sigmoidales y un candidato hiperbólico (Hochreiter & Schmidhuber, 1997):

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i), \quad \tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t, \quad o_t = \sigma(W_o [h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(C_t)$$

### 5.2 Desglose Dimensional y de Parámetros Entrenables

| Componente | Módulo PyTorch | Transformación de Tensores | Parámetros | % del Total |
|---|---|---|---:|---:|
| Embedding Simbólico | `nn.Embedding(435, 256, padding_idx=0)` | `[B, 255] → [B, 255, 256]` | 111,360 | 2.78% |
| Capa Recurrente LSTM 1 | `nn.LSTM` (capa 0: `256 → 512`) | `[B, 255, 256] → [B, 255, 512]` | 1,576,960 | 39.30% |
| Capa Recurrente LSTM 2 | `nn.LSTM` (capa 1: `512 → 512`, `dropout=0.2`) | `[B, 255, 512] → [B, 255, 512]` | 2,101,248 | 52.36% |
| Proyección de Salida (FC) | `nn.Dropout(0.2) + nn.Linear(512, 435)` | `[B, 255, 512] → [B, 255, 435]` | 223,155 | 5.56% |
| **Total (`MusicLM`)** | — | — | **4,012,723** | **100.00%** |

### 5.3 Optimización, Regularización y Función de Pérdida
Se minimiza la **Entropía Cruzada Categórica** ignorando explícitamente los tokens de relleno (`ignore_index = 0`):

$$\mathcal{L}(\theta) = -\frac{1}{|\mathcal{T}|} \sum_{i \in \mathcal{T}} \log P(y_i \mid x_{\le i}; \theta), \qquad \text{PPL} = e^{\mathcal{L}(\theta)}$$

- **Optimizador:** `AdamW` ($\text{lr} = 1 \times 10^{-3}$, $\text{weight\_decay} = 1 \times 10^{-4}$; Loshchilov & Hutter, 2019).
- **Estabilidad BPTT:** *Gradient Clipping* con $\|\nabla\|_2 \le 1.0$ y `batch_size = 32`.
- **Control adaptativo:** `ReduceLROnPlateau` (`factor = 0.5`, `patience = 2`) y *Early Stopping* (`patience = 5`, máximo 30 épocas) restaurando el punto de control `results/parte7_mejor_modelo.pt`.

### 5.4 Convergencia Registrada (Entrenamiento GPU CUDA)

| Época | Train Loss | Train PPL | Val Loss | Val PPL | Observación |
|---:|---:|---:|---:|---:|---|
| 1 | 2.7591 | 15.78 | 2.1462 | 8.55 | Descenso inicial acelerado |
| 2 | 1.7369 | 5.68 | 1.8692 | 6.48 | Mejora en generalización |
| **3** | **1.4208** | **4.14** | **1.7945** | **6.02** | **Mejor checkpoint guardado** |
| 4 | 1.2558 | 3.51 | 1.7958 | 6.02 | Inicio de meseta en validación |
| 5 | 1.1492 | 3.16 | 1.8038 | 6.07 | Divergencia leve Train/Val |
| 6 | 1.0732 | 2.92 | 1.8282 | 6.22 | Sobreajuste progresivo |
| 7 | 0.9916 | 2.70 | 1.8386 | 6.29 | Reducción de LR vía Plateau |
| 8 | 0.9569 | 2.60 | 1.8532 | 6.38 | *Early Stopping* (paciencia 5) |

---

## 6. Generación Autorregresiva Estrictamente con LSTM (Parte 8)

La inferencia implementa cinco mecanismos acoplados para garantizar musicalidad y estabilidad sin recurrir a heurísticas externas de composición:

1. **Propagación Continua del Estado Oculto $(h_t, c_t)$:** Tras calentar la red con la semilla inicial, cada token muestreado se realimenta junto con la tupla `hidden` actualizada paso a paso.
2. **Penalización por Repetición (*Repetition Penalty*, Keskar et al., 2019):** Sobre el conjunto de tokens únicos emitidos en los últimos $W = 32$ pasos, los logits positivos se dividen entre $\alpha = 1.18$ y los negativos se multiplican por $\alpha$, desincentivando atractores degenerativos mientras conserva motivos rítmicos naturales.
3. **Escalado por Temperatura ($T = 0.72$) y Filtrado Top-$K$ ($K = 15$):** Los logits penalizados se escalan por $1/T$, se enmascara `PAD_None` con $-\infty$, se conservan los 15 candidatos más probables y se muestrea estocásticamente vía `torch.multinomial(softmax)`.
4. **Condicionamiento sin Silencios Iniciales:** La semilla de 32 tokens se extrae desde el primer compás con eventos `Pitch_` de una obra de validación (`clean_O sole mio.mid`) y la partitura resultante se alinea exactamente al segundo `0:00`.
5. **Decodificación Pura REMI:** Los **1,332 tokens** (32 semilla + 1,300 generados) se traducen directamente con `tokenizer.decode()` a `results/mi_primera_composicion_ia_remi.mid`.

---

## 7. Evaluación Musical Objetiva (Parte 9)

Comparación entre la composición generada por el mejor checkpoint (época 3, CUDA) y el promedio de las **141 obras reales de validación**:

| Dimensión | Métrica | Validación Real (N=141) | Composición Generada (LSTM) |
|---|---|---:|---:|
| Modelado de Lenguaje | Cross-Entropy Loss (Época 3) | 1.7945 (Val) | 1.4208 (Train) |
| Modelado de Lenguaje | Perplejidad ($\text{PPL} = e^{\mathcal{L}}$) | 6.02 (Val) | 4.14 (Train) |
| Estructura | Notas / Duración total | — | 348 notas / 164.23 s (~2.74 min) |
| Rango Tonal | Pitch Mínimo / Máximo (MIDI) | 43.34 / 75.93 | 41 / 72 (100% en piano 21–108) |
| Centro Tonal | Pitch Promedio (MIDI) | 60.16 | 60.88 |
| Densidad Temporal | Notas por segundo | 4.50 | 2.12 |
| Articulación | Duración media de nota (s) | 0.561 | 0.406 |
| Densidad Métrica | Notas por compás (4/4) | 9.98 | 5.04 |

---

## 8. Síntesis Acústica WAV y Suite Interactiva (Partes 10 y 11)

### 8.1 Motor de Síntesis Aditiva en NumPy (Parte 10)
Sin dependencias externas de FluidSynth ni bancos SoundFont, cada nota de altura MIDI $p$, duración $d = \max(\text{dur}, 0.05)$ y velocidad $v \in [0, 127]$ se sintetiza a $f_s = 22{,}050\text{ Hz}$ (16 bits PCM mono):

$$f_0 = 440.0 \cdot 2^{\frac{p - 69}{12}}, \qquad s(t) = \sin(2\pi f_0 t) + 0.3\sin(4\pi f_0 t)$$
$$E(t) = \min\!\left(\frac{t}{0.01},\, 1.0\right) \cdot \exp\!\left(-\frac{t}{\max(0.7 d,\, 0.1)}\right), \qquad y(t) = \frac{v}{127} \cdot E(t) \cdot s(t)$$

El buffer acumulado se normaliza al factor anti-*clipping* `0.95` y se exporta a `results/mi_primera_composicion_ia_remi.wav`.

### 8.2 Visualización e Interacción (Parte 11)
- **Piano Roll Dinámico (`results/parte10_piano_roll_generada.png`):** Tiempo vs. altura MIDI (21–108), velocidad codificada en mapa `viridis` y tamaño proporcional a la duración.
- **Partitura Animada Interactiva (`results/partitura_composicion_generada.html`):** Transcripción automática a notación ABC renderizada con `abcjs` y Web Audio API con desplazamiento horizontal e iluminación sincronizada.
- **Visualizador de Notas Cayendo (`results/animacion_composicion_generada.html`):** Aplicación autónoma en HTML5 Canvas + Web Audio API donde las barras caen a velocidad constante y únicamente suena e ilumina la nota que cruza la línea dorada «AHORA», eliminando ambigüedad visual en pasajes rápidos o simultáneos.
- **Panel de Control en Vivo (`ipywidgets`):** Permite ajustar Temperatura, Top-K, *Repetition Penalty*, longitud en tokens y pista semilla para componer, evaluar y escuchar nuevas piezas dentro del cuaderno.

---

## 9. Limitaciones, Trabajo Futuro y Referencias

### Limitaciones Reconocidas
1. El aislamiento a una sola pista simplifica arreglos multipista complejos; en las 199 piezas sin piano, la pista de mayor densidad no siempre coincide con la melodía principal.
2. El sobreajuste a partir de la época 4 refleja la diversidad de 11 géneros sin condicionamiento explícito de estilo.
3. Las métricas musicales son descriptivas y complementan —pero no sustituyen— pruebas perceptuales con oyentes humanos.

### Trabajo Futuro
- Condicionamiento explícito por tokens de género/artista y aumento de datos mediante transposición tonal ($\pm 5$ semitonos).
- Modelado multipista con `use_programs=True` y regularización recurrente (*Variational Dropout*).
- Estudio comparativo controlado (métricas + escucha ciega) frente a arquitecturas *Transformer*.

### Referencias Bibliográficas
1. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
2. Huang, Y.-S., & Yang, Y.-H. (2020). *Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions*. Proc. of the 28th ACM International Conference on Multimedia, 1180–1188.
3. Fradet, N., Briot, J.-P., Chhel, F., El Fallah Seghrouchni, A., & Gutowski, N. (2021). *MidiTok: A Python package for MIDI file tokenization*. Extended Abstracts for the Late-Breaking Demo Session of the 22nd ISMIR Conference.
4. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization (AdamW)*. International Conference on Learning Representations (ICLR).
5. Keskar, N. S., McCann, B., Varshney, L. R., Xiong, C., & Socher, R. (2019). *CTRL: A Conditional Transformer Language Model for Controllable Generation*. arXiv preprint arXiv:1909.05858.
