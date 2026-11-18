# 📖 Diccionario de Funciones y Referencia de la API — DeepMidi-LSTM

Este documento proporciona la especificación técnica exhaustiva de todas las constantes globales, diccionarios de configuración, clases y funciones implementadas en el cuaderno principal del proyecto: **[`DeepMidi_LSTM_Proyecto_Integrador.ipynb`](../DeepMidi_LSTM_Proyecto_Integrador.ipynb)** (Partes 0 a 11 + exportación reproducible).

---

## 📑 Tabla de Contenidos

### Configuración Global y Constantes (Partes 0, 2 y 7)
1. [Constantes Globales del Pipeline](#1-constantes-globales-del-pipeline)
2. [`GM_PROGRAMS` y `HIPERPARAMETROS`](#2-gm_programs-e-hiperparametros)

### Exploración y Auditoría MIDI (Parte 2)
3. [`gm_name`](#3-gm_name)
4. [`inspeccionar_midi`](#4-inspeccionar_midi)

### Preparación de Secuencias y Tensores LSTM (Parte 6)
5. [`strip_clean_prefix`](#5-strip_clean_prefix)
6. [`reparto_estratificado`](#6-reparto_estratificado)
7. [`ventanear`](#7-ventanear)
8. [`VentanasDataset` (Clase)](#8-ventanasdataset-clase)

### Modelo Neuronal LSTM e Inferencia Autorregresiva (Partes 7 y 8)
9. [`MusicLM` (Clase)](#9-musiclm-clase)
10. [`generar_musica_tokens`](#10-generar_musica_tokens)

### Evaluación Objetiva y Síntesis Acústica (Partes 9 y 10)
11. [`metricas_score`](#11-metricas_score)
12. [`sintetizar_audio_midi`](#12-sintetizar_audio_midi)

### Visualización, Reproductores Animados e Interfaz Interactiva (Parte 11)
13. [`plot_piano_roll_from_score`](#13-plot_piano_roll_from_score)
14. [`score_to_abc`](#14-score_to_abc)
15. [`player_page`](#15-player_page)
16. [`construir_animacion_notas`](#16-construir_animacion_notas)
17. [`on_generar_click` (Callback `ipywidgets`)](#17-on_generar_click-callback-ipywidgets)

---

## 1. Constantes Globales del Pipeline

Definidas en la **Parte 0 (Celda 02)** para garantizar determinismo, resolución dinámica de rutas (Google Colab o entorno local) y consistencia en todo el flujo:

| Constante | Valor | Descripción |
|---|---|---|
| `SEED` | `42` | Semilla determinista aplicada a `random`, `numpy` y `torch`. |
| `URL_PUBLIC_EXAMPLES` | Google Drive URL | Enlace público de respaldo para descarga automática de `examples/` vía `gdown`. |
| `EXAMPLES_DIR` / `RAW_DIR` | `<DRIVE_PROYECTO>/examples` | Directorio raíz del corpus MIDI estructurado (748 archivos, 11 géneros). |
| `CLEAN_DIR` | `<DRIVE_PROYECTO>/dataset_midi_clean` | Destino de las pistas depuradas monopista (703 archivos `clean_*.mid`). |
| `RESULTS_DIR` | `<DRIVE_PROYECTO>/results` | Carpeta de artefactos de salida (`.pt`, `.npz`, `.json`, `.png`, `.mid`, `.wav`, `.html`). |
| `MIN_PITCH`, `MAX_PITCH` | `21`, `108` | Registro estándar de piano de 88 teclas ($A_0$ a $C_8$). |
| `MIN_NOTES_THRESHOLD` | `50` | Mínimo de notas exigido en la pista seleccionada. |
| `MIN_DURATION_SECONDS` | `5.0` | Duración mínima en segundos de cada obra conservada. |
| `MAX_OUT_OF_RANGE_PCT` | `5.0` | Porcentaje máximo tolerado de notas fuera del rango `[21, 108]`. |
| `TARGET_PROGRAMS` | `set(range(0, 8))` | Programas General MIDI priorizados (pianos acústicos y eléctricos GM 0–7). |
| `MIN_TOKENS_USABLE` | `64` | Longitud mínima de secuencia REMI para ser utilizable en entrenamiento. |
| `MAX_LEN` | `256` | Longitud de ventana deslizante ($X$ e $Y$ de longitud `255`). |
| `SPLIT_TRAIN` | `0.8` | Proporción de división a nivel de canción (80% entrenamiento / 20% validación). |
| `BATCH_SIZE` | `32` | Tamaño de mini-lote de los `DataLoader` de PyTorch. |

---

## 2. `GM_PROGRAMS` e `HIPERPARAMETROS`

- **`GM_PROGRAMS` (`dict[int, str]`, Parte 2 — Celda 06):** Tabla completa de los 128 instrumentos del estándar General MIDI (`0: "Acoustic Grand Piano"` hasta `127: "Gunshot"`).
- **`HIPERPARAMETROS` (`dict[str, int | float]`, Parte 7 — Celda 18):** Diccionario centralizado de arquitectura, optimización e inferencia:
  ```python
  HIPERPARAMETROS = {
      "embedding_dim": 256, "hidden_dim": 512, "num_layers": 2, "dropout": 0.2,
      "batch_size": 32, "max_len": 256, "stride": 128,
      "epochs": 30, "patience": 5, "learning_rate": 1e-3, "weight_decay": 1e-4,
      "grad_clip": 1.0, "scheduler_factor": 0.5, "scheduler_patience": 2,
      "pasos_a_generar": 1300, "temperatura": 0.72, "top_k": 15,
      "repetition_penalty": 1.18, "ventana_repeticion": 32,
  }
  ```

---

## 3. `gm_name`

```python
def gm_name(program: int) -> str
```

- **Ubicación:** Parte 2 (Celda 06).
- **Descripción:** Traduce un número de programa General MIDI (`0`–`127`) a su nombre instrumental legible usando `GM_PROGRAMS`.
- **Parámetros:**
  - `program` (`int`): Código de programa MIDI de una pista no percusiva.
- **Retorno:** `str` con el nombre del instrumento (p. ej., `"Acoustic Grand Piano"`), o `"Programa GM <program>"` si el código no está mapeado.

---

## 4. `inspeccionar_midi`

```python
def inspeccionar_midi(file_path: str | Path) -> dict
```

- **Ubicación:** Parte 2 (Celda 06).
- **Descripción:** Audita simbólica y acústicamente un archivo MIDI individual mediante el motor nativo C++ de `symusic`, extrayendo métricas de duración, densidad de notas, pistas, percusión, instrumentación General MIDI y ajuste al registro de piano de 88 teclas (`21`–`108`).
- **Parámetros:**
  - `file_path` (`str | Path`): Ruta del archivo `.mid` / `.midi` a inspeccionar.
- **Retorno:** `dict` con las claves:
  - `status` (`str`): `"OK"` si la carga fue exitosa o `"ERROR"` si el archivo está corrupto/ilegible.
  - `error_msg` (`str`): Mensaje de excepción en caso de fallo (cadena vacía si `"OK"`).
  - `duration_sec` (`float | None`): Duración total en segundos redondeada a 2 decimales.
  - `total_notes` (`int`): Suma total de notas en todas las pistas.
  - `num_tracks` (`int`): Cantidad de pistas en la partitura.
  - `is_multi_track` (`bool`): `True` si `num_tracks > 1`.
  - `has_drums` (`bool`): `True` si alguna pista posee `is_drum == True`.
  - `programs` (`list[int]`): Lista ordenada de códigos GM únicos presentes en pistas no percusivas.
  - `program_names` (`list[str]`): Nombres legibles correspondientes mediante `gm_name()`.
  - `min_pitch`, `max_pitch` (`int | None`): Altura MIDI mínima y máxima detectada.
  - `pct_in_range_21_108` (`float`): Porcentaje de notas dentro de `[MIN_PITCH, MAX_PITCH]`.
- **Manejo de errores:** Captura cualquier `Exception` de lectura sin interrumpir el escaneo masivo del corpus (identificando los 9 archivos corruptos en `examples/`).

---

## 5. `strip_clean_prefix`

```python
def strip_clean_prefix(name: str) -> str
```

- **Ubicación:** Parte 6 (Celda 16).
- **Descripción:** Remueve de forma iterativa prefijos `"clean_"` redundantes en nombres de archivos MIDI limpios para garantizar correspondencia unívoca entre las listas de validación y los archivos almacenados en `CLEAN_DIR`.
- **Parámetros:**
  - `name` (`str`): Nombre de archivo (p. ej., `"clean_O sole mio.mid"`).
- **Retorno:** `str` con el nombre base normalizado (`"O sole mio.mid"`).

---

## 6. `reparto_estratificado`

```python
def reparto_estratificado(
    nombres: list[str],
    frac_val: float = 1 - SPLIT_TRAIN,
    semilla: int = SEED,
) -> tuple[list[int], list[int]]
```

- **Ubicación:** Parte 6 (Celda 16).
- **Descripción:** Ejecuta una partición reproducible a nivel de obras completas (80% entrenamiento / 20% validación) garantizando **cero fuga de datos (*zero data leakage*)** entre segmentos de una misma canción.
- **Parámetros:**
  - `nombres` (`list[str]`): Nombres de las secuencias utilizables (703 obras).
  - `frac_val` (`float`): Fracción asignada a validación (por defecto `0.2`).
  - `semilla` (`int`): Semilla para `random.Random` (por defecto `42`).
- **Retorno:** Tupla `(train_idx, val_idx)` con listas ordenadas de índices disjuntos (562 entrenamiento / 141 validación).

---

## 7. `ventanear`

```python
def ventanear(
    secuencias: list[list[int]],
    max_len: int = MAX_LEN,
    stride: int = MAX_LEN // 2,
    pad_id: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]
```

- **Ubicación:** Parte 6 (Celda 16).
- **Descripción:** Segmenta cada secuencia completa de IDs REMI mediante una **ventana deslizante (*sliding window*)** de longitud `max_len` con desplazamiento `stride` (solapamiento del 50%) y construye pares autorregresivos desfasados un paso temporal ($X_t \to Y_t = X_{t+1}$).
- **Parámetros:**
  - `secuencias` (`list[list[int]]`): Lista de secuencias de IDs de tokens.
  - `max_len` (`int`): Tamaño de ventana (por defecto `256`, produciendo tensores de longitud `255`).
  - `stride` (`int`): Paso entre ventanas consecutivas (por defecto `128`).
  - `pad_id` (`int`): Identificador de relleno para fragmentos residuales (por defecto `0`).
- **Retorno:** Tupla `(x_list, y_list)` donde cada elemento es un tensor `torch.int64` de forma `[255]`. En el corpus global genera 15,608 ventanas de entrenamiento y 4,104 de validación.

---

## 8. `VentanasDataset` (Clase)

```python
class VentanasDataset(torch.utils.data.Dataset)
```

- **Ubicación:** Parte 6 (Celda 16).
- **Descripción:** Subclase de `torch.utils.data.Dataset` que encapsula las listas de tensores `(x_list, y_list)` para su carga eficiente por mini-lotes mediante `DataLoader`.
- **Métodos:**
  - `__init__(self, x_list: list[torch.Tensor], y_list: list[torch.Tensor]) -> None`: Almacena referencias a los tensores de entrada y objetivo.
  - `__len__(self) -> int`: Devuelve el número total de ventanas efectivas.
  - `__getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]`: Retorna el par `(x[idx], y[idx])` de tipo `torch.int64` y forma `[255]`.

---

## 9. `MusicLM` (Clase)

```python
class MusicLM(torch.nn.Module)
```

- **Ubicación:** Parte 7 (Celda 18).
- **Descripción:** Modelo de lenguaje musical autorregresivo sustentado estrictamente en una **Red Neuronal Recurrente LSTM de 2 capas apiladas** con **4,012,723 parámetros entrenables**.

### Constructor
```python
def __init__(
    self,
    vocab_size: int,
    embedding_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 2,
    dropout: float = 0.2,
    pad_idx: int = 0,
)
```
- **Capas internas:**
  - `self.embedding`: `nn.Embedding(435, 256, padding_idx=0)` — 111,360 parámetros (2.78%).
  - `self.lstm`: `nn.LSTM(256, 512, num_layers=2, batch_first=True, dropout=0.2)` — 3,678,208 parámetros (91.66%).
  - `self.dropout`: `nn.Dropout(0.2)` aplicado tras el embedding y antes de la proyección lineal.
  - `self.fc`: `nn.Linear(512, 435)` — 223,155 parámetros (5.56%).

### Método `forward`
```python
def forward(
    self,
    x: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
```
- **Parámetros:**
  - `x` (`torch.Tensor`): Tensor `torch.int64` de forma `[Batch, Seq_Len]`.
  - `hidden` (`tuple | None`): Estado oculto y estado de celda previos `(h_t, c_t)` de forma `[num_layers, Batch, hidden_dim]`.
- **Retorno:** Tupla `(logits, hidden)` donde `logits` tiene forma `[Batch, Seq_Len, vocab_size]` y `hidden` contiene los estados actualizados para propagación continua en inferencia.

---

## 10. `generar_musica_tokens`

```python
def generar_musica_tokens(
    modelo: nn.Module,
    semilla_ids: list[int],
    pasos_a_generar: int = 1300,
    temperatura: float = 0.72,
    top_k: int = 15,
    repetition_penalty: float = 1.18,
    ventana_repeticion: int = 32,
    pad_id: int = 0,
) -> list[int]
```

- **Ubicación:** Parte 8 (Celda 21).
- **Descripción:** Ejecuta generación musical autorregresiva estrictamente con la LSTM propagando continuamente el estado oculto `(h_t, c_t)` sin truncar la memoria recurrente.
- **Parámetros:**
  - `modelo`: Instancia entrenada de `MusicLM` en modo `.eval()`.
  - `semilla_ids` (`list[int]`): Secuencia de calentamiento (32 tokens iniciando en el primer compás con notas efectivas para evitar silencios iniciales).
  - `pasos_a_generar` (`int`): Tokens nuevos a muestrear (por defecto `1300`, totalizando `1332` tokens).
  - `temperatura` (`float`): Factor de escalado de entropía sobre logits (por defecto `0.72`).
  - `top_k` (`int`): Número de candidatos más probables retenidos antes del muestreo multinomial (por defecto `15`).
  - `repetition_penalty` (`float`): Factor $\alpha$ de penalización sobre tokens emitidos recientemente según Keskar et al. (2019) (por defecto `1.18`).
  - `ventana_repeticion` (`int`): Número de pasos previos considerados para penalizar repeticiones (por defecto `32`).
  - `pad_id` (`int`): ID de relleno enmascarado con $-\infty$ en cada paso.
- **Retorno:** `list[int]` con la secuencia completa de IDs REMI lista para decodificación pura con `tokenizer.decode([tokens])`.

---

## 11. `metricas_score`

```python
def metricas_score(score: symusic.Score) -> dict[str, int | float]
```

- **Ubicación:** Parte 9 (Celda 23).
- **Descripción:** Calcula indicadores musicales objetivos sobre cualquier objeto `symusic.Score` (composición generada o pieza real del conjunto de validación) para comparación cuantitativa.
- **Parámetros:**
  - `score` (`symusic.Score`): Partitura simbólica MIDI.
- **Retorno:** `dict` con las claves:
  - `n_notas` (`int`): Número total de notas.
  - `duracion_s` (`float`): Duración total en segundos.
  - `pitch_min`, `pitch_max` (`int`): Extremos tonales MIDI.
  - `pitch_promedio` (`float`): Altura MIDI media.
  - `notas_por_segundo` (`float`): Densidad temporal de eventos.
  - `duracion_nota_promedio_s` (`float`): Duración media por nota en segundos.
  - `notas_por_compas` (`float`): Densidad rítmica promedio estimada en compases de 4/4.

---

## 12. `sintetizar_audio_midi`

```python
def sintetizar_audio_midi(
    score: symusic.Score,
    ruta_wav_salida: str | Path,
    sample_rate: int = 22050,
) -> None
```

- **Ubicación:** Parte 10 (Celda 25).
- **Descripción:** Motor nativo de síntesis acústica en `NumPy` y Python puro que transforma un `symusic.Score` en un archivo de audio WAV mono de **16 bits PCM** a **22,050 Hz**, sin requerir FluidSynth ni SoundFonts externos.
- **Parámetros:**
  - `score` (`symusic.Score`): Partitura a sintetizar.
  - `ruta_wav_salida` (`str | Path`): Ruta destino del archivo `.wav`.
  - `sample_rate` (`int`): Frecuencia de muestreo en Hz (por defecto `22050`).
- **Proceso interno:**
  1. Convierte cada altura MIDI $p$ en frecuencia $f = 440 \cdot 2^{(p-69)/12}$.
  2. Superpone fundamental y segundo armónico: $\sin(2\pi f t) + 0.3 \sin(4\pi f t)$.
  3. Modula con envolvente exponencial ADSR (`ataque = min(t / 0.01, 1.0)`, `decaimiento = exp(-t / max(duracion * 0.7, 0.1))`) escalada por `velocity / 127.0`.
  4. Normaliza al 95% de amplitud máxima para prevenir saturación (*clipping*) y escribe en formato PCM int16 con `wave`.

---

## 13. `plot_piano_roll_from_score`

```python
def plot_piano_roll_from_score(
    score: symusic.Score,
    titulo: str = "Piano Roll - Composición generada por IA",
) -> None
```

- **Ubicación:** Parte 11 (Celda 27).
- **Descripción:** Genera y guarda el gráfico de Piano Roll (`results/parte10_piano_roll_generada.png`) proyectando el tiempo en segundos contra la altura MIDI (`21`–`108`), codificando la velocidad dinámica mediante el mapa de color `viridis` y la duración mediante el tamaño de cada marcador.
- **Parámetros:**
  - `score` (`symusic.Score`): Composición simbólica.
  - `titulo` (`str`): Título principal de la figura.

---

## 14. `score_to_abc`

```python
def score_to_abc(score: symusic.Score) -> str
```

- **Ubicación:** Parte 11 (Celda 28).
- **Descripción:** Convierte cronológicamente las notas de un `symusic.Score` a una cadena en **notación musical ABC** estándar (con cuantización de duraciones en fracciones de negra y barras de compás `|`), utilizada por el motor `abcjs` para renderizar el pentagrama interactivo.
- **Parámetros:**
  - `score` (`symusic.Score`): Partitura a transcribir.
- **Retorno:** `str` con el cuerpo ABC listo para inyectar en `ABCJS.renderAbc()`.

---

## 15. `player_page`

```python
def player_page(wav_bytes: bytes, abc_notation: str, title: str) -> str
```

- **Ubicación:** Parte 11 (Celda 28).
- **Descripción:** Construye una aplicación web autocontenida (HTML5 + CSS + JavaScript + `abcjs` 6.2.2 + Web Audio API) que incrusta el audio WAV en Base64, renderiza la partitura SVG en desplazamiento horizontal continuo e ilumina en rojo cada nota de forma sincronizada con la reproducción. Guarda una copia independiente en `results/partitura_composicion_generada.html`.
- **Parámetros:**
  - `wav_bytes` (`bytes`): Contenido binario del archivo `.wav` sintetizado.
  - `abc_notation` (`str`): Cadena producida por `score_to_abc()`.
  - `title` (`str`): Título mostrado en el reproductor.
- **Retorno:** `str` con etiqueta `<iframe>` lista para `IPython.display.HTML()`.

---

## 16. `construir_animacion_notas`

```python
def construir_animacion_notas(score: symusic.Score, titulo: str) -> str | None
```

- **Ubicación:** Parte 11 (Celda 29).
- **Descripción:** Genera un visualizador interactivo independiente tipo *Synthesia* en **HTML5 Canvas + Web Audio API** (`results/animacion_composicion_generada.html`). Las notas caen a velocidad constante por columnas según su altura tonal y disparan un oscilador triangular con envolvente exactamente al cruzar la línea dorada «AHORA», mostrando en vivo el nombre de la nota en español, su frecuencia en Hz y su número MIDI. Incluye selector de velocidad (`0.5×`, `1×`, `1.5×`), barra *seek* y controles de reproducción.
- **Parámetros:**
  - `score` (`symusic.Score`): Partitura a animar.
  - `titulo` (`str`): Encabezado de la página HTML generada.
- **Retorno:** `str` con el documento HTML completo, o `None` si la partitura no contiene notas.

---

## 17. `on_generar_click` (Callback `ipywidgets`)

```python
def on_generar_click(_: ipywidgets.Button) -> None
```

- **Ubicación:** Parte 11 (Celda 30).
- **Descripción:** Manejador de eventos del **Panel de Control Interactivo DeepMidi-LSTM** (`ipywidgets`). Lee en tiempo real los widgets de pista semilla (`w_semilla`), temperatura (`w_temp`: `0.30`–`1.30`), Top-K (`w_topk`: `5`–`50`), penalización por repetición (`w_rep`: `1.00`–`1.50`) y longitud (`w_pasos`: `200`–`1500`), ejecuta `generar_musica_tokens()`, decodifica con `MidiTok`, alinea al instante `0:00`, exporta `results/composicion_interfaz_lstm.mid` y `results/composicion_interfaz_lstm.wav`, imprime las métricas objetivas con `metricas_score()` y despliega el reproductor `Audio`.
