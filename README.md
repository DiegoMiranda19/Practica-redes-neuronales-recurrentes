# Predicción de Retornos Logarítmicos de Bitcoin con Redes Neuronales

> **Proyecto académico** — Tópicos de Minería de Datos  
> Universidad Externado de Colombia | 2026-I

## Resumen

Este proyecto investiga si las redes neuronales recurrentes pueden predecir los retornos logarítmicos diarios de Bitcoin (BTC-USD) utilizando exclusivamente datos históricos de precios. Se implementa un pipeline completo de ciencia de datos — desde la adquisición de datos hasta la evaluación estadística rigurosa — y se comparan múltiples arquitecturas de deep learning.

**Hallazgo principal:** Todas las arquitecturas evaluadas (LSTM, GRU, CNN-LSTM, Transformer) reducen significativamente el error cuadrático respecto al baseline naive, pero **ninguna logra predecir la dirección del precio** mejor que el azar. Este resultado es consistente con la Hipótesis de Mercado Eficiente en su forma débil.

## Estructura del repositorio

```
├── data/
│   └── BTC-USD.csv                              # Datos cacheados (se genera automáticamente)
├── models/
│   ├── lstm_btc_lb90.keras                      # Modelo entrenado
│   ├── scaler.pkl                               # StandardScaler ajustado
│   └── config.pkl                               # Configuración e hiperparámetros
├── rnr_retornos_bitcoin_mejorado.ipynb          # Notebook principal: LSTM + búsqueda de ventana
├── rnr_comparacion_arquitecturas.ipynb          # Notebook comparativo: LSTM vs GRU vs CNN-LSTM vs Transformer
├── README.md
└── requirements.txt
```

## Datos

| Característica | Detalle |
|---|---|
| Activo | BTC-USD (Bitcoin vs Dólar) |
| Fuente | Yahoo Finance vía `yfinance` |
| Periodo | Enero 2015 — Abril 2026 |
| Frecuencia | Diaria |
| Observaciones | ~4,117 retornos logarítmicos |
| Variable objetivo | Log-return diario: `ln(P_t / P_{t-1})` |

## Metodología

### Pipeline de datos

1. **Descarga** de precios de cierre diarios desde Yahoo Finance (con fallback a CSV local)
2. **Cálculo de retornos logarítmicos** y eliminación del NaN inicial
3. **Split temporal** 80% / 10% / 10% (Train / Validación / Test) — sin mezcla aleatoria para respetar la causalidad temporal
4. **Escalado** con `StandardScaler` ajustado únicamente sobre el conjunto de entrenamiento para evitar data leakage
5. **Ventanas deslizantes** de `LOOK_BACK` días como input para las redes neuronales

### Notebook 1: Modelo LSTM con búsqueda de ventana

Se entrena una LSTM unidireccional con regularización (Dropout + recurrent_dropout) y se busca la ventana óptima entre {10, 20, 30, 40, 50, 60, 70, 80, 90} días, evaluando cada una en el conjunto de validación.

**Arquitectura:**
```
Input(LOOK_BACK, 1) → LSTM(32) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16, relu) → Dense(1)
```

**Resultados (Test Set, LOOK_BACK=90):**

| Métrica | LSTM | Baseline (naive) |
|---|---|---|
| RMSE | 0.0228 | 0.0331 |
| MAE | 0.0158 | 0.0229 |
| R² | −0.007 | −1.122 |
| Directional Accuracy | 46.9% | 51.9% |
| DM test (p-valor) | 0.005 | — |

### Notebook 2: Comparación de arquitecturas

Se comparan cuatro arquitecturas bajo condiciones controladas (mismos datos, split, scaler, LOOK_BACK=60, callbacks):

| Arquitectura | Parámetros | RMSE | Dir. Acc | R² | DM vs baseline (p) |
|---|---|---|---|---|---|
| LSTM | 13,217 | 0.0225 | 49.4% | −0.006 | 0.003 |
| GRU | 10,241 | 0.0224 | 49.1% | 0.000 | 0.004 |
| CNN-LSTM | 8,497 | 0.0225 | 48.9% | −0.005 | 0.003 |
| Transformer | 9,185 | 0.0225 | 48.6% | −0.007 | 0.003 |
| Baseline (naive) | 0 | 0.0325 | 50.9% | −1.110 | — |

El test de Diebold-Mariano entre pares de arquitecturas arroja p-valores entre 0.59 y 0.96 — **ninguna diferencia significativa** entre modelos.

## Resultados clave

### Lo que los modelos SÍ logran
- Reducen el RMSE ~31% respecto al baseline naive
- La mejora es estadísticamente significativa (Diebold-Mariano, p < 0.005)
- Capturan parcialmente la estructura de volatilidad de los retornos

### Lo que los modelos NO logran
- Ninguna arquitectura predice la dirección del precio mejor que el azar (test binomial, p > 0.60)
- El R² es esencialmente cero en todos los modelos
- Las cuatro arquitecturas convergen a la misma estrategia: predecir retornos cercanos a cero

### Interpretación

Las redes neuronales aprenden a "suavizar" sus predicciones hacia la media (≈ 0), lo cual reduce el MSE pero sacrifica la capacidad direccional. Esto es consistente con la **Hipótesis de Mercado Eficiente** en su forma débil: los precios históricos no contienen información explotable para predecir retornos futuros de BTC.

El resultado no depende de la arquitectura — LSTM, GRU, CNN-LSTM y Transformer producen predicciones estadísticamente indistinguibles, confirmando que el cuello de botella es la **naturaleza de los datos**, no la herramienta.

## Evaluación estadística

Este proyecto va más allá de las métricas estándar e implementa:

- **Test de Diebold-Mariano** — Compara la capacidad predictiva de dos modelos con significancia estadística
- **Test binomial** — Evalúa si el directional accuracy supera significativamente el 50% (azar)
- **Comparación entre pares** — DM test entre todas las combinaciones de arquitecturas

## Requisitos

```
numpy
pandas
matplotlib
scikit-learn
tensorflow>=2.15
yfinance
scipy
```

### Instalación

```bash
git clone https://github.com/<usuario>/rnr-retornos-bitcoin.git
cd rnr-retornos-bitcoin
pip install -r requirements.txt
```

### Ejecución

1. Ejecutar `rnr_retornos_bitcoin_mejorado.ipynb` — modelo LSTM completo con búsqueda de ventana
2. Ejecutar `rnr_comparacion_arquitecturas.ipynb` — comparación de 4 arquitecturas

Los datos se descargan automáticamente desde Yahoo Finance en la primera ejecución.

## Posibles líneas futuras

| Estrategia | Idea |
|---|---|
| Features exógenas | Agregar volumen, RSI, MACD, Bollinger Bands |
| Datos alternativos | Sentimiento de redes sociales, datos on-chain |
| Clasificación binaria | Predecir UP/DOWN optimizando accuracy en vez de MSE |
| Horizonte multi-paso | Retorno acumulado a 5–10 días para reducir ruido |
| Ensemble | Combinar predicciones de múltiples arquitecturas |

## Autor

Estudiante de la Universidad Externado de Colombia  
Materia: Tópicos de Minería de Datos — Tercer Semestre, 2026-I

## Licencia

Este proyecto es de uso académico. Los datos de precios provienen de Yahoo Finance y están sujetos a sus términos de uso.
