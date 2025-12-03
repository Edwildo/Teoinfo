# Fourier Draw: Representación de Contornos mediante Series Complejas y Epiciclos

Este proyecto implementa el artículo **"Dibujando con la Transformada de Fourier: representación de contornos mediante series complejas y epiciclos"**. El objetivo es tomar un contorno 2D, convertirlo en una señal compleja periódica, calcular sus coeficientes de Fourier mediante DFT/FFT, y luego reconstruir y animar el contorno usando una suma finita de armónicos (epiciclos).

## Descripción

El sistema permite:

1. **Representar un contorno como una señal compleja**: Un contorno 2D se convierte en una señal compleja \( z_n = x_n + i y_n \), donde cada punto del contorno se representa como un número complejo.

2. **Calcular coeficientes de Fourier**: Se utiliza la Transformada Discreta de Fourier (DFT) o su versión rápida (FFT) para obtener los coeficientes de Fourier de la señal compleja.

3. **Seleccionar un subconjunto de coeficientes**: Se selecciona un número \( K \) de armónicos basándose en criterios de energía acumulada, permitiendo reconstruir el contorno con un número reducido de términos.

4. **Reconstruir el contorno**: Se reconstruye el contorno original usando solo los \( K \) coeficientes seleccionados y se mide el error de reconstrucción (MSE, PSNR, energía acumulada).

5. **Visualizar resultados**: Se muestra el contorno original comparado con el reconstruido.

6. **Animar epiciclos**: Se genera una animación que muestra cómo una serie de círculos rotando (epiciclos) se combinan para dibujar el contorno en tiempo real.

## Conceptos Fundamentales

### Señal Compleja del Contorno

Un contorno 2D definido por puntos \( (x_n, y_n) \) se convierte en una señal compleja:
\[
z_n = x_n + i y_n
\]

Esta representación permite tratar el contorno como una función periódica en el plano complejo.

### DFT/FFT

La Transformada Discreta de Fourier (DFT) descompone la señal compleja en sus componentes de frecuencia:
\[
C_k = \frac{1}{N} \sum_{n=0}^{N-1} z_n e^{-2\pi i k n / N}
\]

donde \( C_k \) son los coeficientes de Fourier. El módulo \( |C_k| \) representa la amplitud del armónico de frecuencia \( k \), y el argumento \( \arg(C_k) \) representa su fase.

### Selección de K por Energía

La energía total del espectro es:
\[
E_{\text{total}} = \sum_{k} |C_k|^2
\]

Para un umbral \( \tau \) (por ejemplo, 0.95), se selecciona el menor \( K \) tal que la energía acumulada de las frecuencias en el rango \( [-K, K] \) sea al menos \( \tau \times E_{\text{total}} \):
\[
\sum_{k=-K}^{K} |C_k|^2 \geq \tau \times E_{\text{total}}
\]

### Métricas de Calidad

- **MSE (Error Cuadrático Medio)**: Mide el error promedio entre el contorno original y el reconstruido.
  \[
  \text{MSE} = \frac{1}{N} \sum_{n=0}^{N-1} |z_n - \hat{z}_n|^2
  \]

- **PSNR (Peak Signal-to-Noise Ratio)**: Mide la calidad de la reconstrucción en decibeles.
  \[
  \text{PSNR} = 10 \log_{10} \left( \frac{\text{MAX}^2}{\text{MSE}} \right)
  \]

### Animación por Epiciclos

La reconstrucción del contorno se puede visualizar como una serie de círculos (epiciclos) que rotan a diferentes frecuencias. Cada epiciclo tiene:
- **Radio**: \( |C_k| \) (amplitud del coeficiente)
- **Frecuencia**: \( k \) (velocidad de rotación)
- **Fase inicial**: \( \arg(C_k) \)

El punto final del último epiciclo traza el contorno reconstruido.

## Instalación

### Requisitos

- Python 3.11 o superior
- pip

### Instalación del Proyecto

1. Clona o descarga este repositorio.

2. Instala el proyecto en modo desarrollo:
```bash
pip install -e .
```

Esto instalará las dependencias necesarias (`numpy` y `matplotlib`).

### Instalación de Dependencias de Desarrollo (Opcional)

Para ejecutar los tests:
```bash
pip install -e ".[dev]"
```

## Uso

### Formato de Datos de Entrada

El sistema soporta dos formatos de entrada:

#### 1. Archivo de Texto
Archivo con dos columnas (x, y) separadas por espacios.

#### 2. Imágenes (PNG, JPG, BMP, GIF, TIFF)
El sistema extrae contornos automáticamente desde imágenes.

**Mejores resultados con:**
- Formas geométricas simples (círculo, cuadrado, estrella)
- Alto contraste (objeto claro sobre fondo oscuro o viceversa)
- Un solo objeto principal
- Fondo uniforme

Ver `MEJORES_IMAGENES.md` para más detalles.

### Ejecución del Script Principal

El script principal `scripts/run_example.py` permite ejecutar el pipeline completo:

**Con archivo de texto:**
```bash
python scripts/run_example.py --input data/mi_contorno.txt --samples 1024 --tau 0.97 --animate
```

**Con imagen:**
```bash
python scripts/run_example.py --input mi_imagen.png --samples 512 --tau 0.97 --animate
```

El sistema detecta automáticamente si el archivo es una imagen o texto según la extensión.

#### Parámetros Disponibles

- `--input`: Ruta al archivo de contorno (requerido)
- `--samples`: Número de muestras para re-muestreo (default: 1024)
- `--tau`: Umbral de energía acumulada para selección de K (default: 0.95)
- `--k-max`: Número máximo de frecuencias a usar (opcional; si no se especifica, se selecciona por energía)
- `--n-time`: Número de puntos de tiempo para reconstrucción (default: 1000)
- `--animate`: Flag para generar animación de epiciclos
- `--save-animation`: Ruta donde guardar la animación (.mp4 o .gif)
- `--fps`: Frames por segundo para la animación (default: 30)

#### Ejemplos de Uso

**Formas geométricas simples (recomendado):**
```bash
py scripts/run_example.py --input data/circulo_imagen.png --samples 256 --tau 0.95 --animate
```

**Imágenes más complejas:**
```bash
py scripts/run_example.py --input mi_imagen.png --samples 512 --tau 0.97 --animate --save-animation resultado.gif
```

**Ajustar umbral si es necesario:**
```bash
py scripts/run_example.py --input mi_imagen.png --threshold 150 --samples 512 --animate
```

### Uso como Módulo Python

También puedes usar el módulo directamente en tu código Python:

```python
from fourier_draw import (
    Contour,
    compute_fourier_coefficients,
    reconstruct_from_coeffs,
    select_k_by_energy,
    mse,
    psnr,
    animate_epicycles,
    load_contour_from_txt,
)

# Cargar contorno
contour = load_contour_from_txt("data/mi_contorno.txt")

# Cerrar y re-muestrear
contour_closed = contour.closed()
contour_resampled = contour_closed.resample_by_arclength(1024)

# Calcular coeficientes de Fourier
z = contour_resampled.complex_signal
coeffs = compute_fourier_coefficients(z)

# Seleccionar K por energía
k_selected = select_k_by_energy(coeffs, tau=0.95)

# Reconstruir
z_reconstructed = reconstruct_from_coeffs(coeffs, k_selected, n_time=1000)

# Calcular métricas
error = mse(z, z_reconstructed)
quality = psnr(z, z_reconstructed)

print(f"K seleccionado: {k_selected}")
print(f"MSE: {error:.6e}")
print(f"PSNR: {quality:.2f} dB")

# Animar
anim = animate_epicycles(coeffs, k_selected, n_time=1000)
```

## Estructura del Proyecto

```
fourier_draw/
├── __init__.py          # Inicialización del módulo
├── config.py            # Configuración por defecto
├── contour.py           # Representación y procesamiento de contornos
├── fourier.py           # DFT/FFT, selección de coeficientes, reconstrucción
├── metrics.py           # Métricas de evaluación (MSE, PSNR, energía)
├── animation.py         # Visualización y animación de epiciclos
└── io_utils.py          # Carga/guardado de contornos

scripts/
└── run_example.py       # Script principal de demostración / CLI

tests/
├── test_contour.py      # Tests para contour.py
├── test_fourier.py      # Tests para fourier.py
└── test_metrics.py      # Tests para metrics.py

pyproject.toml           # Configuración del proyecto y dependencias
README.md                # Este archivo
```

## Tests

Para ejecutar los tests:

```bash
pytest
```

Para ejecutar tests con cobertura:

```bash
pytest --cov=fourier_draw --cov-report=html
```

## Visualización

### Contorno Original vs Reconstruido

El script genera una figura que muestra:
- **Línea azul sólida**: Contorno original
- **Línea roja discontinua**: Contorno reconstruido
- **Puntos marcados**: Puntos iniciales de cada contorno

### Animación de Epiciclos

La animación muestra:
- **Círculos grises**: Los epiciclos (círculos rotando)
- **Líneas verdes**: Vectores desde el centro de cada epiciclo hasta el punto en el círculo
- **Línea roja**: El trazado del contorno reconstruido a medida que se dibuja
- **Punto rojo**: El punto actual que recorre el contorno
- **Línea azul discontinua (fondo)**: El contorno completo de referencia

La animación muestra cómo los epiciclos se combinan para dibujar el contorno, empezando con los epiciclos más grandes (mayor amplitud) y añadiendo los más pequeños para refinar la forma.

## Optimización

El proyecto está optimizado para trabajar eficientemente con contornos de 1,000 a 5,000 puntos. Para contornos más grandes, considera:

- Aumentar el número de muestras en el re-muestreo
- Ajustar el umbral `tau` para seleccionar más o menos coeficientes
- Reducir `n_time` si la animación es muy lenta

## Licencia

MIT License

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request para discutir cambios.

## Referencias

- [3Blue1Brown - But what is a Fourier series?](https://www.youtube.com/watch?v=r6sGWTCMz2k)
- [Fourier Series Visualization](https://en.wikipedia.org/wiki/Fourier_series)


