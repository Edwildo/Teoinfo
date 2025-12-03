"""
Transformada de Fourier y reconstrucción de contornos.
"""

import numpy as np
from typing import Tuple


def dft_complex_signal(z: np.ndarray) -> np.ndarray:
    """
    Calcula la DFT de una señal compleja usando numpy.fft.fft.

    La DFT está definida como:
        C[k] = (1/N) * sum(n=0 to N-1) z[n] * exp(-2πikn/N)

    Nota: numpy.fft.fft calcula:
        C[k] = sum(n=0 to N-1) z[n] * exp(-2πikn/N)

    Por lo tanto, debemos dividir por N para obtener los coeficientes
    normalizados correctos.

    Parameters
    ----------
    z : np.ndarray
        Señal compleja de forma (N,).

    Returns
    -------
    np.ndarray
        Coeficientes de Fourier de forma (N,).
        C[0] es la componente DC (centroide).
        C[1] a C[N//2] son frecuencias positivas.
        C[N//2+1] a C[N-1] son frecuencias negativas (wrapped).
    """
    if z.ndim != 1:
        raise ValueError("z debe ser un array 1D")

    N = len(z)
    if N == 0:
        raise ValueError("z no puede estar vacío")

    # Calcular FFT y normalizar
    C = np.fft.fft(z) / N

    return C


def idft_complex_signal(C: np.ndarray) -> np.ndarray:
    """
    Calcula la IDFT (transformada inversa) de los coeficientes de Fourier.

    La IDFT está definida como:
        z[n] = sum(k=0 to N-1) C[k] * exp(2πikn/N)

    Parameters
    ----------
    C : np.ndarray
        Coeficientes de Fourier de forma (N,).

    Returns
    -------
    np.ndarray
        Señal compleja reconstruida de forma (N,).
    """
    if C.ndim != 1:
        raise ValueError("C debe ser un array 1D")

    N = len(C)
    if N == 0:
        raise ValueError("C no puede estar vacío")

    # Calcular IFFT y desnormalizar (multiplicar por N)
    z = np.fft.ifft(C) * N

    return z


def compute_fourier_coefficients(z: np.ndarray) -> np.ndarray:
    """
    Calcula los coeficientes de Fourier de una señal compleja.

    Esta función es un alias de dft_complex_signal para mayor claridad
    en el código.

    Parameters
    ----------
    z : np.ndarray
        Señal compleja de forma (N,).

    Returns
    -------
    np.ndarray
        Coeficientes de Fourier de forma (N,).
        Los coeficientes están indexados como en numpy.fft.fft:
        - C[0]: componente DC
        - C[1] a C[N//2]: frecuencias positivas
        - C[N//2+1] a C[N-1]: frecuencias negativas (wrapped)
    """
    return dft_complex_signal(z)


def reconstruct_from_coeffs(
    coeffs: np.ndarray,
    k_max: int,
    n_time: int,
) -> np.ndarray:
    """
    Reconstruye una señal compleja usando un subconjunto de coeficientes.

    Reconstruye la señal usando solo los coeficientes con índices
    en el rango [-k_max, k_max] (considerando el wrapping de frecuencias).

    La reconstrucción se evalúa en n_time puntos equiespaciados en [0, 1).

    Parameters
    ----------
    coeffs : np.ndarray
        Coeficientes de Fourier completos de forma (N,).
    k_max : int
        Número máximo de frecuencias positivas/negativas a usar.
        Se usarán frecuencias desde -k_max hasta k_max (inclusive).
    n_time : int
        Número de puntos de tiempo para la reconstrucción.

    Returns
    -------
    np.ndarray
        Señal compleja reconstruida de forma (n_time,).

    Notes
    -----
    La reconstrucción usa la fórmula:
        z_hat(t) = sum(k=-k_max to k_max) C[k] * exp(2πikt)

    donde t está en [0, 1) y C[k] son los coeficientes de Fourier.
    """
    if coeffs.ndim != 1:
        raise ValueError("coeffs debe ser un array 1D")

    N = len(coeffs)
    if N == 0:
        raise ValueError("coeffs no puede estar vacío")

    if k_max < 0:
        raise ValueError("k_max debe ser no negativo")

    if n_time < 1:
        raise ValueError("n_time debe ser al menos 1")

    # Limitar k_max al rango válido
    k_max = min(k_max, N // 2)

    # Puntos de tiempo en [0, 1)
    t = np.linspace(0, 1, n_time, endpoint=False)

    # Inicializar señal reconstruida
    z_reconstructed = np.zeros(n_time, dtype=complex)

    # Reconstruir usando coeficientes en el rango [-k_max, k_max]
    # Nota: en numpy.fft, las frecuencias negativas están "wrapped"
    # C[0] es k=0
    # C[1] a C[N//2] son k=1 a k=N//2 (frecuencias positivas)
    # C[N//2+1] a C[N-1] son k=-(N//2-1) a k=-1 (frecuencias negativas)

    # Añadir componente DC (k=0)
    if k_max >= 0:
        z_reconstructed += coeffs[0]

    # Añadir frecuencias positivas (k=1 a k=k_max)
    for k in range(1, k_max + 1):
        if k < N:
            z_reconstructed += coeffs[k] * np.exp(2j * np.pi * k * t)

    # Añadir frecuencias negativas (k=-k_max a k=-1)
    # Estas están en las posiciones N-k_max a N-1
    for k in range(1, k_max + 1):
        idx = N - k
        if idx < N:
            z_reconstructed += coeffs[idx] * np.exp(2j * np.pi * (-k) * t)

    return z_reconstructed


def select_k_by_energy(coeffs: np.ndarray, tau: float) -> int:
    """
    Selecciona K basándose en la energía acumulada de los coeficientes.

    Encuentra el menor K tal que la energía acumulada de las frecuencias
    en el rango [-K, K] sea al menos tau * E_total, donde E_total es
    la energía total del espectro.

    Parameters
    ----------
    coeffs : np.ndarray
        Coeficientes de Fourier de forma (N,).
    tau : float
        Umbral de energía acumulada (0.0 - 1.0).
        Por ejemplo, tau=0.95 significa que queremos capturar el 95%
        de la energía total.

    Returns
    -------
    int
        El valor de K seleccionado (número de frecuencias positivas/negativas).

    Raises
    ------
    ValueError
        Si tau no está en [0, 1] o si coeffs está vacío.
    """
    if coeffs.ndim != 1:
        raise ValueError("coeffs debe ser un array 1D")

    N = len(coeffs)
    if N == 0:
        raise ValueError("coeffs no puede estar vacío")

    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau debe estar en [0, 1]")

    # Calcular energía total
    energy_total = np.sum(np.abs(coeffs) ** 2)

    if energy_total < 1e-10:
        # Si no hay energía, retornar 0
        return 0

    # Calcular energía acumulada para cada K
    # Empezar con K=0 (solo DC)
    k_max_possible = N // 2

    for k in range(k_max_possible + 1):
        # Calcular energía acumulada para frecuencias en [-k, k]
        energy_accumulated = 0.0

        # Componente DC (k=0)
        if k >= 0:
            energy_accumulated += np.abs(coeffs[0]) ** 2

        # Frecuencias positivas (k=1 a k=k)
        for k_pos in range(1, k + 1):
            if k_pos < N:
                energy_accumulated += np.abs(coeffs[k_pos]) ** 2

        # Frecuencias negativas (k=-k a k=-1)
        for k_neg in range(1, k + 1):
            idx = N - k_neg
            if idx < N:
                energy_accumulated += np.abs(coeffs[idx]) ** 2

        # Verificar si alcanzamos el umbral
        if energy_accumulated >= tau * energy_total:
            return k

    # Si no se alcanzó el umbral, retornar el máximo posible
    return k_max_possible


def get_coeffs_shifted(coeffs: np.ndarray) -> np.ndarray:
    """
    Reordena los coeficientes usando fftshift para visualización.

    Esto coloca la frecuencia cero en el centro, con frecuencias negativas
    a la izquierda y positivas a la derecha.

    Parameters
    ----------
    coeffs : np.ndarray
        Coeficientes de Fourier en formato numpy.fft.

    Returns
    -------
    np.ndarray
        Coeficientes reordenados con frecuencia cero en el centro.
    """
    return np.fft.fftshift(coeffs)


