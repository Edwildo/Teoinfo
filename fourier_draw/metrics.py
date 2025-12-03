"""
Métricas para evaluar la calidad de la reconstrucción.
"""

import numpy as np
from typing import Union, Tuple


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calcula el Error Cuadrático Medio (MSE) entre dos señales.

    Si las señales son complejas, se calcula el MSE sobre el módulo
    de la diferencia.

    Parameters
    ----------
    original : np.ndarray
        Señal original (compleja o real) de forma (N,).
    reconstructed : np.ndarray
        Señal reconstruida (compleja o real) de forma (N,).

    Returns
    -------
    float
        Error Cuadrático Medio.

    Raises
    ------
    ValueError
        Si las señales no tienen la misma longitud.
    """
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)

    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Las señales deben tener la misma forma: "
            f"{original.shape} vs {reconstructed.shape}"
        )

    # Si son complejas, calcular diferencia compleja
    if np.iscomplexobj(original) or np.iscomplexobj(reconstructed):
        diff = original - reconstructed
        squared_error = np.abs(diff) ** 2
    else:
        # Si son reales, calcular diferencia real
        diff = original - reconstructed
        squared_error = diff ** 2

    return float(np.mean(squared_error))


def psnr(
    original: np.ndarray,
    reconstructed: np.ndarray,
    max_value: float | None = None,
) -> float:
    """
    Calcula el Peak Signal-to-Noise Ratio (PSNR) en decibeles.

    PSNR = 10 * log10(MAX^2 / MSE)

    donde MAX es el valor máximo de la señal original (o el especificado).

    Parameters
    ----------
    original : np.ndarray
        Señal original (compleja o real) de forma (N,).
    reconstructed : np.ndarray
        Señal reconstruida (compleja o real) de forma (N,).
    max_value : float, optional
        Valor máximo para el cálculo de PSNR. Si es None, se usa el
        máximo valor absoluto de la señal original.

    Returns
    -------
    float
        PSNR en decibeles.

    Raises
    ------
    ValueError
        Si las señales no tienen la misma longitud o si max_value <= 0.
    """
    mse_value = mse(original, reconstructed)

    if mse_value < 1e-10:
        # Si el MSE es muy pequeño, el PSNR es infinito
        return float("inf")

    if max_value is None:
        # Calcular máximo valor absoluto de la señal original
        if np.iscomplexobj(original):
            max_value = np.max(np.abs(original))
        else:
            max_value = np.max(np.abs(original))
    else:
        if max_value <= 0:
            raise ValueError("max_value debe ser positivo")

    psnr_value = 10.0 * np.log10((max_value ** 2) / mse_value)

    return float(psnr_value)


def spectral_energy(coeffs: np.ndarray) -> float:
    """
    Calcula la energía total del espectro de Fourier.

    La energía total es la suma de los cuadrados de las magnitudes
    de todos los coeficientes de Fourier.

    Parameters
    ----------
    coeffs : np.ndarray
        Coeficientes de Fourier de forma (N,).

    Returns
    -------
    float
        Energía total del espectro.
    """
    coeffs = np.asarray(coeffs)
    return float(np.sum(np.abs(coeffs) ** 2))


def cumulative_energy(coeffs: np.ndarray, k_max: int) -> float:
    """
    Calcula la energía acumulada de las frecuencias en el rango [-k_max, k_max].

    Parameters
    ----------
    coeffs : np.ndarray
        Coeficientes de Fourier de forma (N,).
    k_max : int
        Número máximo de frecuencias positivas/negativas a incluir.

    Returns
    -------
    float
        Energía acumulada en el rango especificado.
    """
    coeffs = np.asarray(coeffs)
    N = len(coeffs)

    if k_max < 0:
        raise ValueError("k_max debe ser no negativo")

    k_max = min(k_max, N // 2)

    energy = 0.0

    # Componente DC (k=0)
    if k_max >= 0:
        energy += np.abs(coeffs[0]) ** 2

    # Frecuencias positivas (k=1 a k=k_max)
    for k in range(1, k_max + 1):
        if k < N:
            energy += np.abs(coeffs[k]) ** 2

    # Frecuencias negativas (k=-k_max a k=-1)
    for k in range(1, k_max + 1):
        idx = N - k
        if idx < N:
            energy += np.abs(coeffs[idx]) ** 2

    return float(energy)


def energy_ratio(coeffs: np.ndarray, k_max: int) -> float:
    """
    Calcula la razón de energía acumulada respecto a la energía total.

    Parameters
    ----------
    coeffs : np.ndarray
        Coeficientes de Fourier de forma (N,).
    k_max : int
        Número máximo de frecuencias positivas/negativas a incluir.

    Returns
    -------
    float
        Razón de energía acumulada (0.0 - 1.0).
    """
    energy_total = spectral_energy(coeffs)

    if energy_total < 1e-10:
        return 0.0

    energy_accumulated = cumulative_energy(coeffs, k_max)
    return energy_accumulated / energy_total


