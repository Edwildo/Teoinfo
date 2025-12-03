"""
Fourier Draw: Representación de contornos mediante series complejas y epiciclos.

Este módulo implementa la transformación de contornos 2D en señales complejas
periódicas, el cálculo de coeficientes de Fourier mediante DFT/FFT, y la
reconstrucción y animación de contornos usando epiciclos.
"""

__version__ = "0.1.0"

from fourier_draw.contour import Contour
from fourier_draw.fourier import (
    compute_fourier_coefficients,
    reconstruct_from_coeffs,
    select_k_by_energy,
)
from fourier_draw.metrics import mse, psnr, spectral_energy, cumulative_energy
from fourier_draw.animation import plot_static_contours, animate_epicycles
from fourier_draw.io_utils import (
    load_contour_from_txt,
    save_contour_to_txt,
    load_contour_from_image,
    load_contour_auto,
)

__all__ = [
    "Contour",
    "compute_fourier_coefficients",
    "reconstruct_from_coeffs",
    "select_k_by_energy",
    "mse",
    "psnr",
    "spectral_energy",
    "cumulative_energy",
    "plot_static_contours",
    "animate_epicycles",
    "load_contour_from_txt",
    "save_contour_to_txt",
    "load_contour_from_image",
    "load_contour_auto",
]


