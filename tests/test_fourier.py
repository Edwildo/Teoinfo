"""Tests para el módulo fourier."""

import numpy as np
import pytest
from fourier_draw.fourier import (
    dft_complex_signal,
    idft_complex_signal,
    compute_fourier_coefficients,
    reconstruct_from_coeffs,
    select_k_by_energy,
)


class TestDFT:
    """Tests para DFT e IDFT."""

    def test_dft_idft_roundtrip(self):
        """Test que DFT seguida de IDFT recupera la señal original."""
        # Señal de prueba
        z = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j])

        # DFT
        coeffs = dft_complex_signal(z)

        # IDFT
        z_reconstructed = idft_complex_signal(coeffs)

        # Debe recuperar la señal original (con tolerancia numérica)
        assert np.allclose(z, z_reconstructed, rtol=1e-10)

    def test_dft_circle(self):
        """Test de DFT para un círculo perfecto."""
        # Círculo parametrizado: z(t) = exp(2πit)
        n = 64
        t = np.linspace(0, 1, n, endpoint=False)
        z = np.exp(2j * np.pi * t)

        # DFT
        coeffs = dft_complex_signal(z)

        # Para un círculo perfecto, solo debe haber coeficientes en k=1 y k=-1
        # (y posiblemente k=0 para el centro)
        magnitudes = np.abs(coeffs)
        # El coeficiente k=1 debe ser dominante
        assert magnitudes[1] > 0.9  # Debe ser cercano a 1

    def test_compute_fourier_coefficients(self):
        """Test que compute_fourier_coefficients es equivalente a dft_complex_signal."""
        z = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        coeffs1 = dft_complex_signal(z)
        coeffs2 = compute_fourier_coefficients(z)
        assert np.allclose(coeffs1, coeffs2)


class TestReconstruction:
    """Tests para reconstrucción de contornos."""

    def test_reconstruct_from_coeffs_full(self):
        """Test de reconstrucción usando todos los coeficientes."""
        # Señal de prueba
        n = 32
        t = np.linspace(0, 1, n, endpoint=False)
        z_original = np.exp(2j * np.pi * t)  # Círculo

        # Calcular coeficientes
        coeffs = compute_fourier_coefficients(z_original)

        # Reconstruir con todos los coeficientes
        k_max = n // 2
        n_time = 100
        z_reconstructed = reconstruct_from_coeffs(coeffs, k_max, n_time)

        # Interpolar el original a los mismos puntos de tiempo
        z_orig_interp = np.interp(
            np.linspace(0, 1, n_time, endpoint=False),
            t,
            np.real(z_original),
        ) + 1j * np.interp(
            np.linspace(0, 1, n_time, endpoint=False),
            t,
            np.imag(z_original),
        )

        # El error debe ser pequeño
        error = np.mean(np.abs(z_orig_interp - z_reconstructed) ** 2)
        assert error < 0.01  # Tolerancia razonable

    def test_reconstruct_from_coeffs_partial(self):
        """Test de reconstrucción usando un subconjunto de coeficientes."""
        # Señal simple: círculo
        n = 64
        t = np.linspace(0, 1, n, endpoint=False)
        z_original = np.exp(2j * np.pi * t)

        # Calcular coeficientes
        coeffs = compute_fourier_coefficients(z_original)

        # Reconstruir con solo k=1 (debería ser suficiente para un círculo)
        k_max = 1
        n_time = 100
        z_reconstructed = reconstruct_from_coeffs(coeffs, k_max, n_time)

        # Debe ser aproximadamente un círculo
        radius = np.abs(z_reconstructed)
        assert np.allclose(radius, 1.0, atol=0.1)

    def test_reconstruct_from_coeffs_invalid_k(self):
        """Test que falla con k_max inválido."""
        coeffs = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError, match="no negativo"):
            reconstruct_from_coeffs(coeffs, -1, 10)


class TestEnergySelection:
    """Tests para selección de K por energía."""

    def test_select_k_by_energy_tau_1(self):
        """Test con tau=1.0 debe usar todos los coeficientes."""
        # Señal de prueba
        n = 32
        z = np.random.randn(n) + 1j * np.random.randn(n)
        coeffs = compute_fourier_coefficients(z)

        k_selected = select_k_by_energy(coeffs, tau=1.0)
        k_max_possible = n // 2

        # Debe seleccionar el máximo posible (o muy cercano)
        assert k_selected >= k_max_possible - 1

    def test_select_k_by_energy_tau_0(self):
        """Test con tau=0.0 debe seleccionar K=0."""
        n = 32
        z = np.random.randn(n) + 1j * np.random.randn(n)
        coeffs = compute_fourier_coefficients(z)

        k_selected = select_k_by_energy(coeffs, tau=0.0)

        # Debe seleccionar K=0 (solo DC)
        assert k_selected == 0

    def test_select_k_by_energy_increasing_tau(self):
        """Test que al incrementar tau se obtiene K mayor o igual."""
        n = 64
        z = np.random.randn(n) + 1j * np.random.randn(n)
        coeffs = compute_fourier_coefficients(z)

        k1 = select_k_by_energy(coeffs, tau=0.5)
        k2 = select_k_by_energy(coeffs, tau=0.7)
        k3 = select_k_by_energy(coeffs, tau=0.9)

        assert k1 <= k2 <= k3

    def test_select_k_by_energy_invalid_tau(self):
        """Test que falla con tau fuera de [0, 1]."""
        coeffs = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="en \\[0, 1\\]"):
            select_k_by_energy(coeffs, tau=1.5)

    def test_select_k_by_energy_empty(self):
        """Test que falla con array vacío."""
        with pytest.raises(ValueError, match="vacío"):
            select_k_by_energy(np.array([]), tau=0.95)


