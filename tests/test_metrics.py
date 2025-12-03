"""Tests para el módulo metrics."""

import numpy as np
import pytest
from fourier_draw.metrics import mse, psnr, spectral_energy, cumulative_energy, energy_ratio


class TestMSE:
    """Tests para MSE."""

    def test_mse_identical(self):
        """Test que MSE es cero para señales idénticas."""
        z = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        mse_value = mse(z, z)
        assert mse_value == 0.0

    def test_mse_different(self):
        """Test de MSE para señales diferentes."""
        original = np.array([1, 2, 3])
        reconstructed = np.array([1.1, 2.1, 3.1])
        mse_value = mse(original, reconstructed)
        expected = np.mean((0.1) ** 2)  # 0.01
        assert np.isclose(mse_value, expected)

    def test_mse_complex(self):
        """Test de MSE para señales complejas."""
        original = np.array([1 + 1j, 2 + 2j])
        reconstructed = np.array([1.1 + 1.1j, 2.1 + 2.1j])
        mse_value = mse(original, reconstructed)
        # Error en parte real e imaginaria: 0.1 cada una
        # |error|^2 = 0.1^2 + 0.1^2 = 0.02
        expected = 0.02
        assert np.isclose(mse_value, expected, atol=1e-6)

    def test_mse_mismatched_shape(self):
        """Test que falla con formas diferentes."""
        original = np.array([1, 2, 3])
        reconstructed = np.array([1, 2])
        with pytest.raises(ValueError, match="misma forma"):
            mse(original, reconstructed)


class TestPSNR:
    """Tests para PSNR."""

    def test_psnr_identical(self):
        """Test que PSNR es infinito para señales idénticas."""
        z = np.array([1 + 1j, 2 + 2j])
        psnr_value = psnr(z, z)
        assert psnr_value == float("inf")

    def test_psnr_different(self):
        """Test de PSNR para señales diferentes."""
        original = np.array([1.0, 2.0, 3.0])
        reconstructed = np.array([1.1, 2.1, 3.1])
        mse_value = 0.01
        max_value = 3.0
        expected_psnr = 10 * np.log10((max_value ** 2) / mse_value)

        psnr_value = psnr(original, reconstructed, max_value=max_value)
        assert np.isclose(psnr_value, expected_psnr, rtol=1e-6)

    def test_psnr_auto_max(self):
        """Test de PSNR con max_value automático."""
        original = np.array([1.0, 2.0, 3.0])
        reconstructed = np.array([1.1, 2.1, 3.1])
        psnr_value = psnr(original, reconstructed)
        # max_value debe ser 3.0 automáticamente
        assert psnr_value > 0  # Debe ser un valor positivo finito

    def test_psnr_invalid_max(self):
        """Test que falla con max_value inválido."""
        original = np.array([1, 2, 3])
        reconstructed = np.array([1.1, 2.1, 3.1])
        with pytest.raises(ValueError, match="positivo"):
            psnr(original, reconstructed, max_value=0)


class TestSpectralEnergy:
    """Tests para energía espectral."""

    def test_spectral_energy(self):
        """Test de cálculo de energía espectral."""
        coeffs = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        energy = spectral_energy(coeffs)
        expected = np.abs(1 + 1j) ** 2 + np.abs(2 + 2j) ** 2 + np.abs(3 + 3j) ** 2
        assert np.isclose(energy, expected)

    def test_spectral_energy_real(self):
        """Test de energía espectral para coeficientes reales."""
        coeffs = np.array([1, 2, 3])
        energy = spectral_energy(coeffs)
        expected = 1 ** 2 + 2 ** 2 + 3 ** 2
        assert np.isclose(energy, expected)


class TestCumulativeEnergy:
    """Tests para energía acumulada."""

    def test_cumulative_energy_k0(self):
        """Test de energía acumulada con k_max=0 (solo DC)."""
        coeffs = np.array([2.0, 1.0, 1.0, 1.0])
        energy = cumulative_energy(coeffs, k_max=0)
        expected = np.abs(coeffs[0]) ** 2
        assert np.isclose(energy, expected)

    def test_cumulative_energy_k1(self):
        """Test de energía acumulada con k_max=1."""
        coeffs = np.array([1.0, 2.0, 0.5, 0.5])
        n = len(coeffs)
        energy = cumulative_energy(coeffs, k_max=1)
        # Debe incluir: k=0, k=1, k=-1
        expected = (
            np.abs(coeffs[0]) ** 2
            + np.abs(coeffs[1]) ** 2
            + np.abs(coeffs[n - 1]) ** 2
        )
        assert np.isclose(energy, expected)

    def test_cumulative_energy_invalid_k(self):
        """Test que falla con k_max negativo."""
        coeffs = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="no negativo"):
            cumulative_energy(coeffs, k_max=-1)


class TestEnergyRatio:
    """Tests para razón de energía."""

    def test_energy_ratio(self):
        """Test de razón de energía."""
        from fourier_draw.metrics import energy_ratio

        coeffs = np.array([1.0, 2.0, 1.0, 1.0])
        ratio = energy_ratio(coeffs, k_max=1)
        assert 0.0 <= ratio <= 1.0

    def test_energy_ratio_zero_energy(self):
        """Test de razón de energía con energía total cero."""
        from fourier_draw.metrics import energy_ratio

        coeffs = np.array([0.0, 0.0, 0.0])
        ratio = energy_ratio(coeffs, k_max=1)
        assert ratio == 0.0


