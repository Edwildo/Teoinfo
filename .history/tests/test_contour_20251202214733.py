"""Tests para el módulo contour."""

import numpy as np
import pytest
from fourier_draw.contour import Contour


class TestContour:
    """Tests para la clase Contour."""

    def test_init(self):
        """Test de inicialización básica."""
        x = np.array([0, 1, 2, 3])
        y = np.array([0, 1, 2, 3])
        contour = Contour(x, y)
        assert contour.n_points == 4
        assert np.allclose(contour.x, x)
        assert np.allclose(contour.y, y)

    def test_init_mismatched_length(self):
        """Test que falla con longitudes diferentes."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="misma longitud"):
            Contour(x, y)

    def test_init_empty(self):
        """Test que falla con arrays vacíos."""
        with pytest.raises(ValueError, match="vacío"):
            Contour(np.array([]), np.array([]))

    def test_complex_signal(self):
        """Test de conversión a señal compleja."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        contour = Contour(x, y)
        z = contour.complex_signal
        expected = np.array([1 + 4j, 2 + 5j, 3 + 6j])
        assert np.allclose(z, expected)

    def test_is_closed(self):
        """Test de detección de contorno cerrado."""
        # Contorno cerrado (círculo)
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        contour = Contour(x, y)
        assert not contour.is_closed()  # No está cerrado porque no incluye el punto final

        # Contorno explícitamente cerrado
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])
        contour_closed = Contour(x_closed, y_closed)
        assert contour_closed.is_closed()

    def test_closed(self):
        """Test de cierre de contorno."""
        x = np.array([0, 1, 2, 3])
        y = np.array([0, 1, 2, 3])
        contour = Contour(x, y)
        contour_closed = contour.closed()
        assert contour_closed.is_closed()
        assert contour_closed.n_points == contour.n_points + 1
        assert contour_closed.x[0] == contour_closed.x[-1]
        assert contour_closed.y[0] == contour_closed.y[-1]

    def test_resample_by_arclength_circle(self):
        """Test de re-muestreo para un círculo perfecto."""
        # Crear círculo con muchos puntos
        n_original = 100
        theta = np.linspace(0, 2 * np.pi, n_original, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        contour = Contour(x, y)

        # Re-muestrear a menos puntos
        n_new = 50
        contour_resampled = contour.resample_by_arclength(n_new)

        assert contour_resampled.n_points == n_new

        # Verificar que la forma se mantiene aproximadamente
        # (el círculo debe seguir siendo aproximadamente circular)
        center_x = np.mean(contour_resampled.x)
        center_y = np.mean(contour_resampled.y)
        distances = np.sqrt(
            (contour_resampled.x - center_x) ** 2 + (contour_resampled.y - center_y) ** 2
        )
        # El radio debe ser aproximadamente constante
        assert np.std(distances) < 0.1  # Tolerancia razonable

    def test_resample_by_arclength_closed(self):
        """Test de re-muestreo para contorno cerrado."""
        # Círculo cerrado
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])
        contour = Contour(x_closed, y_closed)

        n_new = 64
        contour_resampled = contour.resample_by_arclength(n_new)

        # Para contornos cerrados, el re-muestreo debe mantener el cierre
        assert contour_resampled.n_points == n_new

    def test_resample_by_arclength_invalid(self):
        """Test que falla con n_samples inválido."""
        x = np.array([0, 1, 2, 3])
        y = np.array([0, 1, 2, 3])
        contour = Contour(x, y)

        with pytest.raises(ValueError, match="al menos 2"):
            contour.resample_by_arclength(1)

    def test_len(self):
        """Test del método __len__."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        contour = Contour(x, y)
        assert len(contour) == 3

    def test_repr(self):
        """Test del método __repr__."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        contour = Contour(x, y)
        repr_str = repr(contour)
        assert "Contour" in repr_str
        assert "n_points=3" in repr_str


