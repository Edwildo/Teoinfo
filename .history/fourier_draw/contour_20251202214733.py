"""
Representación y procesamiento de contornos 2D.
"""

import numpy as np
from typing import Optional
from fourier_draw.config import config


class Contour:
    """
    Representa un contorno 2D como una serie de puntos (x, y).

    El contorno puede ser abierto o cerrado, y puede ser re-muestreado
    a lo largo de la longitud de arco para obtener puntos equiespaciados.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Inicializa un contorno a partir de coordenadas x e y.

        Parameters
        ----------
        x : np.ndarray
            Coordenadas x del contorno (1D).
        y : np.ndarray
            Coordenadas y del contorno (1D).

        Raises
        ------
        ValueError
            Si x e y no tienen la misma longitud o están vacíos.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x e y deben ser arrays 1D")

        if len(x) != len(y):
            raise ValueError("x e y deben tener la misma longitud")

        if len(x) == 0:
            raise ValueError("El contorno no puede estar vacío")

        self._x = x
        self._y = y

    @property
    def x(self) -> np.ndarray:
        """Retorna las coordenadas x del contorno."""
        return self._x

    @property
    def y(self) -> np.ndarray:
        """Retorna las coordenadas y del contorno."""
        return self._y

    @property
    def n_points(self) -> int:
        """Retorna el número de puntos del contorno."""
        return len(self._x)

    @property
    def complex_signal(self) -> np.ndarray:
        """
        Convierte el contorno en una señal compleja z = x + i*y.

        Returns
        -------
        np.ndarray
            Array complejo de forma (n_points,) donde z[n] = x[n] + 1j * y[n].
        """
        return self._x + 1j * self._y

    def is_closed(self, tolerance: Optional[float] = None) -> bool:
        """
        Verifica si el contorno está cerrado.

        Un contorno se considera cerrado si la distancia entre el primer
        y último punto es menor que la tolerancia especificada.

        Parameters
        ----------
        tolerance : float, optional
            Tolerancia para considerar el contorno cerrado.
            Por defecto usa config.closure_tolerance.

        Returns
        -------
        bool
            True si el contorno está cerrado, False en caso contrario.
        """
        if tolerance is None:
            tolerance = config.closure_tolerance

        dx = self._x[-1] - self._x[0]
        dy = self._y[-1] - self._y[0]
        distance = np.sqrt(dx**2 + dy**2)

        return distance < tolerance

    def closed(self) -> "Contour":
        """
        Garantiza que el contorno esté cerrado.

        Si el contorno no está cerrado, añade el primer punto al final.

        Returns
        -------
        Contour
            Nuevo contorno cerrado.
        """
        if self.is_closed():
            return Contour(self._x.copy(), self._y.copy())

        # Añadir el primer punto al final para cerrar
        x_closed = np.append(self._x, self._x[0])
        y_closed = np.append(self._y, self._y[0])

        return Contour(x_closed, y_closed)

    def resample_by_arclength(self, n_samples: int) -> "Contour":
        """
        Re-muestrea el contorno a lo largo de la longitud de arco.

        Obtiene n_samples puntos aproximadamente equiespaciados a lo largo
        de la longitud de arco del contorno usando interpolación lineal.

        Parameters
        ----------
        n_samples : int
            Número de puntos en el contorno re-muestreado.

        Returns
        -------
        Contour
            Nuevo contorno re-muestreado.

        Raises
        ------
        ValueError
            Si n_samples < 2.
        """
        if n_samples < 2:
            raise ValueError("n_samples debe ser al menos 2")

        # Calcular longitudes de arco acumuladas
        dx = np.diff(self._x)
        dy = np.diff(self._y)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])  # Longitud acumulada

        # Si el contorno está cerrado, la longitud total incluye el último segmento
        if self.is_closed():
            # Añadir el segmento final si está cerrado
            dx_final = self._x[0] - self._x[-1]
            dy_final = self._y[0] - self._y[-1]
            ds_final = np.sqrt(dx_final**2 + dy_final**2)
            s_total = s[-1] + ds_final
        else:
            s_total = s[-1]

        # Si la longitud total es cero (puntos duplicados), retornar copia
        if s_total < 1e-10:
            return Contour(self._x.copy(), self._y.copy())

        # Parámetros de longitud de arco normalizados [0, 1]
        s_normalized = s / s_total

        # Nuevos parámetros equiespaciados
        if self.is_closed():
            # Para contornos cerrados, excluir el último punto (duplicado)
            s_new = np.linspace(0, 1, n_samples, endpoint=False)
        else:
            # Para contornos abiertos, incluir ambos extremos
            s_new = np.linspace(0, 1, n_samples)

        # Interpolación lineal de x e y
        x_new = np.interp(s_new, s_normalized, self._x)
        y_new = np.interp(s_new, s_normalized, self._y)

        return Contour(x_new, y_new)

    def __len__(self) -> int:
        """Retorna el número de puntos del contorno."""
        return self.n_points

    def __repr__(self) -> str:
        """Representación en string del contorno."""
        return f"Contour(n_points={self.n_points}, closed={self.is_closed()})"


