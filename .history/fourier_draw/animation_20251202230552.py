"""
Visualización y animación de contornos y epiciclos.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, Tuple


def plot_static_contours(
    original: np.ndarray,
    reconstructed: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Dibuja el contorno original y el reconstruido en una figura.

    Parameters
    ----------
    original : np.ndarray
        Contorno original (complejo) de forma (N,).
    reconstructed : np.ndarray
        Contorno reconstruido (complejo) de forma (M,).
    ax : matplotlib.axes.Axes, optional
        Ejes donde dibujar. Si es None, se crea una nueva figura.

    Returns
    -------
    matplotlib.axes.Axes
        Ejes con los contornos dibujados.
    """
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Extraer coordenadas reales e imaginarias
    x_orig = np.real(original)
    y_orig = np.imag(original)
    x_recon = np.real(reconstructed)
    y_recon = np.imag(reconstructed)

    # Dibujar contornos
    ax.plot(x_orig, y_orig, "b-", label="Original", linewidth=2, alpha=0.7)
    ax.plot(x_recon, y_recon, "r--", label="Reconstruido", linewidth=2, alpha=0.7)

    # Marcar puntos iniciales
    ax.plot(x_orig[0], y_orig[0], "bo", markersize=8, label="Inicio original")
    ax.plot(x_recon[0], y_recon[0], "ro", markersize=8, label="Inicio reconstruido")

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Contorno Original vs Reconstruido", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    return ax


def animate_epicycles(
    coeffs: np.ndarray,
    k_max: int,
    n_time: int = 1000,
    save_path: Optional[str] = None,
    fps: int = 30,
    figsize: Tuple[int, int] = (10, 10),
) -> animation.FuncAnimation:
    """
    Anima los epiciclos que dibujan el contorno reconstruido.

    La animación muestra cómo una serie de círculos (epiciclos) rotando
    a diferentes frecuencias se combinan para dibujar el contorno.

    Parameters
    ----------
    coeffs : np.ndarray
        Coeficientes de Fourier de forma (N,).
    k_max : int
        Número máximo de frecuencias a usar en la reconstrucción.
    n_time : int
        Número de puntos de tiempo para la animación.
    save_path : str, optional
        Ruta donde guardar la animación (.mp4 o .gif).
        Si es None, solo se muestra la animación.
    fps : int
        Frames por segundo para la animación guardada.
    figsize : tuple
        Tamaño de la figura (ancho, alto).

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Objeto de animación.
    """
    coeffs = np.asarray(coeffs)
    N = len(coeffs)

    if k_max < 0:
        raise ValueError("k_max debe ser no negativo")

    k_max = min(k_max, N // 2)

    # Ordenar frecuencias por magnitud de coeficiente (de mayor a menor)
    # para que los epiciclos grandes se dibujen primero
    freq_list = []

    # Añadir componente DC (k=0)
    if k_max >= 0:
        freq_list.append((0, coeffs[0]))

    # Añadir frecuencias positivas (k=1 a k=k_max)
    for k in range(1, k_max + 1):
        if k < N:
            freq_list.append((k, coeffs[k]))

    # Añadir frecuencias negativas (k=-k_max a k=-1)
    for k in range(1, k_max + 1):
        idx = N - k
        if idx < N:
            freq_list.append((-k, coeffs[idx]))

    # Ordenar por magnitud (de mayor a menor)
    freq_list.sort(key=lambda x: np.abs(x[1]), reverse=True)
    
    # Optimización: limitar número de epiciclos visibles para velocidad
    # Mostrar solo los más importantes (los que tienen más energía)
    max_visible_epicycles = min(len(freq_list), 20)  # Máximo 20 epiciclos visibles
    freq_list = freq_list[:max_visible_epicycles]

    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Animación de Epiciclos", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Puntos de tiempo
    t_values = np.linspace(0, 1, n_time, endpoint=False)

    # Pre-calcular posiciones del contorno reconstruido (optimizado)
    z_reconstructed = np.zeros(n_time, dtype=complex)
    # Pre-calcular exponenciales para todas las frecuencias
    for k, c_k in freq_list:
        if k == 0:
            z_reconstructed += c_k
        else:
            # Vectorizar el cálculo
            z_reconstructed += c_k * np.exp(2j * np.pi * k * t_values)

    # Ajustar límites de los ejes basándose en el contorno
    margin = 0.1
    x_min, x_max = np.real(z_reconstructed).min(), np.real(z_reconstructed).max()
    y_min, y_max = np.imag(z_reconstructed).min(), np.imag(z_reconstructed).max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    ax.set_xlim(center_x - max_range / 2 - margin, center_x + max_range / 2 + margin)
    ax.set_ylim(center_y - max_range / 2 - margin, center_y + max_range / 2 + margin)

    # Elementos de la animación
    lines = []
    circles = []
    trace_line = None
    current_point = None

    def init():
        """Inicializa la animación."""
        nonlocal trace_line, current_point

        # Línea de trazado (contorno reconstruido)
        trace_line, = ax.plot([], [], "r-", linewidth=2, alpha=0.5, label="Trazado")

        # Punto actual
        current_point, = ax.plot([], [], "ro", markersize=8, label="Punto actual")

        # Contorno completo (fondo)
        ax.plot(
            np.real(z_reconstructed),
            np.imag(z_reconstructed),
            "b--",
            linewidth=1,
            alpha=0.3,
            label="Contorno completo",
        )

        ax.legend(loc="upper right")

        return [trace_line, current_point]

    def animate(frame):
        """Actualiza la animación en cada frame."""
        t = t_values[frame]

        # Calcular posición acumulada de cada epiciclo
        current_pos = 0.0 + 0.0j

        # Limpiar círculos y líneas anteriores (optimizado)
        # Remover en batch para mejor rendimiento
        while lines:
            lines.pop().remove()
        while circles:
            circles.pop().remove()

        # Dibujar cada epiciclo (optimizado: solo los más importantes)
        for k, c_k in freq_list:
            # Radio y fase del epiciclo
            radius = np.abs(c_k)
            phase = np.angle(c_k)

            if radius < 1e-10:
                continue  # Saltar epiciclos muy pequeños

            # Posición del centro del epiciclo (posición acumulada anterior)
            center = current_pos

            # Posición del punto en el epiciclo
            angle = 2 * np.pi * k * t + phase
            point_on_circle = center + radius * np.exp(1j * angle)

            # Optimización: solo dibujar círculos si son suficientemente grandes
            # (reduce overhead visual sin perder información importante)
            if radius > 0.01:  # Solo dibujar círculos visibles
                circle = plt.Circle(
                    (np.real(center), np.imag(center)),
                    radius,
                    fill=False,
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,  # Línea más delgada
                    alpha=0.4,  # Más transparente
                )
                ax.add_patch(circle)
                circles.append(circle)

            # Dibujar línea desde el centro hasta el punto (solo si es visible)
            if radius > 0.005:
                line, = ax.plot(
                    [np.real(center), np.real(point_on_circle)],
                    [np.imag(center), np.imag(point_on_circle)],
                    "g-",
                    linewidth=1.2,  # Línea más delgada
                    alpha=0.6,  # Más transparente
                )
                lines.append(line)

            # Actualizar posición acumulada
            current_pos = point_on_circle

        # Actualizar trazado hasta el punto actual
        z_trace = z_reconstructed[: frame + 1]
        trace_line.set_data(np.real(z_trace), np.imag(z_trace))

        # Actualizar punto actual
        current_point.set_data([np.real(current_pos)], [np.imag(current_pos)])

        return [trace_line, current_point] + lines + circles

    # Crear animación
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=n_time,
        interval=1000 / fps,  # Intervalo en milisegundos
        blit=False,  # No usar blitting para permitir actualización de círculos
        repeat=True,
    )

    # Guardar animación si se especifica
    if save_path is not None:
        if save_path.endswith(".mp4"):
            writer = animation.FFMpegWriter(fps=fps)
            anim.save(save_path, writer=writer)
        elif save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            raise ValueError("save_path debe terminar en .mp4 o .gif")

    return anim


