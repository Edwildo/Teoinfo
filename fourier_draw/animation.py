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
    coeffs,
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
    coeffs : np.ndarray or list of np.ndarray
        Coeficientes de Fourier. Puede ser:
        - Un único array 1D de forma (N,) para una sola curva.
        - Una lista de arrays 1D para múltiples curvas que se animarán juntas.
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
    # Convertir a lista si es un array simple
    if isinstance(coeffs, np.ndarray):
        coeffs_list = [coeffs]
    elif isinstance(coeffs, list):
        coeffs_list = [np.asarray(c) for c in coeffs]
    else:
        raise TypeError("coeffs debe ser un array 1D o una lista de arrays 1D")
    
    if not coeffs_list:
        raise ValueError("coeffs no puede estar vacío")
    
    N = len(coeffs_list[0])

    if k_max < 0:
        raise ValueError("k_max debe ser no negativo")

    k_max = min(k_max, N // 2)

    # Procesar cada conjunto de coeficientes
    freq_lists = []
    for coeffs in coeffs_list:
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
        
        freq_lists.append(freq_list)

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

    # Pre-calcular posiciones del contorno reconstruido para cada curva
    z_reconstructed_list = []
    for coeffs, freq_list in zip(coeffs_list, freq_lists):
        z_reconstructed = np.zeros(n_time, dtype=complex)
        # Pre-calcular exponenciales para todas las frecuencias
        for k, c_k in freq_list:
            if k == 0:
                z_reconstructed += c_k
            else:
                # Vectorizar el cálculo
                z_reconstructed += c_k * np.exp(2j * np.pi * k * t_values)
        z_reconstructed_list.append(z_reconstructed)

    # Combinar todos los contornos reconstruidos para ajustar límites
    z_all = np.concatenate(z_reconstructed_list)

    # Ajustar límites de los ejes basándose en los contornos
    margin = 0.1
    x_min, x_max = np.real(z_all).min(), np.real(z_all).max()
    y_min, y_max = np.imag(z_all).min(), np.imag(z_all).max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    ax.set_xlim(center_x - max_range / 2 - margin, center_x + max_range / 2 + margin)
    ax.set_ylim(center_y - max_range / 2 - margin, center_y + max_range / 2 + margin)

    # Elementos de la animación (solo trazado, sin epiciclos)
    trace_lines = []
    current_points = []
    
    # Colores para cada curva
    colors = plt.cm.tab10(np.linspace(0, 1, len(coeffs_list)))

    def init():
        """Inicializa la animación."""
        # Crear líneas y puntos para cada curva
        for i, (z_reconstructed, color) in enumerate(zip(z_reconstructed_list, colors)):
            # Línea de trazado
            trace_line, = ax.plot([], [], color=color, linewidth=2.5, alpha=0.8, 
                                  #label=f"Trazado {i+1}" if len(coeffs_list) > 1 else "Trazado"
                                  )
            trace_lines.append(trace_line)

            # Punto actual
            current_point, = ax.plot([], [], "o", color=color, markersize=10, 
                                    #label=f"Punto actual {i+1}" if len(coeffs_list) > 1 else "Punto actual", 
                                    zorder=10)
            current_points.append(current_point)

            # Contorno completo de referencia (fondo, opcional)
            ax.plot(
                np.real(z_reconstructed),
                np.imag(z_reconstructed),
                "--",
                color=color,
                linewidth=1,
                alpha=0.2,
                #label=f"Contorno completo {i+1}" if len(coeffs_list) > 1 else "Contorno completo",
            )

        #ax.legend(loc="upper right")
        return trace_lines + current_points

    def animate(frame):
        """Actualiza la animación en cada frame."""
        result = []
        
        for i, (z_reconstructed, trace_line, current_point) in enumerate(
            zip(z_reconstructed_list, trace_lines, current_points)
        ):
            # Calcular posición usando la reconstrucción pre-calculada
            current_pos = z_reconstructed[frame]

            # Actualizar trazado hasta el punto actual
            z_trace = z_reconstructed[: frame + 1]
            trace_line.set_data(np.real(z_trace), np.imag(z_trace))

            # Actualizar punto actual
            current_point.set_data([np.real(current_pos)], [np.imag(current_pos)])
            
            result.extend([trace_line, current_point])

        return result

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

    # Guardar animación si se especifica (optimizado)
    if save_path is not None:
        if save_path.endswith(".mp4"):
            writer = animation.FFMpegWriter(fps=fps)
            anim.save(save_path, writer=writer)
        elif save_path.endswith(".gif"):
            # Guardar GIF con pillow (más rápido)
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            raise ValueError("save_path debe terminar en .mp4 o .gif")

    return anim


