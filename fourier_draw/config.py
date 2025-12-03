"""
Configuración por defecto del proyecto.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuración por defecto para el procesamiento de contornos."""

    # Número de muestras por defecto para re-muestreo
    default_n_samples: int = 1024

    # Umbral de energía acumulada para selección de K (0.0 - 1.0)
    default_tau: float = 0.95

    # Número de puntos de tiempo para reconstrucción
    default_n_time: int = 1000

    # FPS para animaciones
    default_fps: int = 30

    # Duración de animación en segundos
    default_animation_duration: float = 10.0

    # Tolerancia para considerar un contorno cerrado
    closure_tolerance: float = 1e-6


# Instancia global de configuración
config = Config()


