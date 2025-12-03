#!/usr/bin/env python
"""
Script principal para ejecutar el pipeline completo de reconstrucción de contornos.

Ejemplo de uso:
    python scripts/run_example.py --input data/mi_contorno.txt --samples 1024 --tau 0.97 --animate
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Añadir el directorio raíz al path para importar el módulo
sys.path.insert(0, str(Path(__file__).parent.parent))

from fourier_draw import (
    Contour,
    compute_fourier_coefficients,
    reconstruct_from_coeffs,
    select_k_by_energy,
    mse,
    psnr,
    cumulative_energy,
    spectral_energy,
    plot_static_contours,
    animate_epicycles,
    load_contour_auto,
)
from fourier_draw.config import config


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Reconstrucción de contornos mediante Transformada de Fourier"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Ruta al archivo de contorno (formato: dos columnas x y) o imagen (PNG, JPG, etc.)",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Umbral para binarización de imágenes (0-255, default: 128)",
    )

    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invertir imagen antes de procesar (útil para imágenes con fondo oscuro)",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=config.default_n_samples,
        help=f"Número de muestras para re-muestreo (default: {config.default_n_samples})",
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=config.default_tau,
        help=f"Umbral de energía acumulada para selección de K (default: {config.default_tau})",
    )

    parser.add_argument(
        "--k-max",
        type=int,
        default=None,
        help="Número máximo de frecuencias a usar (opcional; si no se especifica, se selecciona por energía)",
    )

    parser.add_argument(
        "--n-time",
        type=int,
        default=config.default_n_time,
        help=f"Número de puntos de tiempo para reconstrucción (default: {config.default_n_time})",
    )

    parser.add_argument(
        "--animate",
        action="store_true",
        help="Generar animación de epiciclos",
    )

    parser.add_argument(
        "--save-animation",
        type=str,
        default=None,
        help="Ruta donde guardar la animación (.mp4 o .gif)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=config.default_fps,
        help=f"Frames por segundo para la animación (default: {config.default_fps})",
    )

    args = parser.parse_args()

    # Validar argumentos
    if args.samples < 2:
        print("Error: --samples debe ser al menos 2", file=sys.stderr)
        sys.exit(1)

    if not 0.0 <= args.tau <= 1.0:
        print("Error: --tau debe estar en [0, 1]", file=sys.stderr)
        sys.exit(1)

    if args.k_max is not None and args.k_max < 0:
        print("Error: --k-max debe ser no negativo", file=sys.stderr)
        sys.exit(1)

    # Cargar contorno (automáticamente detecta si es imagen o texto)
    print(f"Cargando contorno desde: {args.input}")
    try:
        # Detectar si es imagen o texto
        input_path = Path(args.input)
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif"}
        is_image = input_path.suffix.lower() in image_extensions

        if is_image:
            print(f"  Detectado como imagen, extrayendo contorno...")
            contour = load_contour_auto(
                args.input, threshold=args.threshold, invert=args.invert
            )
        else:
            print(f"  Detectado como archivo de texto...")
            contour = load_contour_auto(args.input)

        print(f"  Contorno cargado: {contour.n_points} puntos")
    except Exception as e:
        print(f"Error al cargar contorno: {e}", file=sys.stderr)
        sys.exit(1)

    # Cerrar y re-muestrear por longitud de arco
    print(f"Cerrando y re-muestreando a {args.samples} puntos...")
    contour_closed = contour.closed()
    contour_resampled = contour_closed.resample_by_arclength(args.samples)
    print(f"  Contorno re-muestreado: {contour_resampled.n_points} puntos")

    # Obtener señal compleja
    z = contour_resampled.complex_signal

    # Calcular coeficientes de Fourier
    print("Calculando coeficientes de Fourier...")
    coeffs = compute_fourier_coefficients(z)
    print(f"  Coeficientes calculados: {len(coeffs)} frecuencias")

    # Seleccionar K
    if args.k_max is not None:
        k_selected = args.k_max
        print(f"Usando K={k_selected} (especificado por usuario)")
    else:
        print(f"Seleccionando K por energía (tau={args.tau})...")
        k_selected = select_k_by_energy(coeffs, args.tau)
        print(f"  K seleccionado: {k_selected}")

    # Calcular energía acumulada
    energy_total = spectral_energy(coeffs)
    energy_accumulated = cumulative_energy(coeffs, k_selected)
    energy_ratio = energy_accumulated / energy_total if energy_total > 0 else 0.0
    print(f"  Energía total: {energy_total:.6e}")
    print(f"  Energía acumulada (K={k_selected}): {energy_accumulated:.6e} ({100*energy_ratio:.2f}%)")

    # Reconstruir contorno
    print(f"Reconstruyendo contorno con {args.n_time} puntos de tiempo...")
    z_reconstructed = reconstruct_from_coeffs(coeffs, k_selected, args.n_time)

    # Calcular métricas
    print("Calculando métricas...")
    # Para comparar, necesitamos re-muestrear el original al mismo número de puntos
    z_original_resampled = contour_resampled.complex_signal
    # Interpolar el original a los mismos puntos de tiempo que la reconstrucción
    t_orig = np.linspace(0, 1, len(z_original_resampled), endpoint=False)
    t_recon = np.linspace(0, 1, args.n_time, endpoint=False)
    z_orig_interp = np.interp(t_recon, t_orig, np.real(z_original_resampled)) + 1j * np.interp(
        t_recon, t_orig, np.imag(z_original_resampled)
    )

    mse_value = mse(z_orig_interp, z_reconstructed)
    psnr_value = psnr(z_orig_interp, z_reconstructed)

    print(f"  MSE: {mse_value:.6e}")
    print(f"  PSNR: {psnr_value:.2f} dB")

    # Visualizar contornos
    print("Generando visualización...")
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_static_contours(z_original_resampled, z_reconstructed, ax=ax)
    plt.tight_layout()
    plt.show(block=False)

    # Animación
    if args.animate:
        print("Generando animación de epiciclos...")
        try:
            anim = animate_epicycles(
                coeffs,
                k_selected,
                n_time=args.n_time,
                save_path=args.save_animation,
                fps=args.fps,
            )
            if args.save_animation:
                print(f"  Animación guardada en: {args.save_animation}")
            else:
                print("  Animación mostrada en ventana (cierra la ventana para terminar)")
                plt.show(block=True)
        except Exception as e:
            print(f"Error al generar animación: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print("\n¡Proceso completado!")


if __name__ == "__main__":
    main()


