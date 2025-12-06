#!/usr/bin/env python
"""
Script optimizado para procesar logos complejos con alta precisión.
Extrae múltiples contornos de una imagen y los anima juntos.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from fourier_draw import (
    load_contour_auto,
    compute_fourier_coefficients,
    reconstruct_from_coeffs,
    select_k_by_energy,
    mse,
    psnr,
    plot_static_contours,
    animate_epicycles,
)

print("=" * 70)
print("PROCESAMIENTO OPTIMIZADO DE LOGO (MÚLTIPLES CONTORNOS)")
print("=" * 70)

# Configuración de ALTA PRECISIÓN para máxima exactitud
IMAGEN = "data/naruto.png"
SAMPLES = 2048  # Muchas más muestras para mejor precisión
TAU = 1.0  # Tau = 1.0 (TODAS las frecuencias) para MÁXIMA PRECISIÓN
N_TIME = 1500  # Más puntos para reconstrucción más precisa
THRESHOLD = 128
ANIM_FRAMES = 300  # Más frames para animación más suave
ANIM_FPS = 20  # FPS para animación fluida

print(f"\n📷 Imagen: {IMAGEN}")
print(f"⚙️  Configuración de ALTA PRECISIÓN:")
print(f"   • Muestras: {SAMPLES} (máxima resolución)")
print(f"   • Tau: {TAU} (MÁXIMA PRECISIÓN - TODAS las frecuencias)")
print(f"   • Puntos de tiempo: {N_TIME} (reconstrucción detallada)")

# 1. Cargar contornos
print(f"\n1️⃣  Extrayendo contornos...")
try:
    contours_data = load_contour_auto(IMAGEN, threshold=THRESHOLD)
    
    # Manejar tanto un único Contour como una lista de Contours
    if isinstance(contours_data, list):
        contours = contours_data
    else:
        contours = [contours_data]
    
    print(f"   ✓ Contornos extraídos: {len(contours)} contorno(s)")
    for i, contour in enumerate(contours):
        print(f"     • Contorno {i+1}: {contour.n_points} puntos")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# 2. Procesar cada contorno
print(f"\n2️⃣  Procesando contornos...")
all_coeffs = []
all_contours_resampled = []
all_metrics = []

for i, contour in enumerate(contours):
    print(f"\n   Contorno {i+1}/{len(contours)}:")
    
    # Cerrar y re-muestrear
    contour_closed = contour.closed()
    contour_resampled = contour_closed.resample_by_arclength(SAMPLES)
    all_contours_resampled.append(contour_resampled)
    print(f"   ✓ Re-muestreado a: {contour_resampled.n_points} puntos")
    
    # Calcular coeficientes
    z = contour_resampled.complex_signal
    coeffs = compute_fourier_coefficients(z)
    all_coeffs.append(coeffs)
    print(f"   ✓ Coeficientes: {len(coeffs)} frecuencias")
    
    # Seleccionar K con tau muy alto para máxima precisión
    k_selected = select_k_by_energy(coeffs, tau=TAU)
    print(f"   ✓ K seleccionado: {k_selected} frecuencias")
    
    # Reconstruir
    z_reconstructed = reconstruct_from_coeffs(coeffs, k_selected, n_time=N_TIME)
    print(f"   ✓ Reconstruido: {len(z_reconstructed)} puntos")
    
    # Métricas
    t_orig = np.linspace(0, 1, len(z), endpoint=False)
    t_recon = np.linspace(0, 1, len(z_reconstructed), endpoint=False)
    z_orig_interp = (
        np.interp(t_recon, t_orig, np.real(z))
        + 1j * np.interp(t_recon, t_orig, np.imag(z))
    )
    
    mse_value = mse(z_orig_interp, z_reconstructed)
    psnr_value = psnr(z_orig_interp, z_reconstructed)
    all_metrics.append((mse_value, psnr_value, k_selected))
    print(f"   ✓ MSE: {mse_value:.6e}")
    print(f"   ✓ PSNR: {psnr_value:.2f} dB")

# 3. Visualización estática (cada contorno por separado)
print(f"\n3️⃣  Generando visualizaciones...")
for i, (contour_resampled, coeffs) in enumerate(zip(all_contours_resampled, all_coeffs)):
    z = contour_resampled.complex_signal
    k_selected = all_metrics[i][2]
    z_reconstructed = reconstruct_from_coeffs(coeffs, k_selected, n_time=N_TIME)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_static_contours(z, z_reconstructed, ax=ax)
    ax.set_title(f"Logo: Contorno {i+1} - Original vs Reconstruido", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"contornos/contorno_{i+1}_resultado.png", dpi=150, bbox_inches='tight')
    print(f"   ✓ Guardado: contorno_{i+1}_resultado.png")
    plt.close()

# 4. Animación con todos los contornos juntos
print(f"\n4️⃣  Generando animación de epiciclos con todos los contornos...")
print(f"   ⏳ Generando {ANIM_FRAMES} frames a {ANIM_FPS} fps (animando {len(all_coeffs)} contorno(s) juntos)...")

try:
    anim = animate_epicycles(
        all_coeffs,  # Pasar lista de coeficientes
        max(all_metrics, key=lambda x: x[2])[2],  # k_max = máximo k seleccionado
        n_time=ANIM_FRAMES,
        save_path="contornos_epiciclos.gif",
        fps=ANIM_FPS,
    )
    print(f"   ✓ Animación guardada: contornos_epiciclos.gif")
    print(f"   ✓ {ANIM_FRAMES} frames a {ANIM_FPS} fps (rápido y eficiente)")
except Exception as e:
    print(f"   ⚠️  Error al guardar: {e}")
    import traceback
    traceback.print_exc()
    print(f"   💡 Visualizaciones estáticas disponibles")

print("\n" + "=" * 70)
print("✅ ¡PROCESO COMPLETADO!")
print("=" * 70)
print(f"\n📊 Resultados:")
print(f"   • Contornos procesados: {len(all_coeffs)}")
print(f"   • Archivos generados:")
for i in range(len(all_coeffs)):
    mse_v, psnr_v, k_v = all_metrics[i]
    print(f"     - contorno_{i+1}_resultado.png (PSNR: {psnr_v:.2f} dB, K: {k_v})")
print(f"     - contornos_epiciclos.gif")

print("\n" + "=" * 70)

