#!/usr/bin/env python
"""
Script optimizado para procesar logos complejos con alta precisión.
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
print("PROCESAMIENTO OPTIMIZADO DE LOGO")
print("=" * 70)

# Configuración de ALTA PRECISIÓN para máxima exactitud
IMAGEN = "data/corazon.png"
SAMPLES = 2048  # Muchas más muestras para mejor precisión
TAU = 1.0  # Tau = 1.0 (TODAS las frecuencias) para MÁXIMA PRECISIÓN
N_TIME = 1500  # Más puntos para reconstrucción más precisa
THRESHOLD = 128
ANIM_FRAMES = 300  # Más frames para animación más suave
ANIM_FPS = 20  # FPS para animación fluida

print(f"\n📷 Imagen: {IMAGEN}")
print(f"⚙️  Configuración de ALTA PRECISIÓN:")
print(f"   • Muestras: {SAMPLES} (máxima resolución)")
print(f"   • Tau: {TAU} (máxima precisión - casi todas las frecuencias)")
print(f"   • Puntos de tiempo: {N_TIME} (reconstrucción detallada)")

# 1. Cargar contorno
print(f"\n1️⃣  Extrayendo contorno...")
try:
    contour = load_contour_auto(IMAGEN, threshold=THRESHOLD)
    print(f"   ✓ Contorno extraído: {contour.n_points} puntos")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# 2. Cerrar y re-muestrear
print(f"\n2️⃣  Cerrando y re-muestreando...")
contour_closed = contour.closed()
contour_resampled = contour_closed.resample_by_arclength(SAMPLES)
print(f"   ✓ Re-muestreado a: {contour_resampled.n_points} puntos")

# 3. Calcular coeficientes
print(f"\n3️⃣  Calculando coeficientes de Fourier...")
z = contour_resampled.complex_signal
coeffs = compute_fourier_coefficients(z)
print(f"   ✓ Coeficientes: {len(coeffs)} frecuencias")

# 4. Seleccionar K con tau muy alto para máxima precisión
print(f"\n4️⃣  Seleccionando frecuencias (tau={TAU}) para MÁXIMA PRECISIÓN...")
k_selected = select_k_by_energy(coeffs, tau=TAU)
print(f"   ✓ K seleccionado: {k_selected} frecuencias")
print(f"   ℹ️  Usando casi todas las frecuencias para reconstrucción exacta")

# 5. Reconstruir
print(f"\n5️⃣  Reconstruyendo contorno...")
z_reconstructed = reconstruct_from_coeffs(coeffs, k_selected, n_time=N_TIME)
print(f"   ✓ Reconstruido: {len(z_reconstructed)} puntos")

# 6. Métricas
print(f"\n6️⃣  Calculando métricas...")
t_orig = np.linspace(0, 1, len(z), endpoint=False)
t_recon = np.linspace(0, 1, len(z_reconstructed), endpoint=False)
z_orig_interp = (
    np.interp(t_recon, t_orig, np.real(z))
    + 1j * np.interp(t_recon, t_orig, np.imag(z))
)

mse_value = mse(z_orig_interp, z_reconstructed)
psnr_value = psnr(z_orig_interp, z_reconstructed)
print(f"   ✓ MSE: {mse_value:.6e}")
print(f"   ✓ PSNR: {psnr_value:.2f} dB")

# 7. Visualización
print(f"\n7️⃣  Generando visualización...")
fig, ax = plt.subplots(figsize=(14, 14))
plot_static_contours(z, z_reconstructed, ax=ax)
ax.set_title("Logo: Original vs Reconstruido (Alta Precisión)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("corazon_resultado.png", dpi=200, bbox_inches='tight')
print(f"   ✓ Guardado: corazon_resultado.png")

# 8. Animación optimizada (rápida)
print(f"\n8️⃣  Generando animación rápida...")
print(f"   ⏳ Generando {ANIM_FRAMES} frames a {ANIM_FPS} fps (optimizado para velocidad)...")
try:
    anim = animate_epicycles(
        coeffs,
        k_selected,
        n_time=ANIM_FRAMES,  # Muchos menos frames
        save_path="corazon_epiciclos.gif",
        fps=ANIM_FPS,  # FPS más bajo
    )
    print(f"   ✓ Animación guardada: corazon_epiciclos.gif")
    print(f"   ✓ {ANIM_FRAMES} frames a {ANIM_FPS} fps (rápido y eficiente)")
except Exception as e:
    print(f"   ⚠️  Error al guardar: {e}")
    import traceback
    traceback.print_exc()
    print(f"   💡 Visualización estática disponible en: corazon_resultado.png")

print("\n" + "=" * 70)
print("✅ ¡PROCESO COMPLETADO!")
print("=" * 70)
print(f"\n📊 Resultados:")
print(f"   • Frecuencias usadas: {k_selected}")
print(f"   • Calidad (PSNR): {psnr_value:.2f} dB")
print(f"   • Archivos generados:")
print(f"     - corazon_resultado.png")
print(f"     - corazon_epiciclos.gif")
print("\n" + "=" * 70)

