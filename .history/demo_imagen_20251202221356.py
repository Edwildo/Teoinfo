#!/usr/bin/env python
"""
Demostración completa: Cargar imagen, extraer contorno y reproducir con epiciclos.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from fourier_draw import (
    load_contour_from_image,
    compute_fourier_coefficients,
    reconstruct_from_coeffs,
    select_k_by_energy,
    mse,
    psnr,
    plot_static_contours,
    animate_epicycles,
)

print("=" * 70)
print("DEMOSTRACIÓN: REPRODUCCIÓN DE CONTORNOS DESDE IMÁGENES")
print("=" * 70)

# Seleccionar imagen
imagen = "data/circulo_imagen.png"
print(f"\n📷 Imagen seleccionada: {imagen}")

# 1. Cargar contorno desde imagen
print("\n1️⃣  Extrayendo contorno de la imagen...")
try:
    contour = load_contour_from_image(imagen, threshold=128)
    print(f"   ✓ Contorno extraído: {contour.n_points} puntos")
    print(f"   ✓ Rango X: [{contour.x.min():.1f}, {contour.x.max():.1f}]")
    print(f"   ✓ Rango Y: [{contour.y.min():.1f}, {contour.y.max():.1f}]")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# 2. Cerrar y re-muestrear
print("\n2️⃣  Cerrando y re-muestreando contorno...")
contour_closed = contour.closed()
contour_resampled = contour_closed.resample_by_arclength(512)
print(f"   ✓ Contorno re-muestreado: {contour_resampled.n_points} puntos")

# 3. Calcular coeficientes de Fourier
print("\n3️⃣  Calculando coeficientes de Fourier...")
z = contour_resampled.complex_signal
coeffs = compute_fourier_coefficients(z)
print(f"   ✓ Coeficientes calculados: {len(coeffs)} frecuencias")

# 4. Seleccionar K por energía
print("\n4️⃣  Seleccionando K por energía (tau=0.97)...")
k_selected = select_k_by_energy(coeffs, tau=0.97)
print(f"   ✓ K seleccionado: {k_selected} frecuencias")

# 5. Reconstruir
print("\n5️⃣  Reconstruyendo contorno con epiciclos...")
z_reconstructed = reconstruct_from_coeffs(coeffs, k_selected, n_time=1000)
print(f"   ✓ Contorno reconstruido: {len(z_reconstructed)} puntos")

# 6. Calcular métricas
print("\n6️⃣  Calculando calidad de reconstrucción...")
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

# 7. Visualización estática
print("\n7️⃣  Generando visualización estática...")
fig1, ax1 = plt.subplots(figsize=(12, 12))
plot_static_contours(z, z_reconstructed, ax=ax1)
ax1.set_title("Contorno Original (desde imagen) vs Reconstruido con Epiciclos", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("resultado_imagen.png", dpi=150, bbox_inches='tight')
print(f"   ✓ Visualización guardada en: resultado_imagen.png")

# 8. Animación
print("\n8️⃣  Generando animación de epiciclos...")
print("   ⏳ Esto puede tardar unos segundos...")
try:
    anim = animate_epicycles(
        coeffs,
        k_selected,
        n_time=1000,
        save_path="animacion_imagen.gif",
        fps=30,
    )
    print(f"   ✓ Animación guardada en: animacion_imagen.gif")
    print(f"   ✓ La animación muestra cómo los epiciclos dibujan el contorno")
except Exception as e:
    print(f"   ⚠️  Error al guardar animación: {e}")
    print(f"   💡 Mostrando animación en ventana...")
    anim = animate_epicycles(coeffs, k_selected, n_time=1000)

print("\n" + "=" * 70)
print("✅ ¡DEMOSTRACIÓN COMPLETADA!")
print("=" * 70)
print("\n📊 Resumen:")
print(f"   • Imagen procesada: {imagen}")
print(f"   • Puntos extraídos: {contour.n_points}")
print(f"   • Puntos re-muestreados: {contour_resampled.n_points}")
print(f"   • Frecuencias usadas: {k_selected}")
print(f"   • Calidad (PSNR): {psnr_value:.2f} dB")
print(f"\n📁 Archivos generados:")
print(f"   • resultado_imagen.png - Comparación visual")
print(f"   • animacion_imagen.gif - Animación de epiciclos")
print("\n💡 Puedes probar con otras imágenes:")
print(f"   • data/cuadrado_imagen.png")
print(f"   • data/triangulo_imagen.png")
print(f"   • O cualquier imagen PNG/JPG que tengas")
print("\n" + "=" * 70)

# Mostrar la visualización
plt.show(block=True)

