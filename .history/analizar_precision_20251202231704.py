#!/usr/bin/env python
"""
Análisis de precisión: cuántos puntos/frecuencias se necesitan para reconstrucción exacta.
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
)

print("=" * 70)
print("ANÁLISIS DE PRECISIÓN PARA RECONSTRUCCIÓN EXACTA")
print("=" * 70)

IMAGEN = "data/logo.png"
SAMPLES = 1024
THRESHOLD = 128

# Cargar contorno
print(f"\n📷 Cargando: {IMAGEN}")
contour = load_contour_auto(IMAGEN, threshold=THRESHOLD)
contour_closed = contour.closed()
contour_resampled = contour_closed.resample_by_arclength(SAMPLES)
z = contour_resampled.complex_signal
coeffs = compute_fourier_coefficients(z)

print(f"✓ Contorno: {contour_resampled.n_points} puntos")
print(f"✓ Coeficientes: {len(coeffs)} frecuencias")

# Analizar diferentes valores de tau (energía acumulada)
print(f"\n📊 Análisis de precisión vs número de frecuencias:")
print(f"{'Tau':<8} {'K':<6} {'Energía %':<12} {'MSE':<15} {'PSNR (dB)':<12}")
print("-" * 70)

tau_values = [0.90, 0.95, 0.97, 0.99, 0.995, 0.999, 1.0]
results = []

for tau in tau_values:
    k = select_k_by_energy(coeffs, tau=tau)
    z_recon = reconstruct_from_coeffs(coeffs, k, n_time=1000)
    
    # Interpolar para comparar
    t_orig = np.linspace(0, 1, len(z), endpoint=False)
    t_recon = np.linspace(0, 1, len(z_recon), endpoint=False)
    z_orig_interp = (
        np.interp(t_recon, t_orig, np.real(z))
        + 1j * np.interp(t_recon, t_orig, np.imag(z))
    )
    
    mse_val = mse(z_orig_interp, z_recon)
    psnr_val = psnr(z_orig_interp, z_recon)
    
    # Calcular energía acumulada
    from fourier_draw.metrics import cumulative_energy, spectral_energy
    energy_total = spectral_energy(coeffs)
    energy_accum = cumulative_energy(coeffs, k)
    energy_pct = (energy_accum / energy_total * 100) if energy_total > 0 else 0
    
    results.append((tau, k, energy_pct, mse_val, psnr_val))
    print(f"{tau:<8.3f} {k:<6} {energy_pct:>10.2f}% {mse_val:>14.6e} {psnr_val:>10.2f}")

# Determinar qué se necesita para "exacto"
print(f"\n🎯 Análisis de precisión:")
print(f"\nPara reconstrucción 'exacta' (PSNR > 40 dB):")
exact_results = [r for r in results if r[4] > 40]
if exact_results:
    best = exact_results[0]
    print(f"   • Tau mínimo: {best[0]:.3f}")
    print(f"   • Frecuencias necesarias: {best[1]}")
    print(f"   • PSNR: {best[4]:.2f} dB")
else:
    print(f"   ⚠️  Con las frecuencias disponibles, no se alcanza PSNR > 40 dB")
    print(f"   • Mejor resultado: {results[-1][4]:.2f} dB con {results[-1][1]} frecuencias")
    print(f"   • Para mejorar: aumentar SAMPLES o usar todas las frecuencias")

# Análisis de la forma del logo
print(f"\n📐 Análisis de la complejidad del logo:")
print(f"   • Formas angulares: Requieren muchas frecuencias altas")
print(f"   • Letras CHANEL: Formas complejas con esquinas agudas")
print(f"   • Logo CC: Curvas suaves pero intercaladas")
print(f"   • Total: Forma MUY compleja para Fourier")

# Recomendaciones
print(f"\n💡 Recomendaciones:")
print(f"\n1. Para mejor precisión con este logo:")
print(f"   • Usar tau = 0.999 o 1.0 (todas las frecuencias)")
print(f"   • Aumentar SAMPLES a 2048 o más")
print(f"   • Esto requerirá más tiempo de procesamiento")

print(f"\n2. Logos más sencillos que funcionarían mejor:")
print(f"   ✅ Círculo simple")
print(f"   ✅ Cuadrado")
print(f"   ✅ Estrella de 5 puntas")
print(f"   ✅ Corazón")
print(f"   ✅ Letra simple (A, O, C)")
print(f"   ✅ Flecha")
print(f"   ✅ Triángulo")

print(f"\n3. Características de logos 'fáciles':")
print(f"   • Formas suaves (sin esquinas agudas)")
print(f"   • Poca complejidad (1-3 elementos)")
print(f"   • Sin texto pequeño")
print(f"   • Contornos simples")

print(f"\n" + "=" * 70)

# Gráfico de precisión vs frecuencias
print(f"\n📈 Generando gráfico de precisión...")
tau_plot = [r[0] for r in results]
k_plot = [r[1] for r in results]
psnr_plot = [r[4] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: PSNR vs K
ax1.plot(k_plot, psnr_plot, 'b-o', linewidth=2, markersize=8)
ax1.axhline(y=40, color='r', linestyle='--', label='PSNR = 40 dB (Excelente)')
ax1.axhline(y=30, color='orange', linestyle='--', label='PSNR = 30 dB (Bueno)')
ax1.set_xlabel('Número de Frecuencias (K)', fontsize=12)
ax1.set_ylabel('PSNR (dB)', fontsize=12)
ax1.set_title('Precisión vs Número de Frecuencias', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Gráfico 2: PSNR vs Tau
ax2.plot(tau_plot, psnr_plot, 'g-s', linewidth=2, markersize=8)
ax2.axhline(y=40, color='r', linestyle='--', label='PSNR = 40 dB (Excelente)')
ax2.axhline(y=30, color='orange', linestyle='--', label='PSNR = 30 dB (Bueno)')
ax2.set_xlabel('Tau (Energía Acumulada)', fontsize=12)
ax2.set_ylabel('PSNR (dB)', fontsize=12)
ax2.set_title('Precisión vs Tau', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("analisis_precision.png", dpi=150, bbox_inches='tight')
print(f"✓ Gráfico guardado: analisis_precision.png")

print(f"\n✅ Análisis completado!")

