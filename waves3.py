"""
Simulación de propagación de ondas esféricas en un medio con viento.
RECEPTOR FIJO - La onda llega a él en el tiempo Δt.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

# Configuración de la figura
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(-6, 28)
ax.set_ylim(-10, 18)
ax.set_aspect('equal')
ax.axis('off')

# Parámetros físicos
c = 6.0           # Velocidad del sonido en el medio (m/s)
v_viento = 2.5    # Velocidad del viento (m/s)
alpha = np.pi/8   # Ángulo del viento respecto a la horizontal (22.5°)

# POSICIONES FIJAS
emisor_pos = np.array([0, 0])      # Emisor: SIEMPRE fijo en origen
receptor_pos = np.array([18, 0]) # Receptor: SIEMPRE fijo

# Calcular el tiempo Δt necesario para que la onda llegue al receptor
def calcular_dt_llegada():
    dx = receptor_pos[0] - emisor_pos[0]
    dy = receptor_pos[1] - emisor_pos[1]
    ux, uy = np.cos(alpha), np.sin(alpha)

    A = v_viento**2 * (ux**2 + uy**2) - c**2
    B = -2 * v_viento * (dx*ux + dy*uy)
    C = dx**2 + dy**2

    discriminante = B**2 - 4*A*C
    dt1 = (-B + np.sqrt(discriminante)) / (2*A)
    dt2 = (-B - np.sqrt(discriminante)) / (2*A)

    dt_llegada = min(dt for dt in [dt1, dt2] if dt > 0)
    return dt_llegada

dt_llegada = calcular_dt_llegada()
L = np.linalg.norm(receptor_pos - emisor_pos)
c_aparente = L / dt_llegada

def calcular_estado(t):
    """Calcula el estado de la simulación en el tiempo t."""
    desplazamiento_viento = v_viento * t * np.array([np.cos(alpha), np.sin(alpha)])
    punto_emision = emisor_pos + desplazamiento_viento
    radio_onda = c * t
    distancia_centro_a_receptor = np.linalg.norm(receptor_pos - punto_emision)
    onda_llego = distancia_centro_a_receptor <= radio_onda

    return punto_emision, radio_onda, desplazamiento_viento, onda_llego

# Elementos gráficos
onda_circulo, = ax.plot([], [], 'k--', linewidth=2.5)
centro_onda = ax.plot([], [], 's', color='orange', markersize=12,
                      markeredgecolor='red', markeredgewidth=2)[0]
emisor_plot = ax.plot([], [], 's', color='red', markersize=14,
                      markeredgecolor='darkred', markeredgewidth=2)[0]
receptor_plot = ax.plot([], [], 's', color='orange', markersize=12,
                        markeredgecolor='red', markeredgewidth=2)[0]

flecha_viento = FancyArrowPatch((0, 0), (0, 0),
                                arrowstyle='->', mutation_scale=25,
                                color='blue', linewidth=2.5)
ax.add_patch(flecha_viento)

linea_emision, = ax.plot([], [], 'b:', linewidth=1.5)
linea_directa, = ax.plot([], [], 'k-', linewidth=1.2)
linea_horizontal, = ax.plot([], [], 'k:', linewidth=1.5)

texto_info = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow',
                              edgecolor='black', alpha=0.9))

texto_leyenda = ax.text(0.98, 0.98, '', transform=ax.transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white',
                                 edgecolor='gray', alpha=0.9))

# ANOTACIÓN CON FLECHA QUE APUNTA AL CENTRO DE LA ONDA (dinámica)
anotacion_centro = ax.annotate('', xy=(0, 0), xytext=(8, 12),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=9, color='black',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9),
            zorder=10)

def init():
    onda_circulo.set_data([], [])
    centro_onda.set_data([], [])
    emisor_plot.set_data([emisor_pos[0]], [emisor_pos[1]])
    receptor_plot.set_data([receptor_pos[0]], [receptor_pos[1]])
    flecha_viento.set_positions((0, 0), (0, 0))
    linea_emision.set_data([], [])
    linea_directa.set_data([], [])
    linea_horizontal.set_data([], [])
    texto_info.set_text('')
    texto_leyenda.set_text('')
    anotacion_centro.set_visible(False)
    return (onda_circulo, centro_onda, emisor_plot, receptor_plot,
            flecha_viento, linea_emision, linea_directa, linea_horizontal,
            texto_info, texto_leyenda, anotacion_centro)

def animate(t):
    punto_emision, radio_onda, desplazamiento, onda_llego = calcular_estado(t)

    # Dibujar círculo de onda
    theta = np.linspace(0, 2*np.pi, 200)
    circulo_x = punto_emision[0] + radio_onda * np.cos(theta)
    circulo_y = punto_emision[1] + radio_onda * np.sin(theta)
    onda_circulo.set_data(circulo_x, circulo_y)

    # Actualizar posiciones
    centro_onda.set_data([punto_emision[0]], [punto_emision[1]])
    emisor_plot.set_data([emisor_pos[0]], [emisor_pos[1]])
    receptor_plot.set_data([receptor_pos[0]], [receptor_pos[1]])

    flecha_viento.set_positions(
        (emisor_pos[0], emisor_pos[1]),
        (punto_emision[0], punto_emision[1])
    )

    linea_emision.set_data([emisor_pos[0], punto_emision[0]],
                           [emisor_pos[1], punto_emision[1]])
    linea_directa.set_data([punto_emision[0], receptor_pos[0]],
                           [punto_emision[1], receptor_pos[1]])
    linea_horizontal.set_data([emisor_pos[0], receptor_pos[0]],
                              [emisor_pos[1], receptor_pos[1]])

    # === CLAVE: Actualizar la anotación para que apunte al centro de la onda ===
    # El texto se mantiene fijo en (8, 12), pero la flecha apunta a punto_emision
    anotacion_centro.xy = (punto_emision[0], punto_emision[1])  # ← DESTINO DE LA FLECHA
    anotacion_centro.set_text(
        'Place to which the point of the medium in\n which the wave has been emitted has\ndisplaced after Δt '
        'due to wind velocity'
    )
    anotacion_centro.set_visible(True)

    # Información dinámica
    viento_dt = v_viento * t
    c_dt = c * t

    dx = receptor_pos[0] - punto_emision[0]
    dy = receptor_pos[1] - punto_emision[1]
    beta = -np.arctan2(dy, dx)

    proy_viento = viento_dt * np.cos(alpha)
    proy_onda = c_dt * np.cos(beta)

    info_text = (
        f'  Time: t = {t:.3f} s / Δt = {dt_llegada:.3f} s\n'
        f'\n'
        f'  Displacement of medium:\n'
        f'    v_wind·Δt = {viento_dt:.2f} m\n'
        f'    α angle= {np.degrees(alpha):.1f}°\n'
        f'    center of the wave: ({punto_emision[0]:.2f}, {punto_emision[1]:.2f})\n'
        f'\n'
        f'  Propagation of wave:\n'
        f'    c·Δt = {c_dt:.2f} m\n'
        f'    β angle= {np.degrees(beta):.1f}°\n'
        f'\n'
        f'  Source-observer distance:\n'
        f'    L = c\'·Δt = {L:.2f} m\n'
        f'\n'
        f'  Equation:\n'
        f'    L = v_wind·Δt·cos(α) + c·Δt·cos(β)\n'
        f'    {L:.2f} = {proy_viento:.2f} + {proy_onda:.2f}\n'
        f'\n  Apparent velocity:\n'
        f'    c\' = L/t = {L/t:.2f} m/s (instant)\n'
        f'    Final c\' final = {c_aparente:.2f} m/s\n'

    )

    estado = "✓ Wave arrived at observer" if onda_llego else "○ Wave propagating..."
    info_text += f'\n  {estado}'

    texto_info.set_text(info_text)

    leyenda_text = (
        'Legend:\n'
        '  ■ Red: Source (fixed at t=0)\n'
        '  ■ Orange: Observer (fixed)\n'
        '  ■ Orange: Center of wave\n'
        '      (moves with the wind)\n'
        '  --- Dashed black: spherical wave\n'
        '  → Blue: wind vector·Δt\n'
        '  ··· Blue: path of the emitting point\n'
        '  — Black: path of the wave c·Δt\n'
        '  ··· Black: Apparent distance L = c\'·Δt'
    )
    texto_leyenda.set_text(leyenda_text)

    return (onda_circulo, centro_onda, emisor_plot, receptor_plot,
            flecha_viento, linea_emision, linea_directa, linea_horizontal,
            texto_info, texto_leyenda, anotacion_centro)

# Anotaciones estáticas
ax.annotate('Source\n(emits wave at t=0)',
            xy=(emisor_pos[0], emisor_pos[1]), xytext=(-2, -2),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10, color='red', fontweight='bold', ha='center')

ax.annotate('Observer\n(fixed)',
            xy=(receptor_pos[0], receptor_pos[1]), xytext=(receptor_pos[0]+2, -2),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10, color='red', fontweight='bold')

# Flecha del viento en leyenda
ax.annotate('', xy=(3*np.cos(alpha), 3*np.sin(alpha)), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.text(3.5*np.cos(alpha), 3.5*np.sin(alpha), r'$\vec{v}_{wind}$',
        fontsize=12, color='blue', fontweight='bold')

ax.set_title('Moving material medium (wind)\n'
             'Spherical wave with stationary source and observer',
             fontsize=14, fontweight='bold', pad=20)

# Crear animación
t_vals = np.linspace(0.01, dt_llegada * 1, 150)


pausado = [False]

def toggle_pausa(event):
    if pausado[0]:
        anim.event_source.start()
        boton.label.set_text('Pause')
    else:
        anim.event_source.stop()
        boton.label.set_text('Continue')
    pausado[0] = not pausado[0]

# Crear área del botón (debajo de la figura)
ax_boton = fig.add_axes([0.35, 0.01, 0.15, 0.05])
boton = Button(ax_boton, 'Pause', color='lightblue', hovercolor='skyblue')
boton.on_clicked(toggle_pausa)

# Crear animación (sin repeat como pediste antes)
anim = FuncAnimation(fig, animate, init_func=init, frames=t_vals,
                     interval=10, blit=True, repeat=True)

plt.show()
plt.tight_layout()
plt.show()
