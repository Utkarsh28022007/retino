import math

def polar_to_cart(cx, cy, r, angle_deg):
    rad = math.radians(angle_deg)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)

def describe_arc(cx, cy, r, start_deg, end_deg):
    x1, y1 = polar_to_cart(cx, cy, r, start_deg)
    x2, y2 = polar_to_cart(cx, cy, r, end_deg)
    delta = (end_deg - start_deg) % 360
    large_arc = 1 if delta > 180 else 0
    return f"M {x1:.2f} {y1:.2f} A {r} {r} 0 {large_arc} 1 {x2:.2f} {y2:.2f}"

cx, cy = 50, 50
r = 40
sw = 8.5

# Look carefully at the user image:
# Bottom arc: spans from ~3 deg to ~177 deg (clockwise from 3 o'clock through 6 o'clock to 9 o'clock)
arc_bottom = describe_arc(cx, cy, r, 3, 177)

# Top-left arc: from ~183 deg to ~267 deg (from 9 o'clock up to ~12 o'clock)
arc_top_left = describe_arc(cx, cy, r, 183, 267)

# Top-mid arc: from ~273 deg to ~321 deg (from ~12 o'clock to ~1:45)
arc_top_mid = describe_arc(cx, cy, r, 273, 321)

# Top-right arc: from ~327 deg to ~357 deg (small light blue arc around 2 o'clock to 3 o'clock)
arc_top_right = describe_arc(cx, cy, r, 327, 357)

# Cutout center for the pupil
# In the image, the notch is around 323 deg (pointing straight between the top-mid and top-right arcs)
cutout_r = 11.5
cutout_dist = 22
cutout_x, cutout_y = polar_to_cart(cx, cy, cutout_dist, 323)

svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100%" height="100%">
  <defs>
    <mask id="retina-pupil-cutout">
      <rect x="0" y="0" width="100" height="100" fill="white" />
      <circle cx="{cutout_x:.2f}" cy="{cutout_y:.2f}" r="{cutout_r}" fill="black" />
    </mask>
  </defs>

  <!-- Segmented Outer Ring -->
  <!-- Bottom Arc (Royal Navy Blue) -->
  <path d="{arc_bottom}" fill="none" stroke="#253592" stroke-width="{sw}" stroke-linecap="butt" />

  <!-- Top-Left Arc (Cerulean Blue) -->
  <path d="{arc_top_left}" fill="none" stroke="#009ee3" stroke-width="{sw}" stroke-linecap="butt" />

  <!-- Top-Middle Arc (Sky Blue) -->
  <path d="{arc_top_mid}" fill="none" stroke="#00a7e7" stroke-width="{sw}" stroke-linecap="butt" />

  <!-- Top-Right Small Arc (Light Cyan Blue) -->
  <path d="{arc_top_right}" fill="none" stroke="#7cd4f7" stroke-width="{sw}" stroke-linecap="butt" />

  <!-- Central Pupil with Circular Reflection Notch -->
  <circle cx="{cx}" cy="{cy}" r="24.5" fill="#18181b" mask="url(#retina-pupil-cutout)" />
</svg>
"""

with open("frontend/retina_logo.svg", "w", encoding="utf-8") as f:
    f.write(svg_content)

print("Generated frontend/retina_logo.svg successfully")
