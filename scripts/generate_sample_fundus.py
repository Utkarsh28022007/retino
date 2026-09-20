"""
RetinaSetu - Realistic Synthetic Clinical Fundus Generator
Generates clinical benchmark fundus images representing ICDR Grades 0 to 4
and ungradeable coaxial flash glare for testing and demonstrations.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
os.makedirs(OUT_DIR, exist_ok=True)

def create_base_retina(width=600, height=600):
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    center_x, center_y = width // 2, height // 2
    retina_radius = int(min(width, height) * 0.44)

    # Gradient retinal fundus base (orange-reddish choroid)
    for r in range(retina_radius, 0, -2):
        factor = r / retina_radius
        # Center is deeper orange-red, edges fall off slightly
        red = int(185 * (1 - 0.25 * (factor**2)))
        green = int(75 * (1 - 0.35 * (factor**2)))
        blue = int(18 * (1 - 0.4 * (factor**2)))
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], fill=(red, green, blue))
    
    # Optic Disc (yellowish-orange oval in nasal quadrant, e.g. Left eye -> left side)
    od_x = center_x - int(retina_radius * 0.48)
    od_y = center_y + int(retina_radius * 0.05)
    od_rx = int(retina_radius * 0.16)
    od_ry = int(retina_radius * 0.19)
    for r in range(od_rx, 0, -1):
        f = r / od_rx
        od_r = int(245 - 20 * f)
        od_g = int(210 - 30 * f)
        od_b = int(120 - 40 * f)
        draw.ellipse([od_x - r, od_y - int(r*1.18), od_x + r, od_y + int(r*1.18)], fill=(od_r, od_g, od_b))
    # Optic cup
    draw.ellipse([od_x - od_rx//2, od_y - od_ry//2, od_x + od_rx//2, od_y + od_ry//2], fill=(255, 235, 170))

    # Fovea / Macula (dark avascular zone temporal to disc)
    fovea_x = center_x + int(retina_radius * 0.22)
    fovea_y = od_y + int(retina_radius * 0.04)
    fovea_r = int(retina_radius * 0.22)
    for r in range(fovea_r, 0, -2):
        f = (fovea_r - r) / fovea_r
        # Darker choroidal pigmentation
        m_r = int(140 - 35 * f)
        m_g = int(45 - 15 * f)
        m_b = int(10)
        draw.ellipse([fovea_x - r, fovea_y - r, fovea_x + r, fovea_y + r], fill=(m_r, m_g, m_b))

    # Blood vessel arcades branching out of the optic disc
    vessel_color = (120, 20, 10)
    vessel_draw = ImageDraw.Draw(img)

    # Superior temporal arcade
    pts_sup = [
        (od_x + 5, od_y - 10),
        (od_x + 60, od_y - 80),
        (od_x + 140, od_y - 130),
        (center_x + 70, od_y - 145),
        (fovea_x + 80, od_y - 110),
        (fovea_x + 140, od_y - 50)
    ]
    for i in range(len(pts_sup)-1):
        width_line = max(1, 4 - i)
        vessel_draw.line([pts_sup[i], pts_sup[i+1]], fill=vessel_color, width=width_line)

    # Inferior temporal arcade
    pts_inf = [
        (od_x + 5, od_y + 10),
        (od_x + 60, od_y + 80),
        (od_x + 140, od_y + 130),
        (center_x + 70, od_y + 145),
        (fovea_x + 80, od_y + 110),
        (fovea_x + 140, od_y + 50)
    ]
    for i in range(len(pts_inf)-1):
        width_line = max(1, 4 - i)
        vessel_draw.line([pts_inf[i], pts_inf[i+1]], fill=vessel_color, width=width_line)

    # Nasal branches
    vessel_draw.line([(od_x - 5, od_y - 5), (od_x - 60, od_y - 70), (od_x - 100, od_y - 100)], fill=vessel_color, width=3)
    vessel_draw.line([(od_x - 5, od_y + 5), (od_x - 60, od_y + 70), (od_x - 100, od_y + 100)], fill=vessel_color, width=3)

    # Small arterioles and venules
    for angle in [0.3, 0.7, 1.2, -0.4, -0.9, -1.3]:
        p0 = (od_x + int(20 * math.cos(angle)), od_y + int(20 * math.sin(angle)))
        p1 = (od_x + int(80 * math.cos(angle)), od_y + int(80 * math.sin(angle)))
        vessel_draw.line([p0, p1], fill=vessel_color, width=2)

    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img, (od_x, od_y), (fovea_x, fovea_y), retina_radius

def add_microaneurysms(img, count=5, seed=42):
    np.random.seed(seed)
    w, h = img.size
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    for _ in range(count):
        # Place near vascular arcades or macula perimeter
        ang = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(50, 180)
        x = int(cx + dist * np.cos(ang))
        y = int(cy + dist * np.sin(ang))
        r = np.random.randint(1, 3) # tiny pinpoint red dots
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(85, 10, 5))
    return img

def add_hard_exudates(img, clusters=3, seed=42):
    np.random.seed(seed)
    w, h = img.size
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    for c in range(clusters):
        # Clusters of bright yellowish waxy dots
        base_x = int(cx + np.random.uniform(30, 140))
        base_y = int(cy + np.random.uniform(-80, 80))
        num_dots = np.random.randint(15, 35)
        for _ in range(num_dots):
            dx = int(np.random.normal(0, 12))
            dy = int(np.random.normal(0, 12))
            r = np.random.randint(2, 5)
            x = base_x + dx
            y = base_y + dy
            # Waxy yellow lipid color
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(245, 230, 120))
    return img

def add_hemorrhages(img, count=6, seed=42):
    np.random.seed(seed)
    w, h = img.size
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    for _ in range(count):
        ang = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(40, 200)
        x = int(cx + dist * np.cos(ang))
        y = int(cy + dist * np.sin(ang))
        rx = np.random.randint(5, 14)
        ry = np.random.randint(4, 10)
        # Dark deep red blot hemorrhages
        draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=(70, 12, 8))
    return img

def add_neovascularization(img, od_loc, seed=42):
    np.random.seed(seed)
    draw = ImageDraw.Draw(img)
    od_x, od_y = od_loc
    # Fine network of abnormal friable new vessels (NVD) emanating around the optic disc
    for i in range(16):
        ang = np.random.uniform(-np.pi/2, np.pi/2)
        cur_x, cur_y = od_x + np.random.randint(-10, 10), od_y + np.random.randint(-15, 15)
        length = np.random.randint(40, 95)
        for step in range(length // 4):
            nxt_x = cur_x + int(4 * np.cos(ang) + np.random.randint(-2, 3))
            nxt_y = cur_y + int(4 * np.sin(ang) + np.random.randint(-3, 4))
            draw.line([(cur_x, cur_y), (nxt_x, nxt_y)], fill=(135, 18, 12), width=1)
            cur_x, cur_y = nxt_x, nxt_y
    
    # Also add pre-retinal sub-hyaloid boat-shaped hemorrhage
    draw.chord([od_x + 70, od_y + 20, od_x + 180, od_y + 110], 0, 180, fill=(60, 5, 5))
    return img

def add_coaxial_glare_and_blur(img):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    # Intense coaxial flash reflection spot (glare)
    glare_x, glare_y = w // 2 - 40, h // 2 - 30
    for r in range(45, 0, -2):
        alpha_val = int(255 * (1 - r / 45))
        draw.ellipse([glare_x - r, glare_y - r, glare_x + r, glare_y + r], fill=(255, 255, 240))
    # Severe motion blur
    img = img.filter(ImageFilter.GaussianBlur(radius=6.0))
    return img

def main():
    print("Generating clinical benchmark retinal images for RetinaSetu...")
    # Grade 0: No DR
    img0, od, fovea, r = create_base_retina()
    img0.save(os.path.join(OUT_DIR, "grade0_normal.png"))

    # Grade 1: Mild NPDR (Microaneurysms only)
    img1, od, fovea, r = create_base_retina()
    img1 = add_microaneurysms(img1, count=4, seed=101)
    img1.save(os.path.join(OUT_DIR, "grade1_mild.png"))

    # Grade 2: Moderate NPDR (Hard Exudates + MAs + Blots)
    img2, od, fovea, r = create_base_retina()
    img2 = add_microaneurysms(img2, count=14, seed=202)
    img2 = add_hard_exudates(img2, clusters=2, seed=203)
    img2 = add_hemorrhages(img2, count=4, seed=204)
    img2.save(os.path.join(OUT_DIR, "grade2_moderate.png"))

    # Grade 3: Severe NPDR (Extensive 4-quadrant hemorrhages, dense exudates, venous beading)
    img3, od, fovea, r = create_base_retina()
    img3 = add_microaneurysms(img3, count=38, seed=301)
    img3 = add_hard_exudates(img3, clusters=5, seed=302)
    img3 = add_hemorrhages(img3, count=16, seed=303)
    img3.save(os.path.join(OUT_DIR, "grade3_severe.png"))

    # Grade 4: Proliferative DR (NVD fronds + Preretinal hemorrhage + Exudates)
    img4, od, fovea, r = create_base_retina()
    img4 = add_microaneurysms(img4, count=45, seed=401)
    img4 = add_hard_exudates(img4, clusters=6, seed=402)
    img4 = add_hemorrhages(img4, count=20, seed=403)
    img4 = add_neovascularization(img4, od, seed=404)
    img4.save(os.path.join(OUT_DIR, "grade4_pdr.png"))

    # Ungradeable: Coaxial Flash Glare & Motion Blur
    img_glare, od, fovea, r = create_base_retina()
    img_glare = add_coaxial_glare_and_blur(img_glare)
    img_glare.save(os.path.join(OUT_DIR, "ungradeable_glare.png"))

    print("Generated all 6 benchmark images in sample_data/:")
    for f in os.listdir(OUT_DIR):
        print(f" - {f}")

if __name__ == "__main__":
    main()
