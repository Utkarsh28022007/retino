"""
RetinaSetu - Automated UI Verification & Screenshot Capturer
Uses Playwright to interactively test and capture all views of RetinaSetu.
"""
import os
import asyncio
from playwright.async_api import async_playwright

ARTIFACT_DIR = r"C:\Users\hs488\.gemini\antigravity-ide\brain\60374ac1-cb77-4278-99b8-6f49e502dfb1"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1400, "height": 900})
        page = await context.new_page()

        print("Navigating to RetinaSetu at http://127.0.0.1:8000/...")
        await page.goto("http://127.0.0.1:8000/", wait_until="networkidle")
        await page.wait_for_timeout(1000)

        # 1. Capture Screening Dashboard (Default Moderate NPDR loaded)
        print("Capturing 01_screening_dashboard.png...")
        await page.screenshot(path=os.path.join(ARTIFACT_DIR, "01_screening_dashboard.png"))

        # 2. Click Grad-CAM Overlay button
        print("Switching to Grad-CAM Overlay...")
        await page.click("button[data-layer='gradcam']")
        await page.wait_for_timeout(500)
        await page.screenshot(path=os.path.join(ARTIFACT_DIR, "02_gradcam_overlay.png"))

        # 3. Click Retinal Anatomical Layers Tab
        print("Navigating to Anatomical Layers Tab...")
        await page.click("#tab-btn-segmentation")
        await page.wait_for_timeout(500)
        await page.screenshot(path=os.path.join(ARTIFACT_DIR, "03_anatomical_layers.png"))

        # 4. Click Grad-CAM Explainability Tab
        print("Navigating to Explainability Tab...")
        await page.click("#tab-btn-explainability")
        await page.wait_for_timeout(500)
        await page.screenshot(path=os.path.join(ARTIFACT_DIR, "04_explainability_tab.png"))

        # 5. Open Clinical Referral Slip Modal
        print("Opening Clinical Referral Slip Modal...")
        await page.click("#btn-generate-report")
        await page.wait_for_timeout(500)
        await page.screenshot(path=os.path.join(ARTIFACT_DIR, "05_clinical_referral_slip.png"))

        # Close Modal
        await page.click("#btn-close-modal")
        await page.wait_for_timeout(300)

        # 6. Click Simulink District Optimizer Tab & Run Simulation
        print("Navigating to Simulink Optimizer Tab...")
        await page.click("#tab-btn-simulink")
        await page.wait_for_timeout(500)
        await page.click("#btn-run-simulation")
        await page.wait_for_timeout(1000)
        await page.screenshot(path=os.path.join(ARTIFACT_DIR, "06_simulink_optimizer.png"))

        # 7. Click Benchmarks & Validation Tab
        print("Navigating to Benchmarks & Validation Tab...")
        await page.click("#tab-btn-benchmarks")
        await page.wait_for_timeout(600)
        await page.screenshot(path=os.path.join(ARTIFACT_DIR, "07_benchmarks_table.png"))

        # 8. Test Ungradeable Preset
        print("Testing Ungradeable Glare Preset...")
        await page.click("#tab-btn-screening")
        await page.wait_for_timeout(300)
        await page.click(".preset-item[data-id='ungradeable_glare.png']")
        await page.wait_for_timeout(1000)
        await page.screenshot(path=os.path.join(ARTIFACT_DIR, "08_ungradeable_glare_rejection.png"))

        await browser.close()
        print("All UI verification screenshots captured successfully!")

if __name__ == "__main__":
    asyncio.run(main())
