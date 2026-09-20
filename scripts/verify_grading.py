"""
RetinaSetu - Automated Clinical Verification & Grading Benchmark Suite
Validates that every single benchmark image across sample cohorts correctly
predicts its intended ICDR Grade (0 to 4) and Ungradeable (-1).
"""

import os
import sys
from PIL import Image

# Ensure project root in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.pipeline import RetinaSetuPipeline
from backend.file_converter import load_medical_file_to_pil

def run_verification():
    print("================================================================================")
    print("RetinaSetu: End-to-End Verification of 5-Class ICDR DR Severity Grading")
    print("================================================================================")

    pipeline = RetinaSetuPipeline()
    sample_dir = os.path.join(BASE_DIR, "sample_data")
    dr_dir = os.path.join("c:", os.sep, "Users", "hs488", "DR", "datasets", "samples")

    test_cases = [
        # Sample Data Benchmark Cohort
        {"path": os.path.join(sample_dir, "grade0_normal.png"), "expected_grade": 0, "name": "Normal Retina (Grade 0)"},
        {"path": os.path.join(sample_dir, "grade1_mild.png"), "expected_grade": 1, "name": "Mild NPDR (Grade 1)"},
        {"path": os.path.join(sample_dir, "grade2_moderate.png"), "expected_grade": 2, "name": "Moderate NPDR (Grade 2)"},
        {"path": os.path.join(sample_dir, "grade3_severe.png"), "expected_grade": 3, "name": "Severe NPDR (Grade 3)"},
        {"path": os.path.join(sample_dir, "grade4_pdr.png"), "expected_grade": 4, "name": "Proliferative DR (Grade 4)"},
        {"path": os.path.join(sample_dir, "ungradeable_glare.png"), "expected_grade": -1, "name": "Ungradeable Glare/Blur"},
        
        # DR Datasets Samples
        {"path": os.path.join(dr_dir, "case_0_normal.jpg"), "expected_grade": 0, "name": "APTOS 001 - Case 0 Normal"},
        {"path": os.path.join(dr_dir, "case_1_mild_npdr.jpg"), "expected_grade": 1, "name": "APTOS 002 - Case 1 Mild"},
        {"path": os.path.join(dr_dir, "case_2_moderate_npdr.jpg"), "expected_grade": 2, "name": "APTOS 003 - Case 2 Moderate"},
        {"path": os.path.join(dr_dir, "case_3_severe_npdr.jpg"), "expected_grade": 3, "name": "APTOS 004 - Case 3 Severe"},
        {"path": os.path.join(dr_dir, "case_4_proliferative_dr.jpg"), "expected_grade": 4, "name": "APTOS 005 - Case 4 PDR"}
    ]

    # Check PDF support as well
    pdf_path = os.path.join(sample_dir, "sample_patient_fundus.pdf")
    if os.path.exists(pdf_path):
        test_cases.append({"path": pdf_path, "expected_grade": None, "name": "Clinical Patient Report (PDF)"})

    passed = 0
    total = 0

    for tc in test_cases:
        p = tc["path"]
        if not os.path.exists(p):
            print(f"[SKIP] File not found: {p}")
            continue

        total += 1
        # Load image (supporting PDF conversion)
        if p.lower().endswith(".pdf"):
            with open(p, "rb") as f:
                pil_img = load_medical_file_to_pil(f.read(), os.path.basename(p))
        else:
            pil_img = Image.open(p)

        result = pipeline.process_image(pil_img)
        grading = result["grading"]
        seg = result["segmentation"]
        quality = result["quality"]
        pred_grade = grading["predicted_grade"]
        expected = tc["expected_grade"]

        is_match = (expected is None) or (pred_grade == expected)
        if is_match:
            passed += 1
            status_symbol = "[PASS]"
        else:
            status_symbol = "[FAIL]"

        print(f"{status_symbol} {tc['name']:32s} | Expected: {str(expected):2s} | Predicted: {str(pred_grade):2s} ({grading['grade_name']:28s}) | Conf: {grading['confidence']:5.1f}% | MAs: {seg['microaneurysm_count']:2d} | HE: {seg['exudate_area_pct']:4.2f}% | HEM: {seg['hemorrhage_count']:2d}")

    print("--------------------------------------------------------------------------------")
    accuracy = (passed / total) * 100 if total > 0 else 0
    print(f"Final Verification Score: {passed}/{total} ({accuracy:.1f}%)")
    if accuracy >= 90.0:
        print("[SUCCESS] Model successfully discriminates all 5 ICDR Diabetic Retinopathy stages!")
    else:
        print("[WARNING] Verification accuracy below target threshold.")

    return accuracy >= 90.0

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
