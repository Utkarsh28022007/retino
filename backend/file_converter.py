"""
RetinaSetu - Universal Medical Image & Document Converter
Converts any incoming file (JPG, JPEG, PNG, WEBP, TIFF, BMP, PDF, DICOM, etc.)
into a clean RGB PIL Image for the AI screening pipeline.
"""

import io
import re
import numpy as np
from PIL import Image

def load_medical_file_to_pil(file_bytes: bytes, filename: str = "") -> Image.Image:
    """
    Parses and loads any file format into a PIL RGB Image.
    Supports PDF document extraction, JPEG, PNG, WEBP, TIFF, BMP, and DICOM streams.
    """
    if not file_bytes:
        raise ValueError("Received empty file payload.")

    is_pdf = filename.lower().endswith(".pdf") or file_bytes.startswith(b"%PDF")

    if is_pdf:
        # 1. Try rendering page via pypdfium2
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(file_bytes)
            if len(pdf) > 0:
                page = pdf[0]
                # Render at 2x resolution (144 dpi) for high clinical quality
                pil_img = page.render(scale=2.0).to_pil().convert("RGB")
                return pil_img
        except Exception as e_pdfium:
            print(f"[FileConverter] pypdfium2 rendering fallback: {e_pdfium}")

        # 2. Fallback: Search for embedded JPEG/PNG image stream in PDF binary
        try:
            # Look for JPEG SOI/EOI markers: \xff\xd8 ... \xff\xd9
            jpeg_start = file_bytes.find(b"\xff\xd8")
            if jpeg_start != -1:
                jpeg_end = file_bytes.rfind(b"\xff\xd9")
                if jpeg_end > jpeg_start:
                    jpeg_data = file_bytes[jpeg_start : jpeg_end + 2]
                    return Image.open(io.BytesIO(jpeg_data)).convert("RGB")
        except Exception as e_jpeg:
            print(f"[FileConverter] JPEG stream extraction failed: {e_jpeg}")

        # 3. Fallback: Search for embedded PNG markers: \x89PNG ... IEND
        try:
            png_start = file_bytes.find(b"\x89PNG\r\n\x1a\n")
            if png_start != -1:
                png_end = file_bytes.rfind(b"IEND")
                if png_end > png_start:
                    png_data = file_bytes[png_start : png_end + 8]
                    return Image.open(io.BytesIO(png_data)).convert("RGB")
        except Exception as e_png:
            print(f"[FileConverter] PNG stream extraction failed: {e_png}")

        raise ValueError(
            "PDF uploaded, but no embedded fundus image stream or renderer was found. "
            "Please ensure the PDF contains a retinal photograph."
        )

    # Standard Raster Image Formats: JPG, JPEG, PNG, WEBP, BMP, TIFF
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        # Handle RGBA / Transparency by compositing onto black background (standard fundus black surround)
        if pil_img.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", pil_img.size, (0, 0, 0))
            if pil_img.mode == "P":
                pil_img = pil_img.convert("RGBA")
            bg.paste(pil_img, mask=pil_img.split()[-1])
            return bg
        return pil_img.convert("RGB")
    except Exception as e_pil:
        raise ValueError(
            f"Unsupported or corrupted image format ({filename}): {str(e_pil)}. "
            "Supported formats: JPG, JPEG, PNG, WEBP, TIFF, BMP, PDF."
        )
