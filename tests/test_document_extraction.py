"""
Test document text and image extraction and forensics analysis.
Verifies that:
1. Text is extracted from the document and analyzed with TextDeepfakeDetector.
2. Embedded images (photos/stamps) are extracted and analyzed with ImageDeepfakeDetector.
"""

import io
import os
import unittest
import numpy as np
from PIL import Image

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from detector.document_detector import DocumentForensicsDetector


class TestDocumentExtractionAndForensics(unittest.TestCase):
    def setUp(self):
        self.detector = DocumentForensicsDetector()

    def test_scanned_image_document(self):
        # Create a mock scanned invoice image
        doc_img = Image.new("RGB", (400, 600), color=(245, 245, 245))
        res = self.detector.analyze(doc_img)
        self.assertEqual(res["modality"], "document")
        self.assertIn("extraction_telemetry", res)
        self.assertIn("extracted_images_count", res["extraction_telemetry"])
        self.assertIn("fake_probability", res)

    @unittest.skipUnless(HAS_PYPDF, "pypdf not installed in current environment")
    def test_pdf_with_text_and_embedded_image(self):
        # 1. Create an embedded JPEG image (with synthetic grid pattern)
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[::4, ::4] = 255
        img = Image.fromarray(arr)
        bio = io.BytesIO()
        img.save(bio, format="JPEG")
        jpeg_bytes = bio.getvalue()

        # 2. Build PDF in memory
        writer = pypdf.PdfWriter()
        page = writer.add_blank_page(width=400, height=600)

        # Embedded image XObject with indirect reference
        image_obj = pypdf.generic.DecodedStreamObject()
        image_obj.set_data(jpeg_bytes)
        image_obj.update({
            pypdf.generic.NameObject("/Type"): pypdf.generic.NameObject("/XObject"),
            pypdf.generic.NameObject("/Subtype"): pypdf.generic.NameObject("/Image"),
            pypdf.generic.NameObject("/Width"): pypdf.generic.NumberObject(64),
            pypdf.generic.NameObject("/Height"): pypdf.generic.NumberObject(64),
            pypdf.generic.NameObject("/ColorSpace"): pypdf.generic.NameObject("/DeviceRGB"),
            pypdf.generic.NameObject("/BitsPerComponent"): pypdf.generic.NumberObject(8),
            pypdf.generic.NameObject("/Filter"): pypdf.generic.NameObject("/DCTDecode"),
        })

        img_ref = writer._add_object(image_obj)
        page[pypdf.generic.NameObject("/Resources")] = pypdf.generic.DictionaryObject({
            pypdf.generic.NameObject("/XObject"): pypdf.generic.DictionaryObject({
                pypdf.generic.NameObject("/Im1"): img_ref
            })
        })

        # Add text content stream with typical LLM markers
        content = pypdf.generic.DecodedStreamObject()
        content.set_data(b"BT /F1 12 Tf 50 500 Td (Furthermore, it is crucial to delve into the tapestry of artificial intelligence.) Tj ET")
        page[pypdf.generic.NameObject("/Contents")] = content

        out = io.BytesIO()
        writer.write(out)
        pdf_bytes = out.getvalue()

        # 3. Run Document Detector analysis
        res = self.detector.analyze(pdf_bytes, is_pdf=True)

        self.assertEqual(res["modality"], "document")
        self.assertIn("extraction_telemetry", res)
        # Verify text was extracted
        self.assertTrue(len(res["extraction_telemetry"]["extracted_text_snippet"]) > 0)
        self.assertIsNotNone(res["extraction_telemetry"]["text_analysis"])
        # Verify embedded image was extracted and analyzed
        self.assertEqual(res["extraction_telemetry"]["extracted_images_count"], 1)
        self.assertEqual(len(res["extraction_telemetry"]["image_analyses"]), 1)

        print("\nExtracted Text Snippet:", res["extraction_telemetry"]["extracted_text_snippet"])
        print("Extracted Images Count:", res["extraction_telemetry"]["extracted_images_count"])
        print("Document Analysis Verdict:", res["classification"])
        print("Forensic Signals:", res["forensic_signals"])


if __name__ == "__main__":
    unittest.main()
