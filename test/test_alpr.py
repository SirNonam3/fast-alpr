"""
Test ALPR end-to-end.
"""

from pathlib import Path

import cv2
import pytest

from fast_alpr.alpr import ALPR

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


@pytest.mark.parametrize(
    "img_path, expected_plates", [(ASSETS_DIR / "test_image.png", {"5AU5341"})]
)
def test_end_to_end(img_path: Path, expected_plates: set[str]) -> None:
    im = cv2.imread(str(img_path))
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_hub_ocr_model="european-plates-mobile-vit-v2-model",
    )
    actual_result = alpr.predict(im)
    actual_plates = {x.ocr.text for x in actual_result if x.ocr is not None}
    assert actual_plates == expected_plates
