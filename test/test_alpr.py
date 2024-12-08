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
def test_default_alpr(img_path: Path, expected_plates: set[str]) -> None:
    im = cv2.imread(str(img_path))
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="global-plates-mobile-vit-v2-model",
    )
    actual_result = alpr.predict(im)
    actual_plates = {x.ocr.text for x in actual_result if x.ocr is not None}
    assert actual_plates == expected_plates

    for res in actual_result:
        bbox = res.detection.bounding_box
        height, width = im.shape[:2]
        x1, y1 = max(bbox.x1, 0), max(bbox.y1, 0)
        x2, y2 = min(bbox.x2, width), min(bbox.y2, height)

        assert 0 <= x1 < width, f"x1 coordinate {x1} out of bounds (0, {width})"
        assert 0 <= x2 <= width, f"x2 coordinate {x2} out of bounds (0, {width})"
        assert 0 <= y1 < height, f"y1 coordinate {y1} out of bounds (0, {height})"
        assert 0 <= y2 <= height, f"y2 coordinate {y2} out of bounds (0, {height})"
        assert x1 < x2, f"x1 ({x1}) should be less than x2 ({x2})"
        assert y1 < y2, f"y1 ({y1}) should be less than y2 ({y2})"

        if res.ocr is not None:
            conf = res.ocr.confidence
            if isinstance(conf, list):
                assert all(0.0 <= x <= 1.0 for x in conf)
            elif isinstance(conf, float):
                assert 0.0 <= conf <= 1.0
            else:
                raise TypeError(f"Unexpected type for confidence: {type(conf).__name__}")
