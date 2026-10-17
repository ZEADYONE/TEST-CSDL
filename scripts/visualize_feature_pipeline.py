"""Create an auditable visual explanation of every handcrafted descriptor.

The script only reads an image and writes PNG/HTML diagnostics. It does not
modify PostgreSQL, stored feature vectors, or the web application.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.services.color_features import (  # noqa: E402
    JOINT_BINS,
    JOINT_DIM,
    SPATIAL_DIM,
    SPATIAL_REGION_DIM,
    SPATIAL_REGIONS,
    _joint_hsv_histogram,
)
from app.services.feature_extractor import BirdFeatureExtractor  # noqa: E402
from app.services.gabor_features import (  # noqa: E402
    GABOR_KERNELS,
    GABOR_SIZE,
)
from app.services.hog_features import (  # noqa: E402
    HOG_BINS,
    HOG_BLOCK_CELLS,
    HOG_CELL_SIZE,
    HOG_SIZE,
)
from app.services.preprocessing import (  # noqa: E402
    load_image,
    mask_bbox,
    preprocess_array,
    rgba_image,
    write_image,
)
from app.services.shape_features import SILHOUETTE_DIM  # noqa: E402
from app.services.texture_features import (  # noqa: E402
    GRAY_LEVELS,
    LBP_DIM,
    _masked_glcm,
)


GLCM_OFFSETS = ((0, 1), (-1, 1), (-1, 0), (-1, -1))
GLCM_LABELS = ("0 degrees", "45 degrees", "90 degrees", "135 degrees")


def normalized_uint8(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    selected = array[valid] if valid is not None and np.any(valid) else array.ravel()
    if not selected.size:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(selected, (1.0, 99.0))
    if high <= low + 1e-8:
        high = low + 1.0
    output = np.clip((array - low) * 255.0 / (high - low), 0.0, 255.0).astype(np.uint8)
    if valid is not None:
        output[~valid] = 0
    return output


def color_map(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    mapped = cv2.applyColorMap(normalized_uint8(values, valid), cv2.COLORMAP_TURBO)
    if valid is not None:
        mapped[~valid] = 0
    return mapped


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (15, 23, 19), -1)
    cv2.putText(
        output,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 250, 247),
        1,
        cv2.LINE_AA,
    )
    return output


def hue_histogram_image(histogram: np.ndarray, width: int = 720, height: int = 300) -> np.ndarray:
    marginal = histogram.reshape(JOINT_BINS).sum(axis=(1, 2))
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    baseline = height - 38
    plot_top = 40
    maximum = max(float(marginal.max()), 1e-8)
    bar_width = width / len(marginal)
    for index, value in enumerate(marginal):
        x1 = int(round(index * bar_width))
        x2 = int(round((index + 1) * bar_width)) - 2
        bar_height = int(round((baseline - plot_top) * float(value) / maximum))
        hue = int(round((index + 0.5) * 180 / len(marginal))) % 180
        color = cv2.cvtColor(
            np.asarray([[[hue, 230, 235]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
        cv2.rectangle(canvas, (x1 + 2, baseline - bar_height), (max(x1 + 2, x2), baseline), tuple(int(v) for v in color), -1)
        cv2.putText(canvas, str(index), (x1 + 4, baseline + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, baseline), (width - 1, baseline), (50, 50, 50), 1)
    return add_label(canvas, "Global HSV: marginal distribution over 18 hue bins")


def spatial_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = image.copy()
    bbox = mask_bbox(mask)
    if bbox is None:
        return output
    x1, y1, x2, y2 = bbox
    boundaries = np.linspace(x1, x2, SPATIAL_REGIONS + 1, dtype=int)
    colors = ((65, 179, 255), (109, 227, 201), (199, 128, 255))
    overlay = output.copy()
    for index in range(SPATIAL_REGIONS):
        region = np.zeros_like(mask, dtype=bool)
        region[y1:y2, boundaries[index] : boundaries[index + 1]] = mask[
            y1:y2, boundaries[index] : boundaries[index + 1]
        ]
        overlay[region] = colors[index]
    output = cv2.addWeighted(output, 0.58, overlay, 0.42, 0)
    for index, boundary in enumerate(boundaries[1:-1], start=1):
        cv2.line(output, (int(boundary), y1), (int(boundary), y2), (255, 255, 255), 2)
        cv2.putText(output, str(index + 1), (int(boundary) + 5, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, "1", (x1 + 8, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def lbp_visualizations(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maps = []
    for points, radius in ((8, 1), (16, 2)):
        values = local_binary_pattern(gray, points, radius, method="uniform")
        mapped = color_map(values, mask)
        maps.append(add_label(mapped, f"Uniform LBP P={points}, R={radius}"))
    return maps[0], maps[1]


def glcm_contact_sheet(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    quantized = np.minimum(
        gray.astype(np.uint16) * GRAY_LEVELS // 256, GRAY_LEVELS - 1
    ).astype(np.uint8)
    cells: list[np.ndarray] = []
    for (dy, dx), label in zip(GLCM_OFFSETS, GLCM_LABELS):
        matrix = _masked_glcm(quantized, mask, dy, dx)
        display = np.log1p(matrix * 10_000.0)
        heatmap = color_map(display)
        heatmap = cv2.resize(heatmap, (280, 280), interpolation=cv2.INTER_NEAREST)
        cells.append(add_label(heatmap, f"GLCM {label}"))
    return np.vstack((np.hstack(cells[:2]), np.hstack(cells[2:])))


def gabor_contact_sheet(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray[~mask] = float(gray[mask].mean())
    resized = cv2.resize(gray, (GABOR_SIZE, GABOR_SIZE), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask.astype(np.uint8), (GABOR_SIZE, GABOR_SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)
    cells: list[np.ndarray] = []
    for index, (wavelength, kernel) in enumerate(GABOR_KERNELS):
        response = np.abs(cv2.filter2D(resized, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT))
        mapped = color_map(response, resized_mask)
        mapped = cv2.resize(mapped, (176, 176), interpolation=cv2.INTER_AREA)
        orientation = (index % 6) * 30
        cells.append(add_label(mapped, f"lambda={wavelength:g}, theta={orientation} deg"))
    rows = [np.hstack(cells[start : start + 6]) for start in range(0, len(cells), 6)]
    return np.vstack(rows)


def silhouette_heatmap(values: np.ndarray) -> np.ndarray:
    grid = values.reshape(8, 8)
    mapped = cv2.applyColorMap(np.clip(grid * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    mapped = cv2.resize(mapped, (512, 512), interpolation=cv2.INTER_NEAREST)
    for position in range(1, 8):
        coordinate = position * 64
        cv2.line(mapped, (coordinate, 0), (coordinate, 511), (245, 245, 245), 1)
        cv2.line(mapped, (0, coordinate), (511, coordinate), (245, 245, 245), 1)
    return add_label(mapped, "BBox-normalized silhouette grid 8x8")


def hog_visualization(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray[~mask] = 0.0
    resized = cv2.resize(gray, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)
    _, visual = hog(
        resized,
        orientations=HOG_BINS,
        pixels_per_cell=(HOG_CELL_SIZE, HOG_CELL_SIZE),
        cells_per_block=(HOG_BLOCK_CELLS, HOG_BLOCK_CELLS),
        block_norm="L2-Hys",
        transform_sqrt=True,
        visualize=True,
        feature_vector=True,
    )
    mapped = color_map(visual)
    base = cv2.cvtColor(normalized_uint8(resized), cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.38, mapped, 0.62, 0)
    return add_label(cv2.resize(overlay, (512, 512), interpolation=cv2.INTER_CUBIC), "HOG oriented-gradient response")


def html_image(path: Path, title: str, description: str) -> str:
    return (
        "<article class='card'>"
        f"<h3>{html.escape(title)}</h3>"
        f"<img src='{html.escape(path.name)}' alt='{html.escape(title)}'>"
        f"<p>{html.escape(description)}</p>"
        "</article>"
    )


def build_report(source: Path, output_dir: Path, already_preprocessed: bool) -> Path:
    settings = get_settings()
    raw = load_image(source)
    processed = preprocess_array(
        raw,
        already_preprocessed=already_preprocessed,
        settings=settings,
    )
    bundle = BirdFeatureExtractor().extract(processed.image, processed.mask)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = output_dir / "00_source.png"
    normalized_path = output_dir / "01_normalized.png"
    mask_path = output_dir / "02_mask.png"
    hsv_path = output_dir / "03_global_hsv.png"
    spatial_path = output_dir / "04_spatial_hsv_regions.png"
    lbp1_path = output_dir / "05_lbp_r1.png"
    lbp2_path = output_dir / "06_lbp_r2.png"
    glcm_path = output_dir / "07_glcm_matrices.png"
    gabor_path = output_dir / "08_gabor_responses.png"
    silhouette_path = output_dir / "09_silhouette_grid.png"
    hog_path = output_dir / "10_hog_response.png"

    write_image(source_path, processed.original_bgr)
    write_image(normalized_path, rgba_image(processed.image, processed.mask))
    write_image(mask_path, processed.mask.astype(np.uint8) * 255)

    hsv = cv2.cvtColor(processed.image, cv2.COLOR_BGR2HSV)
    global_histogram = _joint_hsv_histogram(hsv, processed.mask, JOINT_BINS)
    write_image(hsv_path, hue_histogram_image(global_histogram))
    write_image(spatial_path, spatial_overlay(processed.image, processed.mask))

    gray = cv2.cvtColor(processed.image, cv2.COLOR_BGR2GRAY)
    lbp_r1, lbp_r2 = lbp_visualizations(gray, processed.mask)
    write_image(lbp1_path, lbp_r1)
    write_image(lbp2_path, lbp_r2)
    write_image(glcm_path, glcm_contact_sheet(gray, processed.mask))
    write_image(gabor_path, gabor_contact_sheet(processed.image, processed.mask))
    write_image(silhouette_path, silhouette_heatmap(bundle.shape[:SILHOUETTE_DIM]))
    write_image(hog_path, hog_visualization(processed.image, processed.mask))

    cards = "".join(
        (
            html_image(source_path, "Ảnh đầu vào", "Ảnh được đọc từ tệp trước bước chuẩn hóa."),
            html_image(normalized_path, "Ảnh sau tiền xử lý", "Chim được crop, giữ tỷ lệ, đưa vào canvas 384×384 và loại nền bằng mask."),
            html_image(mask_path, "Binary mask", "Vùng trắng là foreground được dùng khi trích xuất mọi đặc trưng."),
            html_image(hsv_path, "Global HSV — 288D", "Biểu đồ biên theo Hue của histogram HSV toàn bộ con chim; vector thật còn chứa Saturation và Value."),
            html_image(spatial_path, "Spatial HSV — 216D", "Ba vùng trái–giữa–phải giữ lại vị trí tương đối của màu trên chim."),
            html_image(lbp1_path, "LBP P=8, R=1", "Mã vi kết cấu ở bán kính một pixel; màu chỉ dùng để trực quan hóa mã LBP."),
            html_image(lbp2_path, "LBP P=16, R=2", "Mã vi kết cấu ở bán kính hai pixel; hai histogram tạo thành vector LBP 28D."),
            html_image(glcm_path, "GLCM — 10D", "Bốn ma trận đồng xuất hiện mức xám; vector lưu mean/std của contrast, energy, homogeneity, correlation và entropy."),
            html_image(gabor_path, "Gabor — 36D", "18 đáp ứng từ ba bước sóng và sáu hướng; vector lưu mean/std của từng đáp ứng."),
            html_image(silhouette_path, "BBox Silhouette — 64D", "Mask được cắt theo bounding box, canh giữa trên canvas vuông rồi chia lưới 8×8; mỗi ô lưu tỷ lệ foreground."),
            html_image(hog_path, "HOG — 1764D", "Các hướng gradient cục bộ mô tả đường biên, mỏ, cánh, thân và đuôi."),
        )
    )
    document = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Handcrafted feature visualization</title>
  <style>
    :root {{ color-scheme: light; --ink:#17231d; --muted:#627068; --line:#d8e0d9; --panel:#f7f9f5; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Segoe UI,Arial,sans-serif; color:var(--ink); background:#eef2ea; }}
    main {{ max-width:1440px; margin:auto; padding:28px; }}
    h1 {{ margin:0 0 8px; }} .lead {{ color:var(--muted); margin:0 0 24px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:18px; }}
    .card {{ background:white; border:1px solid var(--line); border-radius:12px; padding:14px; box-shadow:0 3px 12px #2035250d; }}
    .card h3 {{ margin:0 0 10px; }} .card img {{ width:100%; height:330px; object-fit:contain; background:var(--panel); border-radius:8px; }}
    .card p {{ color:var(--muted); line-height:1.45; margin:10px 0 0; }}
    table {{ width:100%; border-collapse:collapse; }} td,th {{ padding:9px; border-bottom:1px solid var(--line); text-align:left; }}
    code {{ background:#e8eee7; padding:2px 5px; border-radius:4px; }}
  </style>
</head>
<body><main>
  <h1>Trực quan pipeline đặc trưng thủ công</h1>
  <p class="lead">Nguồn: <code>{html.escape(str(source))}</code>. Các ảnh màu giả chỉ minh họa đáp ứng; vector tìm kiếm vẫn dùng giá trị số thực từ pipeline.</p>
  <section class="grid">{cards}</section>
  <article class="card" style="margin-top:18px">
    <h3>Kích thước vector thực tế</h3>
    <table><tbody>
      <tr><td>Global HSV</td><td>{JOINT_DIM}</td></tr>
      <tr><td>Spatial HSV</td><td>{SPATIAL_DIM}</td></tr>
      <tr><td>LBP</td><td>{LBP_DIM}</td></tr>
      <tr><td>GLCM</td><td>10</td></tr>
      <tr><td>Gabor</td><td>36</td></tr>
      <tr><td>BBox Silhouette</td><td>64</td></tr>
      <tr><td>HOG</td><td>1764</td></tr>
      <tr><th>Tổng</th><th>{bundle.color.size + bundle.texture.size + bundle.shape.size}</th></tr>
    </tbody></table>
  </article>
</main></body></html>"""
    report_path = output_dir / "index.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input bird image")
    parser.add_argument(
        "--already-preprocessed",
        action="store_true",
        help="Use an existing alpha/background mask instead of running YOLO segmentation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination (default: results/feature_visualization/<image-stem>)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.image.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else PROJECT_ROOT / "results" / "feature_visualization" / source.stem
    )
    report_path = build_report(source, output_dir, args.already_preprocessed)
    print(f"Saved feature visualization: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
