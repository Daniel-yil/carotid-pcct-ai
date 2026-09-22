from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import subprocess
import json
import csv
import shutil
import io
import threading

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw

import carotid_analysis_pipeline_merged as pipeline


APP_DIR = Path(__file__).parent

INPUT_DIR = APP_DIR / "input"
PREDICTION_DIR = APP_DIR / "prediction"
OUTPUT_DIR = APP_DIR / "output"

INDEX_FILE = APP_DIR / "index.html"

INPUT_DIR.mkdir(exist_ok=True)
PREDICTION_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Basic paths
# ============================================================

def get_input_image_path():
    return INPUT_DIR / "carotid_case_0000.nii.gz"


def get_segmentation_path():
    return PREDICTION_DIR / "carotid_case.nii.gz"


def get_calcification_path():
    return OUTPUT_DIR / "calcification_map.nii.gz"


def get_fai_fat_mask_path():
    return OUTPUT_DIR / "FAI_fat_mask.nii.gz"


# ============================================================
# Small cache so slider requests are fast
# ============================================================

_ARRAY_CACHE = {}
_CACHE_LOCK = threading.Lock()


def clear_array_cache():
    with _CACHE_LOCK:
        _ARRAY_CACHE.clear()


def load_nifti_array(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")

    mtime = path.stat().st_mtime_ns
    key = str(path.resolve())

    with _CACHE_LOCK:
        cached = _ARRAY_CACHE.get(key)

        if cached is not None and cached["mtime"] == mtime:
            return cached["array"], cached["image"]

    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)

    if array.ndim != 3:
        raise ValueError(
            f"Expected a 3D image at {path.name}, but got shape {array.shape}."
        )

    with _CACHE_LOCK:
        _ARRAY_CACHE[key] = {
            "mtime": mtime,
            "array": array,
            "image": image,
        }

    return array, image


def load_ct_array():
    array, _ = load_nifti_array(get_input_image_path())
    return array


# ============================================================
# CSV / folder helpers
# ============================================================

def read_csv_as_dict(filename):
    path = OUTPUT_DIR / filename

    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clear_folder(folder):
    if not folder.exists():
        return

    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


# ============================================================
# CT display + overlay utilities
# ============================================================

def normalize_ct_slice(slice_array, window_center=40, window_width=400):
    """Convert one CT slice to 8-bit grayscale."""
    lower = window_center - window_width / 2.0
    upper = window_center + window_width / 2.0

    clipped = np.clip(slice_array, lower, upper)
    normalized = (clipped - lower) / (upper - lower) * 255.0

    return normalized.astype(np.uint8)


def clamp_slice_index(index, total_slices):
    if total_slices < 1:
        raise ValueError("The CT contains no slices.")

    return max(0, min(int(index), total_slices - 1))


def validate_same_shape(ct_array, other_array, name):
    if ct_array.shape != other_array.shape:
        raise ValueError(
            f"{name} shape {other_array.shape} does not match CT shape {ct_array.shape}."
        )


def base_ct_rgb(slice_index):
    ct_array, _ = load_nifti_array(get_input_image_path())
    slice_index = clamp_slice_index(slice_index, ct_array.shape[0])

    gray = normalize_ct_slice(ct_array[slice_index])
    rgb = np.stack([gray, gray, gray], axis=-1)

    return rgb.astype(np.uint8), slice_index, ct_array.shape[0]


def alpha_overlay(base_rgb, mask_slice, color_map, alpha=0.48):
    """
    Filled color overlay.
    mask_slice contains 0 for background and 1/2/3/4 for vessel labels.
    """
    result = base_rgb.astype(np.float32).copy()

    for label_value, color in color_map.items():
        region = mask_slice == label_value

        if not np.any(region):
            continue

        color_arr = np.array(color, dtype=np.float32)
        result[region] = (
            (1.0 - alpha) * result[region]
            + alpha * color_arr
        )

    return np.clip(result, 0, 255).astype(np.uint8)


def png_bytes_from_rgb(rgb_array):
    image = Image.fromarray(rgb_array, mode="RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


# Vessel colors:
# 1 = RECA -> red
# 2 = RICA -> green
# 3 = LECA -> blue
# 4 = LICA -> yellow
VESSEL_COLORS = {
    1: (255, 70, 70),
    2: (50, 205, 80),
    3: (70, 125, 255),
    4: (255, 220, 0),
}


def get_segmentation_slice_png(slice_index):
    base_rgb, slice_index, _ = base_ct_rgb(slice_index)

    seg, _ = load_nifti_array(get_segmentation_path())
    ct, _ = load_nifti_array(get_input_image_path())
    validate_same_shape(ct, seg, "Segmentation")

    rgb = alpha_overlay(
        base_rgb,
        seg[slice_index],
        VESSEL_COLORS,
        alpha=0.52,
    )

    return png_bytes_from_rgb(rgb)


def get_calcification_slice_png(slice_index):
    base_rgb, slice_index, _ = base_ct_rgb(slice_index)

    calc, _ = load_nifti_array(get_calcification_path())
    ct, _ = load_nifti_array(get_input_image_path())
    validate_same_shape(ct, calc, "Calcification map")

    # Keep calcification visually distinct from vessel segmentation.
    calc_colors = {
        1: (255, 0, 220),
        2: (255, 0, 220),
        3: (255, 0, 220),
        4: (255, 0, 220),
    }

    rgb = alpha_overlay(
        base_rgb,
        calc[slice_index],
        calc_colors,
        alpha=0.75,
    )

    return png_bytes_from_rgb(rgb)


def get_fai_slice_png(slice_index):
    base_rgb, slice_index, _ = base_ct_rgb(slice_index)

    fat_mask, _ = load_nifti_array(get_fai_fat_mask_path())
    ct, _ = load_nifti_array(get_input_image_path())
    validate_same_shape(ct, fat_mask, "FAI fat mask")

    rgb = alpha_overlay(
        base_rgb,
        fat_mask[slice_index],
        VESSEL_COLORS,
        alpha=0.68,
    )

    return png_bytes_from_rgb(rgb)


# ============================================================
# Minimum-diameter location image
# ============================================================

def get_minimum_diameter_markers():
    """Return the minimum-diameter point for each vessel from radius_profile.csv."""
    radius_file = OUTPUT_DIR / "radius_profile.csv"

    if not radius_file.exists():
        raise FileNotFoundError(f"Missing radius profile: {radius_file}")

    markers = {}

    with open(radius_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            vessel = row["vessel"]
            diameter = float(row["diameter_mm"])

            if vessel not in markers or diameter < markers[vessel]["diameter_mm"]:
                markers[vessel] = {
                    "diameter_mm": diameter,
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                }

    return markers


def get_minimum_diameter_slice_png(vessel="RECA"):
    vessel = vessel.upper()
    label_map = {"RECA": 1, "RICA": 2, "LECA": 3, "LICA": 4}

    if vessel not in label_map:
        raise ValueError(f"Unknown vessel: {vessel}")

    markers = get_minimum_diameter_markers()
    if vessel not in markers:
        raise ValueError(f"No minimum-diameter point found for {vessel}.")

    ct, _ = load_nifti_array(get_input_image_path())
    seg, seg_img = load_nifti_array(get_segmentation_path())
    validate_same_shape(ct, seg, "Segmentation")

    spacing = seg_img.GetSpacing()
    marker = markers[vessel]

    # Keep coordinate conversion consistent with the current pipeline,
    # which stores x/y/z as voxel index * spacing.
    x = int(round(marker["x"] / spacing[0]))
    y = int(round(marker["y"] / spacing[1]))
    z = int(round(marker["z"] / spacing[2]))

    z = clamp_slice_index(z, ct.shape[0])
    y = max(0, min(y, ct.shape[1] - 1))
    x = max(0, min(x, ct.shape[2] - 1))

    gray = normalize_ct_slice(ct[z])
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)

    # Overlay only the selected vessel on the minimum-diameter slice.
    selected_mask = (seg[z] == label_map[vessel]).astype(np.uint8) * label_map[vessel]
    rgb = alpha_overlay(rgb, selected_mask, VESSEL_COLORS, alpha=0.52)

    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)

    # Yellow ring + cross marks the centroid of the minimum-diameter slice.
    r = 11
    draw.ellipse((x-r, y-r, x+r, y+r), outline=(255, 230, 0), width=3)
    draw.line((x-16, y, x+16, y), fill=(255, 230, 0), width=2)
    draw.line((x, y-16, x, y+16), fill=(255, 230, 0), width=2)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()



# ============================================================
# Calcification summary enrichment
# ============================================================

CALCIFICATION_HU_THRESHOLD = 130.0
CALCIFICATION_MIN_LESION_AREA_MM2 = 1.0

def agatston_density_factor(max_hu):
    """Classic Agatston density weighting based on lesion maximum HU."""
    max_hu = float(max_hu)
    if max_hu < 130:
        return 0
    if max_hu < 200:
        return 1
    if max_hu < 300:
        return 2
    if max_hu < 400:
        return 3
    return 4


def calculate_agatston_from_mask(ct_array, calc_mask, spacing):
    """
    Agatston-style score from the generated calcification mask.
    Scoring is performed slice-by-slice using 2D connected components.
    """
    pixel_area_mm2 = float(spacing[0]) * float(spacing[1])
    total_score = 0.0

    for z in range(calc_mask.shape[0]):
        binary_slice = (calc_mask[z] > 0).astype(np.uint8)
        if not np.any(binary_slice):
            continue

        slice_img = sitk.GetImageFromArray(binary_slice)
        cc_img = sitk.ConnectedComponent(slice_img)
        cc = sitk.GetArrayFromImage(cc_img)

        for component_id in np.unique(cc):
            if component_id == 0:
                continue

            lesion = cc == component_id
            area_mm2 = float(np.sum(lesion)) * pixel_area_mm2
            if area_mm2 < CALCIFICATION_MIN_LESION_AREA_MM2:
                continue

            max_hu = float(np.max(ct_array[z][lesion]))
            total_score += area_mm2 * agatston_density_factor(max_hu)

    return total_score


def enrich_calcification_results(rows):
    """
    Add Agatston score and calcification-volume percentage.

    Calcification-volume percentage =
        calcification volume / segmented vessel volume * 100
    """
    if not rows:
        return rows

    ct, ct_img = load_nifti_array(get_input_image_path())
    seg, _ = load_nifti_array(get_segmentation_path())
    calc, _ = load_nifti_array(get_calcification_path())

    validate_same_shape(ct, seg, "Segmentation")
    validate_same_shape(ct, calc, "Calcification map")

    spacing = ct_img.GetSpacing()
    voxel_volume_mm3 = (
        float(spacing[0]) *
        float(spacing[1]) *
        float(spacing[2])
    )

    label_map = {"RECA": 1, "RICA": 2, "LECA": 3, "LICA": 4}
    enriched = []

    for row in rows:
        row = dict(row)
        vessel = str(row.get("vessel", "")).upper()
        label_value = label_map.get(vessel)

        if label_value is None:
            enriched.append(row)
            continue

        vessel_mask = seg == label_value
        calc_mask = calc == label_value

        vessel_volume_mm3 = float(np.sum(vessel_mask)) * voxel_volume_mm3
        calcification_volume_mm3 = float(np.sum(calc_mask)) * voxel_volume_mm3

        # Keep displayed volume and percentage based on exactly the same mask.
        row["calcification_volume_mm3"] = f"{calcification_volume_mm3:.3f}"
        row["vessel_volume_mm3"] = f"{vessel_volume_mm3:.3f}"

        if vessel_volume_mm3 > 0:
            percentage = calcification_volume_mm3 / vessel_volume_mm3 * 100.0
            row["calcification_volume_percent"] = f"{percentage:.3f}"
        else:
            row["calcification_volume_percent"] = "-"

        # Reuse an existing Agatston value if the pipeline already outputs one.
        existing_score = (
            row.get("agatston_score")
            or row.get("Agatston_score")
            or row.get("agatston")
        )

        if existing_score not in (None, ""):
            row["agatston_score"] = existing_score
        else:
            score = calculate_agatston_from_mask(ct, calc_mask, spacing)
            row["agatston_score"] = f"{score:.3f}"

        enriched.append(row)

    return enriched


# ============================================================
# HTTP helpers
# ============================================================

def send_json(handler, payload, status=200):
    data = json.dumps(payload).encode("utf-8")

    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()

    try:
        handler.wfile.write(data)
    except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
        pass


def send_png(handler, png_data):
    handler.send_response(200)
    handler.send_header("Content-Type", "image/png")
    handler.send_header("Content-Length", str(len(png_data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()

    try:
        handler.wfile.write(png_data)
    except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
        # Normal when the browser cancels an old slider image request.
        pass


# ============================================================
# HTTP handler
# ============================================================

class CarotidHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed_url = urlparse(self.path)

        if parsed_url.path == "/":
            try:
                html = INDEX_FILE.read_bytes()

                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(html)

            except Exception as e:
                self.send_error(500, str(e))

            return

        if parsed_url.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if parsed_url.path == "/image-info":
            try:
                ct_array = load_ct_array()

                send_json(
                    self,
                    {
                        "success": True,
                        "num_slices": int(ct_array.shape[0]),
                        "height": int(ct_array.shape[1]),
                        "width": int(ct_array.shape[2]),
                    },
                )

            except Exception as e:
                send_json(
                    self,
                    {
                        "success": False,
                        "error": str(e),
                    },
                    status=500,
                )

            return

        if parsed_url.path == "/minimum-diameter-image":
            try:
                query = parse_qs(parsed_url.query)
                vessel = query.get("vessel", ["RECA"])[0]
                png_data = get_minimum_diameter_slice_png(vessel)
                send_png(self, png_data)
            except Exception as e:
                try:
                    self.send_error(500, str(e))
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    pass
            return

        slice_routes = {
            "/segmentation-slice": get_segmentation_slice_png,
            "/calcification-slice": get_calcification_slice_png,
            "/fai-slice": get_fai_slice_png,
        }

        if parsed_url.path in slice_routes:
            try:
                query = parse_qs(parsed_url.query)
                slice_index = int(query.get("index", ["0"])[0])

                png_data = slice_routes[parsed_url.path](slice_index)
                send_png(self, png_data)

            except Exception as e:
                # Do not crash the server if an image is not ready yet.
                try:
                    self.send_error(500, str(e))
                except (
                    BrokenPipeError,
                    ConnectionAbortedError,
                    ConnectionResetError,
                ):
                    pass

            return

        self.send_error(404)


    def do_POST(self):
        parsed_url = urlparse(self.path)

        if parsed_url.path != "/analyze":
            self.send_error(404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))

            if content_length == 0:
                raise ValueError("No image was uploaded.")

            uploaded_data = self.rfile.read(content_length)

            # ------------------------------------------------
            # Clear previous patient data/results
            # ------------------------------------------------
            clear_folder(INPUT_DIR)
            clear_folder(PREDICTION_DIR)
            clear_folder(OUTPUT_DIR)
            clear_array_cache()

            # ------------------------------------------------
            # Save uploaded NIfTI
            # ------------------------------------------------
            input_file = get_input_image_path()

            with open(input_file, "wb") as f:
                f.write(uploaded_data)

            print("\nUploaded image:", input_file)

            # ------------------------------------------------
            # Step 1: nnU-Net segmentation
            # ------------------------------------------------
            command = [
                "nnUNetv2_predict",
                "-i", str(INPUT_DIR),
                "-o", str(PREDICTION_DIR),
                "-d", "501",
                "-c", "3d_fullres",
                "-f", "0",
                "-chk", "checkpoint_best.pth",
            ]

            print("\nRunning segmentation...")
            subprocess.run(command, check=True)

            mask_file = get_segmentation_path()

            if not mask_file.exists():
                raise FileNotFoundError(
                    "Segmentation mask was not generated."
                )

            print("Segmentation completed.")

            # ------------------------------------------------
            # Step 2: Post-processing
            # ------------------------------------------------
            print("\nRunning post-processing...")

            pipeline.generate_centerline()
            pipeline.calculate_length()
            pipeline.calculate_radius()
            pipeline.calculate_stenosis()
            pipeline.calculate_calcification()
            pipeline.calculate_fai()

            print("Post-processing completed.")

            clear_array_cache()

            # ------------------------------------------------
            # Step 3: Read CSV results
            # ------------------------------------------------
            stenosis = read_csv_as_dict("stenosis_result.csv")
            length = read_csv_as_dict("length_result.csv")
            calcification = enrich_calcification_results(
                read_csv_as_dict("calcification_result.csv")
            )
            fai = read_csv_as_dict("fai_result.csv")

            ct_array = load_ct_array()

            result = {
                "success": True,
                "stenosis": stenosis,
                "length": length,
                "calcification": calcification,
                "fai": fai,
                "image": {
                    "num_slices": int(ct_array.shape[0]),
                    "height": int(ct_array.shape[1]),
                    "width": int(ct_array.shape[2]),
                },
            }

            send_json(self, result, status=200)

        except Exception as e:
            print("\nERROR:", e)

            send_json(
                self,
                {
                    "success": False,
                    "error": str(e),
                },
                status=500,
            )


if __name__ == "__main__":
    server = ThreadingHTTPServer(
        ("127.0.0.1", 8081),
        CarotidHandler,
    )

    print("====================================")
    print("Carotid AI Web App")
    print("Open:")
    print("http://127.0.0.1:8081")
    print("====================================")

    server.serve_forever()
