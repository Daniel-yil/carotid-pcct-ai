# ==========================================================
# Carotid Analysis Pipeline
#
# Part 1:
# Configuration + Centerline Extraction
#
# Input:
# CTA:
# data/CASE_NAME_0000.nii.gz
#
# Mask:
# data/CASE_NAME.nii.gz
#
# Output:
# centerline_points.csv
#
# ==========================================================


import os
import csv
import math

import numpy as np
import SimpleITK as sitk



# ==========================================================
# 1. Configuration
# ==========================================================


# =========================
# This is where you could change your data
# =========================

from pathlib import Path
 
APP_DIR = Path(__file__).parent
 
INPUT_DIR = APP_DIR / "input"
PREDICTION_DIR = APP_DIR / "prediction"
OUTPUT_DIR = APP_DIR / "output"
 
OUTPUT_DIR.mkdir(exist_ok=True)
 
CTA_PATH = str(
    INPUT_DIR / "carotid_case_0000.nii.gz"
)
 
LABEL_PATH = str(
    PREDICTION_DIR / "carotid_case.nii.gz"
)
 
CENTERLINE_OUTPUT = str(
    OUTPUT_DIR / "centerline_points.csv"
)


# ==========================================================
# 2. Check files
# ==========================================================


def check_files():

    print("==============================")
    print("Checking input files")
    print("==============================")


    if not os.path.exists(CTA_PATH):

        raise FileNotFoundError(
            "Missing CTA: " + CTA_PATH
        )


    if not os.path.exists(LABEL_PATH):

        raise FileNotFoundError(
            "Missing mask: " + LABEL_PATH
        )


    print("CTA:")
    print(CTA_PATH)


    print("Mask:")
    print(LABEL_PATH)



# ==========================================================
# 3. Load mask
# ==========================================================


def load_label():

    print("\nLoading mask...")


    label_img = sitk.ReadImage(
        LABEL_PATH
    )


    label = sitk.GetArrayFromImage(
        label_img
    )


    spacing = label_img.GetSpacing()


    print(
        "Mask shape:",
        label.shape
    )


    print(
        "Labels:",
        np.unique(label)
    )


    return label, spacing



# ==========================================================
# 4. Generate centerline
#
# Method:
# slice-based centroid extraction
#
# Label:
# 1 RECA
# 2 RICA
# 3 LECA
# 4 LICA
#
# ==========================================================


def generate_centerline():

    print("\n==============================")
    print("Generating centerline")
    print("==============================")


    label_img = sitk.ReadImage(
        LABEL_PATH
    )


    label = sitk.GetArrayFromImage(
        label_img
    )


    spacing = label_img.GetSpacing()



    vessels = {
        "RECA": 1,
        "RICA": 2,
        "LECA": 3,
        "LICA": 4
    }



    points = []



    for vessel_name, label_value in vessels.items():


        print(
            "Processing:",
            vessel_name
        )


        mask = (
            label == label_value
        )


        coords = np.where(
            mask
        )


        if len(coords[0]) == 0:

            print(
                vessel_name,
                "not found"
            )

            continue




        unique_z = np.unique(
            coords[0]
        )



        for z in unique_z:


            slice_points = np.where(
                mask[z]
            )


            if len(slice_points[0]) == 0:

                continue



            y_center = np.mean(
                slice_points[0]
            )


            x_center = np.mean(
                slice_points[1]
            )



            # voxel index -> physical coordinate

            physical_x = (
                x_center
                *
                spacing[0]
            )


            physical_y = (
                y_center
                *
                spacing[1]
            )


            physical_z = (
                z
                *
                spacing[2]
            )



            points.append(
                [
                    vessel_name,
                    physical_x,
                    physical_y,
                    physical_z
                ]
            )





    with open(
        CENTERLINE_OUTPUT,
        "w",
        newline=""
    ) as f:


        writer = csv.writer(f)


        writer.writerow(
            [
                "vessel",
                "x",
                "y",
                "z"
            ]
        )


        writer.writerows(
            points
        )



    print(
        "\nSaved:"
    )

    print(
        CENTERLINE_OUTPUT
    )


    print(
        "Number of points:",
        len(points)
    )


    return CENTERLINE_OUTPUT


# ==========================================================
# Part 2:
# Length Measurement + Radius Profile
# ==========================================================



LENGTH_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "length_result.csv"
)


RADIUS_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "radius_profile.csv"
)




# ==========================================================
# 5. Calculate vessel length
#
# Using consecutive centerline points
#
# ==========================================================


def calculate_length():


    print("\n==============================")
    print("Calculating vessel length")
    print("==============================")


    if not os.path.exists(
        CENTERLINE_OUTPUT
    ):

        raise FileNotFoundError(
            "Centerline file missing"
        )



    vessels = {}


    with open(
        CENTERLINE_OUTPUT,
        "r"
    ) as f:


        reader = csv.DictReader(f)


        for row in reader:


            vessel = row["vessel"]


            point = np.array(
                [
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"])
                ]
            )


            if vessel not in vessels:

                vessels[vessel] = []


            vessels[vessel].append(
                point
            )



    results = []



    for vessel, points in vessels.items():


        length = 0



        for i in range(
            len(points)-1
        ):


            distance = np.linalg.norm(
                points[i+1]
                -
                points[i]
            )


            length += distance



        results.append(
            [
                vessel,
                round(length,2)
            ]
        )


        print(
            vessel,
            "length:",
            round(length,2),
            "mm"
        )



    with open(
        LENGTH_OUTPUT,
        "w",
        newline=""
    ) as f:


        writer = csv.writer(f)


        writer.writerow(
            [
                "vessel",
                "length_mm"
            ]
        )


        writer.writerows(
            results
        )



    print(
        "Saved:",
        LENGTH_OUTPUT
    )


    return LENGTH_OUTPUT




# ==========================================================
# 6. Calculate radius profile
#
# Equivalent radius:
#
# r = sqrt(area / pi)
#
# ==========================================================


def calculate_radius():


    print("\n==============================")
    print("Calculating radius profile")
    print("==============================")


    label_img = sitk.ReadImage(
        LABEL_PATH
    )


    label = sitk.GetArrayFromImage(
        label_img
    )


    spacing = label_img.GetSpacing()



    # read centerline

    center_points = []



    with open(
        CENTERLINE_OUTPUT,
        "r"
    ) as f:


        reader = csv.DictReader(f)


        for row in reader:


            center_points.append(
                {
                    "vessel": row["vessel"],

                    "x": float(row["x"]),

                    "y": float(row["y"]),

                    "z": float(row["z"])
                }
            )



    label_map = {
        "RECA": 1,
        "RICA": 2,
        "LECA": 3,
        "LICA": 4
    }



    results = []



    for p in center_points:


        vessel = p["vessel"]


        label_value = label_map[vessel]



        # physical z -> voxel index

        z_index = int(
            p["z"]
            /
            spacing[2]
        )



        if z_index >= label.shape[0]:

            continue



        slice_mask = (

            label[z_index]
            ==
            label_value

        )



        area = (

            np.sum(slice_mask)

            *

            spacing[0]

            *

            spacing[1]

        )



        if area == 0:

            continue



        radius = math.sqrt(
            area
            /
            math.pi
        )


        diameter = (
            2
            *
            radius
        )



        results.append(
            [

                vessel,

                p["x"],

                p["y"],

                p["z"],

                radius,

                diameter

            ]
        )



    with open(
        RADIUS_OUTPUT,
        "w",
        newline=""
    ) as f:


        writer = csv.writer(f)


        writer.writerow(

            [

                "vessel",

                "x",

                "y",

                "z",

                "radius_mm",

                "diameter_mm"

            ]

        )


        writer.writerows(
            results
        )



    print(
        "Saved:",
        RADIUS_OUTPUT
    )


    print(
        "Measurements:",
        len(results)
    )


    return RADIUS_OUTPUT


# ==========================================================
# Part 3:
# Stenosis Evaluation + Calcified Plaque Analysis
# ==========================================================



STENOSIS_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stenosis_result.csv"
)


CALCIFICATION_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "calcification_result.csv"
)




# ==========================================================
# 7. Calculate stenosis
#
# Formula:
#
# stenosis(%) =
# (1 - minimum diameter / normal diameter) * 100
#
# ==========================================================


def calculate_stenosis():


    print("\n==============================")
    print("Calculating stenosis")
    print("==============================")


    if not os.path.exists(
        RADIUS_OUTPUT
    ):

        raise FileNotFoundError(
            "Radius profile missing"
        )



    vessel_diameters = {}



    with open(
        RADIUS_OUTPUT,
        "r"
    ) as f:


        reader = csv.DictReader(f)


        for row in reader:


            vessel = row["vessel"]


            diameter = float(
                row["diameter_mm"]
            )


            if vessel not in vessel_diameters:

                vessel_diameters[vessel] = []


            vessel_diameters[vessel].append(
                diameter
            )



    results = []



    for vessel, diameters in vessel_diameters.items():


        normal_diameter = max(
            diameters
        )


        minimum_diameter = min(
            diameters
        )


        average_diameter = float(
            np.mean(diameters)
        )


        stenosis = (

            1
            -
            minimum_diameter
            /
            normal_diameter

        ) * 100



        results.append(
            [

                vessel,

                round(normal_diameter,2),

                round(average_diameter,2),

                round(minimum_diameter,2),

                round(stenosis,2)

            ]
        )



        print("----------------")
        print(vessel)

        print(
            "Normal diameter:",
            round(normal_diameter,2),
            "mm"
        )

        print(
            "Average diameter:",
            round(average_diameter,2),
            "mm"
        )

        print(
            "Minimum diameter:",
            round(minimum_diameter,2),
            "mm"
        )

        print(
            "Stenosis:",
            round(stenosis,2),
            "%"
        )



    with open(
        STENOSIS_OUTPUT,
        "w",
        newline=""
    ) as f:


        writer = csv.writer(f)


        writer.writerow(

            [

                "vessel",

                "normal_diameter_mm",

                "average_diameter_mm",

                "minimum_diameter_mm",

                "stenosis_percent"

            ]

        )


        writer.writerows(
            results
        )



    print(
        "Saved:",
        STENOSIS_OUTPUT
    )


    return STENOSIS_OUTPUT





# ==========================================================
# 8. Calcification Analysis
#
# New implementation:
# - Calcification ROI is INSIDE the segmented vessel
# - Calcium threshold: HU >= 130
# - 2D axial connected components
# - Minimum lesion area: 1 mm2
# - Agatston-style area x density weighting
# - Saves CSV + ROI map + calcification map
# ==========================================================

CALC_THRESHOLD = 130
MIN_LESION_AREA_MM2 = 1.0
PERIVASCULAR_DISTANCE_MM = 3.0  # still used by FAI

CALC_MAP_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "calcification_map.nii.gz"
)

CALC_ROI_MAP_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "calcification_ROI.nii.gz"
)


def create_perivascular_roi(
    vessel_mask,
    reference_img,
    distance_mm
):
 
    vessel_img = sitk.GetImageFromArray(
        vessel_mask.astype(np.uint8)
    )
 
    vessel_img.CopyInformation(
        reference_img
    )
 
    distance_img = sitk.SignedMaurerDistanceMap(
        vessel_img,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True
    )
 
    distance = sitk.GetArrayFromImage(
        distance_img
    )
 
    roi = (
        (distance > 0)
&
        (distance <= distance_mm)
    )
 
    return roi
 
 
# ==========================================================
# 4. Calcification analysis
# ==========================================================
 
def agatston_density_factor(max_hu):
    """Return the classic Agatston density weighting factor."""

    if max_hu < 130:
        return 0
    elif max_hu < 200:
        return 1
    elif max_hu < 300:
        return 2
    elif max_hu < 400:
        return 3
    else:
        return 4


def calculate_calcification():

    print("\n===================================")
    print(" Calcification + Agatston Analysis Start ")
    print("===================================")

    if not os.path.exists(CTA_PATH):
        raise FileNotFoundError(
            "Missing CTA: " + CTA_PATH
        )

    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(
            "Missing label: " + LABEL_PATH
        )

    # ------------------------------------------------------
    # Load image and segmentation
    # ------------------------------------------------------

    cta_img = sitk.ReadImage(CTA_PATH)
    cta = sitk.GetArrayFromImage(cta_img)

    label_img = sitk.ReadImage(LABEL_PATH)
    label = sitk.GetArrayFromImage(label_img)

    print("\nCTA shape:", cta.shape)
    print("Label shape:", label.shape)
    print("Labels:", np.unique(label))
    print("CTA HU min:", np.min(cta))
    print("CTA HU max:", np.max(cta))

    if cta.shape != label.shape:
        raise ValueError(
            "CTA and label shapes do not match: "
            f"{cta.shape} vs {label.shape}"
        )

    spacing = cta_img.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    pixel_area = spacing[0] * spacing[1]

    print("CTA spacing:", spacing)
    print("Axial pixel area:", pixel_area, "mm2")

    # ------------------------------------------------------
    # Vessel labels
    # ------------------------------------------------------

    vessels = {
        "RECA": 1,
        "RICA": 2,
        "LECA": 3,
        "LICA": 4
    }

    # ------------------------------------------------------
    # Output maps
    # calcification_ROI = INSIDE the vessel segmentation
    # calcification_map = qualifying calcium components only
    # ------------------------------------------------------

    calc_map = np.zeros_like(label, dtype=np.uint8)
    roi_map = np.zeros_like(label, dtype=np.uint8)

    results = []

    for vessel, label_value in vessels.items():

        print("\n-------------------------------")
        print("Processing:", vessel)
        print("-------------------------------")

        vessel_mask = (label == label_value)
        vessel_voxels = int(np.sum(vessel_mask))

        print("Vessel voxels:", vessel_voxels)

        if vessel_voxels == 0:
            print(vessel, "not found.")
            results.append([
                vessel,
                0.0,
                0,
                0.0,
                0,
                "",
                "",
                ""
            ])
            continue

        # ROI is strictly INSIDE the vessel segmentation.
        roi = vessel_mask
        roi_map[roi] = label_value

        roi_values = cta[roi]
        roi_min = float(np.min(roi_values))
        roi_max = float(np.max(roi_values))
        roi_mean = float(np.mean(roi_values))

        print("ROI location: inside vessel mask")
        print("ROI voxels:", int(np.sum(roi)))
        print("ROI HU min:", roi_min)
        print("ROI HU max:", roi_max)
        print("ROI HU mean:", roi_mean)

        total_agatston = 0.0
        lesion_count = 0
        vessel_calc_mask = np.zeros_like(vessel_mask, dtype=bool)

        # --------------------------------------------------
        # Agatston-style calculation on each axial slice
        # 1) HU >= 130 inside the vessel
        # 2) 2D connected components
        # 3) keep components with area >= 1 mm2
        # 4) area * density factor (1-4)
        # --------------------------------------------------

        for z in range(cta.shape[0]):

            candidate_slice = (
                vessel_mask[z]
                &
                (cta[z] >= CALC_THRESHOLD)
            )

            if not np.any(candidate_slice):
                continue

            cc_img = sitk.ConnectedComponent(
                sitk.GetImageFromArray(
                    candidate_slice.astype(np.uint8)
                )
            )
            cc = sitk.GetArrayFromImage(cc_img)

            component_ids = np.unique(cc)
            component_ids = component_ids[component_ids != 0]

            for component_id in component_ids:

                component = (cc == component_id)
                component_pixels = int(np.sum(component))
                component_area = component_pixels * pixel_area

                if component_area < MIN_LESION_AREA_MM2:
                    continue

                component_hu = cta[z][component]
                component_max_hu = float(np.max(component_hu))
                density_factor = agatston_density_factor(
                    component_max_hu
                )

                lesion_score = component_area * density_factor
                total_agatston += lesion_score
                lesion_count += 1

                vessel_calc_mask[z][component] = True

        calcified_voxels = int(np.sum(vessel_calc_mask))
        calcification_volume = calcified_voxels * voxel_volume

        if calcified_voxels > 0:
            calcium_values = cta[vessel_calc_mask]
            mean_calcification_hu = float(np.mean(calcium_values))
            max_calcification_hu = float(np.max(calcium_values))
        else:
            mean_calcification_hu = None
            max_calcification_hu = None

        calc_map[vessel_calc_mask] = label_value

        print("Agatston score:", round(total_agatston, 2))
        print("Detected lesions:", lesion_count)
        print("Calcified voxels:", calcified_voxels)
        print(
            "Calcification volume:",
            round(calcification_volume, 3),
            "mm3"
        )
        print("Mean calcification HU:", mean_calcification_hu)
        print("Max calcification HU:", max_calcification_hu)

        results.append([
            vessel,
            round(total_agatston, 2),
            calcified_voxels,
            round(calcification_volume, 3),
            lesion_count,
            round(roi_max, 2),
            (
                round(mean_calcification_hu, 2)
                if mean_calcification_hu is not None
                else ""
            ),
            (
                round(max_calcification_hu, 2)
                if max_calcification_hu is not None
                else ""
            )
        ])

    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------

    with open(
        CALCIFICATION_OUTPUT,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)
        writer.writerow([
            "vessel",
            "agatston_score",
            "calcified_voxel",
            "calcification_volume_mm3",
            "lesion_count",
            "ROI_max_HU",
            "mean_calcification_HU",
            "max_calcification_HU"
        ])
        writer.writerows(results)

    print("\nSaved:", CALCIFICATION_OUTPUT)

    # ------------------------------------------------------
    # Save vessel-inside ROI map
    # ------------------------------------------------------

    roi_img = sitk.GetImageFromArray(roi_map)
    roi_img.CopyInformation(cta_img)
    sitk.WriteImage(roi_img, CALC_ROI_MAP_OUTPUT)
    print("Saved:", CALC_ROI_MAP_OUTPUT)

    # ------------------------------------------------------
    # Save qualifying calcification map
    # ------------------------------------------------------

    calc_img = sitk.GetImageFromArray(calc_map)
    calc_img.CopyInformation(cta_img)
    sitk.WriteImage(calc_img, CALC_MAP_OUTPUT)
    print("Saved:", CALC_MAP_OUTPUT)

    print("\n===================================")
    print(" Calcification + Agatston Analysis Finished ")
    print("===================================")


# ==========================================================
# Part 4:
# FAI Analysis + Main Pipeline
# ==========================================================



FAI_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "fai_result.csv"
)



# ==========================================================
# 9. FAI Analysis
#
# New implementation:
# - Perivascular ROI outside the segmented vessel
# - ROI distance: 3 mm
# - Fat HU range: -190 to -30 HU
# - Saves CSV + ROI map + fat mask + HU map
# ==========================================================

FAT_MIN_HU = -190
FAT_MAX_HU = -30

FAI_MAP_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "FAI_map.nii.gz"
)

FAI_ROI_MAP_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "FAI_ROI.nii.gz"
)

FAT_MASK_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "FAI_fat_mask.nii.gz"
)


def calculate_fai():
 
    print("\n===================================")
    print(" FAI Analysis Start ")
    print("===================================")
 
    # ------------------------------------------------------
    # Check files
    # ------------------------------------------------------
 
    if not os.path.exists(CTA_PATH):
        raise FileNotFoundError(
            "Missing CTA: " + CTA_PATH
        )
 
    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(
            "Missing label: " + LABEL_PATH
        )
 
    # ------------------------------------------------------
    # Load CTA
    # ------------------------------------------------------
 
    cta_img = sitk.ReadImage(
        CTA_PATH
    )
 
    cta = sitk.GetArrayFromImage(
        cta_img
    )
 
    # ------------------------------------------------------
    # Load segmentation
    # ------------------------------------------------------
 
    label_img = sitk.ReadImage(
        LABEL_PATH
    )
 
    label = sitk.GetArrayFromImage(
        label_img
    )
 
    print("\nCTA shape:", cta.shape)
    print("Label shape:", label.shape)
 
    print(
        "Labels:",
        np.unique(label)
    )
 
    print(
        "CTA HU min:",
        np.min(cta)
    )
 
    print(
        "CTA HU max:",
        np.max(cta)
    )
 
    # ------------------------------------------------------
    # Geometry check
    # ------------------------------------------------------
 
    if cta.shape != label.shape:
 
        raise ValueError(
            "CTA and label shapes do not match: "
            f"{cta.shape} vs {label.shape}"
        )
 
    spacing = cta_img.GetSpacing()
 
    print(
        "CTA spacing:",
        spacing
    )
 
    voxel_volume = (
        spacing[0]
        *
        spacing[1]
        *
        spacing[2]
    )
 
    # ------------------------------------------------------
    # Vessel labels
    # ------------------------------------------------------
 
    vessels = {
        "RECA": 1,
        "RICA": 2,
        "LECA": 3,
        "LICA": 4
    }
 
    # ------------------------------------------------------
    # Output maps
    # ------------------------------------------------------
 
    roi_map = np.zeros_like(
        label,
        dtype=np.uint8
    )
 
    fat_mask_map = np.zeros_like(
        label,
        dtype=np.uint8
    )
 
    fai_map = np.full(
        cta.shape,
        np.nan,
        dtype=np.float32
    )
 
    results = []
 
    # ======================================================
    # Process vessels
    # ======================================================
 
    for vessel, label_value in vessels.items():
 
        print("\n-------------------------------")
        print("Processing:", vessel)
        print("-------------------------------")
 
        vessel_mask = (
            label == label_value
        )
 
        vessel_voxels = int(
            np.sum(vessel_mask)
        )
 
        print(
            "Vessel voxels:",
            vessel_voxels
        )
 
        if vessel_voxels == 0:
 
            print(
                vessel,
                "not found."
            )
 
            results.append(
                [
                    vessel,
                    "",
                    0,
                    0,
                    0,
                    "",
                    ""
                ]
            )
 
            continue
 
        # --------------------------------------------------
        # Create perivascular ROI
        # --------------------------------------------------
 
        roi = create_perivascular_roi(
            vessel_mask,
            cta_img,
            PERIVASCULAR_DISTANCE_MM
        )
 
        roi_voxels = int(
            np.sum(roi)
        )
 
        roi_volume = (
            roi_voxels
            *
            voxel_volume
        )
 
        roi_map[
            roi
        ] = label_value
 
        print(
            "ROI voxels:",
            roi_voxels
        )
 
        print(
            "ROI volume:",
            round(roi_volume, 3),
            "mm3"
        )
 
        # --------------------------------------------------
        # Check HU distribution inside ROI
        # --------------------------------------------------
 
        if roi_voxels > 0:
 
            roi_values = cta[
                roi
            ]
 
            roi_min_hu = float(
                np.min(roi_values)
            )
 
            roi_max_hu = float(
                np.max(roi_values)
            )
 
            roi_mean_hu = float(
                np.mean(roi_values)
            )
 
        else:
 
            roi_min_hu = None
            roi_max_hu = None
            roi_mean_hu = None
 
        print(
            "ROI HU min:",
            roi_min_hu
        )
 
        print(
            "ROI HU max:",
            roi_max_hu
        )
 
        print(
            "ROI HU mean:",
            roi_mean_hu
        )
 
        # --------------------------------------------------
        # Fat voxels
        # --------------------------------------------------
 
        fat_mask = (
            roi
&
            (cta >= FAT_MIN_HU)
&
            (cta <= FAT_MAX_HU)
        )
 
        fat_voxels = int(
            np.sum(fat_mask)
        )
 
        fat_volume = (
            fat_voxels
            *
            voxel_volume
        )
 
        print(
            "Unique fat voxels:",
            fat_voxels
        )
 
        print(
            "Fat volume:",
            round(fat_volume, 3),
            "mm3"
        )
 
        # --------------------------------------------------
        # Calculate FAI
        # --------------------------------------------------
 
        if fat_voxels > 0:
 
            fat_values = cta[
                fat_mask
            ]
 
            fai_value = float(
                np.mean(
                    fat_values
                )
            )
 
            fat_min_hu = float(
                np.min(
                    fat_values
                )
            )
 
            fat_max_hu = float(
                np.max(
                    fat_values
                )
            )
 
    
            fai_map[
                fat_mask
            ] = cta[
                fat_mask
            ]
 
            fat_mask_map[
                fat_mask
            ] = label_value
 
        else:
 
            fai_value = None
            fat_min_hu = None
            fat_max_hu = None
 
        print(
            "FAI:",
            fai_value
        )
 
        print(
            "Fat HU min:",
            fat_min_hu
        )
 
        print(
            "Fat HU max:",
            fat_max_hu
        )
 
        # --------------------------------------------------
        # Save result
        # --------------------------------------------------
 
        results.append(
            [
                vessel,
 
                (
                    round(
                        fai_value,
                        3
                    )
                    if fai_value is not None
                    else ""
                ),
 
                fat_voxels,
 
                round(
                    fat_volume,
                    3
                ),
 
                roi_voxels,
 
                (
                    round(
                        roi_min_hu,
                        2
                    )
                    if roi_min_hu is not None
                    else ""
                ),
 
                (
                    round(
                        roi_max_hu,
                        2
                    )
                    if roi_max_hu is not None
                    else ""
                )
            ]
        )
 
    # ======================================================
    # Save CSV
    # ======================================================
 
    with open(
        FAI_OUTPUT,
        "w",
        newline=""
    ) as f:
 
        writer = csv.writer(f)
 
        writer.writerow(
            [
                "vessel",
                "FAI_HU",
                "fat_voxel_number",
                "fat_volume_mm3",
                "ROI_voxel_number",
                "ROI_min_HU",
                "ROI_max_HU"
            ]
        )
 
        writer.writerows(
            results
        )
 
    print(
        "\nSaved:",
        FAI_OUTPUT
    )
 
    # ======================================================
    # Save ROI map
    # ======================================================
 
    roi_img = sitk.GetImageFromArray(
        roi_map
    )
 
    roi_img.CopyInformation(
        cta_img
    )
 
    sitk.WriteImage(
        roi_img,
        FAI_ROI_MAP_OUTPUT
    )
 
    print(
        "Saved:",
        FAI_ROI_MAP_OUTPUT
    )
 
    # ======================================================
    # Save fat mask
    # ======================================================
 
    fat_img = sitk.GetImageFromArray(
        fat_mask_map
    )
 
    fat_img.CopyInformation(
        cta_img
    )
 
    sitk.WriteImage(
        fat_img,
        FAT_MASK_OUTPUT
    )
 
    print(
        "Saved:",
        FAT_MASK_OUTPUT
    )
 
    # ======================================================
    # Save FAI HU map
    # ======================================================
 
    fai_img = sitk.GetImageFromArray(
        fai_map
    )
 
    fai_img.CopyInformation(
        cta_img
    )
 
    sitk.WriteImage(
        fai_img,
        FAI_MAP_OUTPUT
    )
 
    print(
        "Saved:",
        FAI_MAP_OUTPUT
    )
 
    print("\n===================================")
    print(" FAI Analysis Finished ")
    print("===================================")


# ==========================================================
# 10. Main Pipeline
# ==========================================================


def main():


    print("\n")
    print("===================================")
    print(" Carotid Analysis Pipeline Start ")
    print("===================================")



    # Step 0
    check_files()



    # Step 1
    generate_centerline()



    # Step 2
    calculate_length()



    # Step 3
    calculate_radius()



    # Step 4
    calculate_stenosis()



    # Step 5
    calculate_calcification()



    # Step 6
    calculate_fai()



    print("\n")
    print("===================================")
    print(" Pipeline Finished Successfully ")
    print("===================================")



    print("\nGenerated files:")


    print(
        CENTERLINE_OUTPUT
    )


    print(
        LENGTH_OUTPUT
    )


    print(
        RADIUS_OUTPUT
    )


    print(
        STENOSIS_OUTPUT
    )


    print(
        CALCIFICATION_OUTPUT
    )


    print(
        FAI_OUTPUT
    )

    print(
        CALC_ROI_MAP_OUTPUT
    )

    print(
        CALC_MAP_OUTPUT
    )

    print(
        FAI_ROI_MAP_OUTPUT
    )

    print(
        FAT_MASK_OUTPUT
    )

    print(
        FAI_MAP_OUTPUT
    )




# ==========================================================
# Run
# ==========================================================


if __name__ == "__main__":

    main()