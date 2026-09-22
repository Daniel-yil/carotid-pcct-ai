import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from config import (
    TEST_IMAGES,
    PREDICTION_DIR,
)

#This is where you need to change to the name of the file#
CASE_ID = "carotid_050"


def main():

    image_path = (
        TEST_IMAGES
        / f"{CASE_ID}_0000.nii.gz"
    )

    prediction_path = (
        PREDICTION_DIR
        / f"{CASE_ID}.nii.gz"
    )

    image = sitk.ReadImage(
        str(image_path)
    )

    prediction = sitk.ReadImage(
        str(prediction_path)
    )

    image_array = sitk.GetArrayFromImage(
        image
    )

    prediction_array = (
        sitk.GetArrayFromImage(
            prediction
        )
    )

    slices_with_segmentation = np.where(
        np.any(
            prediction_array > 0,
            axis=(1, 2)
        )
    )[0]

    if len(slices_with_segmentation) == 0:

        print(
            "No foreground segmentation found."
        )

        return

    slice_index = slices_with_segmentation[
        len(slices_with_segmentation) // 2
    ]

    ct_slice = image_array[
        slice_index
    ]

    seg_slice = prediction_array[
        slice_index
    ]

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        ct_slice,
        cmap="gray"
    )

    masked_segmentation = np.ma.masked_where(
        seg_slice == 0,
        seg_slice
    )

    plt.imshow(
        masked_segmentation,
        alpha=0.45
    )

    plt.title(
        f"{CASE_ID} - Slice {slice_index}"
    )

    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()