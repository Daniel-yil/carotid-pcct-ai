import numpy as np

import SimpleITK as sitk
 
from config import LABELS_TR
 
 
def main():
 
    print("Checking labels in:")

    print(LABELS_TR)
 
    if not LABELS_TR.exists():

        print("\nERROR: labelsTr folder does not exist.")

        return
 
    label_files = sorted(

        LABELS_TR.glob("*.nii.gz")

    )
 
    print(

        f"\nFound {len(label_files)} label files."
        
    )
 
    if len(label_files) == 0:

        print(

            "\nERROR: No .nii.gz label files found."

        )

        return
 
    all_values = set()
 
    for index, label_file in enumerate(

        label_files,

        start=1

    ):
 
        image = sitk.ReadImage(

            str(label_file)

        )
 
        array = sitk.GetArrayFromImage(

            image

        )
 
        values = np.unique(array)
 
        all_values.update(

            values.tolist()

        )
 
        print(

            f"[{index}/{len(label_files)}] "

            f"{label_file.name}: {values}"

        )
 
    print("\n==============================")

    print("Label check finished")

    print("==============================")
 
    print(

        "All label values found:",

        sorted(all_values)

    )
 
 
if __name__ == "__main__":

    main()
 