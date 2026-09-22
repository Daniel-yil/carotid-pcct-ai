import shutil
 
import numpy as np
import SimpleITK as sitk
 
from config import LABELS_TR
 
 
OLD_VALUE = 255
NEW_VALUE = 4
 
BACKUP_DIR = LABELS_TR.parent / "labelsTr_backup"
 
 
def main():
 
    if not BACKUP_DIR.exists():
        shutil.copytree(LABELS_TR, BACKUP_DIR)
 
        print(
            f"Backup created:\n"
            f"{BACKUP_DIR}\n"
        )
 
    files = sorted(LABELS_TR.glob("*.nii.gz"))
 
    for index, path in enumerate(files, start=1):
 
        print(
            f"[{index}/{len(files)}] "
            f"Processing {path.name}"
        )
 
        image = sitk.ReadImage(str(path))
 
        array = sitk.GetArrayFromImage(image)
 
        old_values = np.unique(array)
 
        if OLD_VALUE not in old_values:
            print("No remapping required.")
            continue
 
        array[array == OLD_VALUE] = NEW_VALUE
 
        output = sitk.GetImageFromArray(
            array.astype(np.uint8)
        )
 
        output.CopyInformation(image)
 
        sitk.WriteImage(
            output,
            str(path),
            useCompression=True
        )
 
        print(
            "Changed:",
            old_values,
            "→",
            np.unique(array)
        )
 
    print("\nLabel remapping complete.")
 
 
if __name__ == "__main__":
    main()