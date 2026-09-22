import os
import subprocess
 
from config import (
    DATASET_ID,
    NNUNET_RAW,
    NNUNET_PREPROCESSED,
    NNUNET_RESULTS,
)
 
 
def main():
 
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
 
    os.environ["nnUNet_preprocessed"] = str(
        NNUNET_PREPROCESSED
    )
 
    os.environ["nnUNet_results"] = str(
        NNUNET_RESULTS
    )
 
    command = [
        "nnUNetv2_plan_and_preprocess",
        "-d",
        str(DATASET_ID),
        "--verify_dataset_integrity",
    ]
 
    print("Running:")
    print(" ".join(command))
 
    subprocess.run(
        command,
        check=True
    )
 
    print("\nPreprocessing completed.")
 
 
if __name__ == "__main__":
    main()