import os
import subprocess

from config import (
    DATASET_ID,
    CONFIGURATION,
    FOLD,
    CHECKPOINT,
    TEST_IMAGES,
    PREDICTION_DIR,
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

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    command = [
        "nnUNetv2_predict",

        "-i",
        str(TEST_IMAGES),

        "-o",
        str(PREDICTION_DIR),

        "-d",
        str(DATASET_ID),

        "-c",
        CONFIGURATION,

        "-f",
        str(FOLD),

        "-chk",
        CHECKPOINT,
    ]

    print("Running:")
    print(" ".join(command))

    subprocess.run(
        command,
        check=True
    )

    print("\nPrediction completed.")

    print(
        "Results saved to:",
        PREDICTION_DIR
    )


if __name__ == "__main__":
    main()