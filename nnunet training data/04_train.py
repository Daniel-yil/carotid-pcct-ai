import os
import subprocess

from config import (
    DATASET_ID,
    CONFIGURATION,
    FOLD,
    NNUNET_RAW,
    NNUNET_PREPROCESSED,
    NNUNET_RESULTS,
)


CONTINUE_TRAINING = True


def main():

    os.environ["nnUNet_raw"] = str(NNUNET_RAW)

    os.environ["nnUNet_preprocessed"] = str(
        NNUNET_PREPROCESSED
    )

    os.environ["nnUNet_results"] = str(
        NNUNET_RESULTS
    )

    command = [
        "nnUNetv2_train",
        str(DATASET_ID),
        CONFIGURATION,
        str(FOLD),
    ]

    if CONTINUE_TRAINING:
        command.append("--c")

    print("Running:")
    print(" ".join(command))

    subprocess.run(
        command,
        check=True
    )


if __name__ == "__main__":
    main()