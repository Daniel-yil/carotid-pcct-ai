from pathlib import Path

#This is where you need to change the path
DATASET_ID = 501
DATASET_NAME = "Dataset501_Carotid"

#The places below are where you should change your path
BASE_DIR = Path(r"")   # set your nnU-Net data directory

NNUNET_RAW = BASE_DIR / "nnUNet_raw"
NNUNET_PREPROCESSED = BASE_DIR / "nnUNet_preprocessed"
NNUNET_RESULTS = BASE_DIR / "nnUNet_results"

DATASET_DIR = NNUNET_RAW / DATASET_NAME

IMAGES_TR = DATASET_DIR / "imagesTr"
LABELS_TR = DATASET_DIR / "labelsTr"

#train
#The place you should change
TEST_IMAGES = Path(r"  ")     #  set test image directory
PREDICTION_DIR = Path(r"    ")     # set prediction output directory

CONFIGURATION = "3d_fullres"
FOLD = 0

CHECKPOINT = "checkpoint_best.pth"
