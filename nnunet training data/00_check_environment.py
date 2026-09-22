import sys
import torch
import nnunetv2


def main():
    print("Python:")
    print(sys.version)

    print("\nPyTorch:")
    print(torch.__version__)

    print("\nCUDA available:")
    print(torch.cuda.is_available())

    if torch.cuda.is_available():
        print("\nGPU:")
        print(torch.cuda.get_device_name(0))

        print("\nCUDA version:")
        print(torch.version.cuda)

    print("\nnnU-Net:")
    print("nnUNetv2 imported successfully")

    print("\nEnvironment ready.")


if __name__ == "__main__":
    main()