import SimpleITK as sitk
from pathlib import Path
 
#This is where you can change the path
image_dir = Path(
    r"C:\Users\Desktop\nnUNet_data\nnUNet_raw\Dataset501_Carotid\imagesTr"
)
 
label_dir = Path(
    r"C:\Users\Desktop\nnUNet_data\nnUNet_raw\Dataset501_Carotid\labelsTr"
)
 
 
for mask_path in sorted(label_dir.glob("*.nii.gz")):
 
    case = mask_path.name.replace(".nii.gz", "")
 
    image_path = image_dir / f"{case}_0000.nii.gz"
 
    if not image_path.exists():
        print("Missing image:", case)
        continue
 
    print("Processing:", case)
 
    
    image = sitk.ReadImage(str(image_path))
 
    
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
 
    
    mask = sitk.ReadImage(str(mask_path))
 
    print("Before:")
    print("Image spacing:", spacing)
    print("Mask spacing:", mask.GetSpacing())
 

    mask.SetSpacing(spacing)
    mask.SetOrigin(origin)
   
    sitk.WriteImage(
        mask,
        str(mask_path)
    )
 
    print("After:")
    print("Mask spacing:", mask.GetSpacing())
    print("Fixed:", case)
    print("------------------------")
 
 
print("All finished!")