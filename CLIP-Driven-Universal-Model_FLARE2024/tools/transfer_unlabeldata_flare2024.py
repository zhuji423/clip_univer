import os
import shutil


amos_path = r"/pub/data/yangdeq/Flare2024/Train-Unlabeled/AMOS"
RSNA_path = r"/pub/data/yangdeq/Flare2024/Train-Unlabeled/RSNA-Abdomen-image-1240"
label_path = r"/pub/data/yangdeq/Flare2024/26_FLARE2024/converted_unlabel"
dest_path_amos = r"/pub/data/yangdeq/Flare2024/26_FLARE2024/imagesTr/14_AMOS"
dest_path_rsna = r"/pub/data/yangdeq/Flare2024/26_FLARE2024/imagesTr/15_RSNA"
for names in os.listdir(label_path):
    img_name = names.split(".nii.gz")[0] + "_0000.nii.gz"
    # print(names,img_name)
    img_path_amos = os.path.join(amos_path, img_name)
    img_path_rsna = os.path.join(RSNA_path, img_name)
    if "amos" in names and not os.path.exists(img_path_amos):
        shutil.copy(img_path_amos, os.path.join(dest_path_amos, img_name))
        print(f"succefully transfer {img_name} to {os.path.join(dest_path_amos, img_name)}")
    elif "RSNA" in names and not os.path.exists(img_path_rsna):
        shutil.copy(img_path_rsna, os.path.join(dest_path_rsna, img_name))
        print(f"succefully transfer {img_name} to {os.path.join(dest_path_rsna, img_name)}")

    else:
        print(f"file:{img_name}  exists!!!!!!!!!!!!!")
