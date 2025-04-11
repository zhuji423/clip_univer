import os
base_dir = r"/pub/data/yangdeq/Flare2024"
write_ct_path = r"26_FLARE2024/imagesTr/"
write_label_path = r"26_FLARE2024/converted_label/"
write_unlabel_path = r"26_FLARE2024/converted_unlabel/"

with open('/home/yangdq/project/module/CLIP-Driven-Universal-Model_FLARE2024/dataset/FLARE2024_list/Flare_train.txt', 'a+') as f:
    for file in os.listdir(os.path.join(base_dir, write_unlabel_path)):
        # name = file.split('_0000')[0]+".nii.gz"
        name = file.split('.nii.gz')[0]+"_0000.nii.gz"

        ct_name = os.path.join(write_ct_path, name)
        label_name = os.path.join(write_unlabel_path, file)
        print(ct_name,"   ", label_name)
        # if name not in os.listdir(labelpath):
        f.write(f"{ct_name}\t{label_name}\n")