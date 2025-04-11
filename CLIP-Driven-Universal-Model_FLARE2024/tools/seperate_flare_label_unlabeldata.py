import os
import SimpleITK  as sitk
import numpy as np
# image = r"/pub/data/yangdeq/Flare2024/26_FLARE2024/imagesTr"
label = r"/pub/data/yangdeq/Flare2024/Train-Unlabeled/pseudo_label/pseudo-label-FLARE23-1st-aladdin5-seg-RSNA-Tumor"
save_label_path  = r"/pub/data/yangdeq/Flare2024/26_FLARE2024/converted_unlabel"
s = 0
map_dict = {"amos":14,"RSNA":15}
for names in os.listdir(label):
    image_path = os.path.join(label, names)
    print(image_path)
    # for name in os.listdir(image_path):
    # print(name)
    # new_name = name.split("_0000.nii.gz")[0] + ".nii.gz"
    save_path = os.path.join(save_label_path,names)
    if not os.path.exists(save_path):
        lb_name = os.path.join(label,names)
        print(f"lb_name:{lb_name},names:{names} label: {names[0]} sum:{s}")
        ## read lb_name
        lb = sitk.ReadImage(lb_name)
        lb_array = sitk.GetArrayFromImage(lb)
        
        lb_array[lb_array == 1] = map_dict[names[:4]]
        lb_unique = np.unique(lb_array)
        ## save lb_name
        savemask = sitk.GetImageFromArray(lb_array)
        savemask.SetOrigin(lb.GetOrigin())
        savemask.SetDirection(lb.GetDirection())
        savemask.SetSpacing(lb.GetSpacing())
        # save_path = os.path.join(save_label_path,str(names[0]+"_"+new_name))
        
        sitk.WriteImage(savemask,save_path)
        print(f"file:{lb_name} have been changed into {lb_unique}")
        s+=1
        # break
    else:
        print(f"file:{save_path}  exists!!!!!!!!!!!!!")
