import os
import SimpleITK  as sitk
import numpy as np
image = r"/pub/data/yangdeq/Flare2024/26_FLARE2024/imagesTr"
label = r"/pub/data/yangdeq/Flare2024/26_FLARE2024/labelsTr"
save_label_path  = r"/pub/data/yangdeq/Flare2024/26_FLARE2024/converted_label"
s = 0
for names in os.listdir(image):
    image_path = os.path.join(image, names)
    if "MSD" not in names :
        pass
        # print(image_path)
        # for name in os.listdir(image_path):
        #     print(name)
        #     new_name = name.split("_0000.nii.gz")[0] + ".nii.gz"
        #     save_path = os.path.join(save_label_path,new_name)
        #     if not os.path.exists(save_path):
        #         lb_name = os.path.join(label,new_name)
        #         print(f"lb_name:{lb_name},names:{names} label: {names[0]} sum:{s}")
        #         ## read lb_name
        #         lb = sitk.ReadImage(lb_name)
        #         lb_array = sitk.GetArrayFromImage(lb)
        #         lb_array[lb_array == 1] = int(names[0])
        #         lb_unique = np.unique(lb_array)
        #         ## save lb_name
        #         savemask = sitk.GetImageFromArray(lb_array)
        #         savemask.SetOrigin(lb.GetOrigin())
        #         savemask.SetDirection(lb.GetDirection())
        #         savemask.SetSpacing(lb.GetSpacing())
        #         # save_path = os.path.join(save_label_path,str(names[0]+"_"+new_name))
                
        #         sitk.WriteImage(savemask,save_path)
        #         print(f"file:{lb_name} have been changed into {lb_unique}")
        #         s+=1
        #     else:
        #         print(f"file:{save_path}  exists!!!!!!!!!!!!!")
    
    elif "MSD" in names:
        print(image_path)
        msd_dict = {'MSD_colon': 9, 'MSD_pancreas': 10, 'MSD_hepaticvessel': 11, 'MSD_lung': 12, 'MSD_liver': 13}
        # used for generate the mask label for msd dataset
        msd_lb = set()
        # for name in os.listdir(image_path):
        #     category = name.split("_0000.nii.gz")[0].split("_")[0] + "_"  +name.split("_0000.nii.gz")[0].split("_")[1]
        #     print(category)
        #     msd_lb.add(category)
        # msd_dict = dict.fromkeys(msd_lb)
        # print(msd_dict)
        for name in os.listdir(image_path):
            category = name.split("_0000.nii.gz")[0].split("_")[0] + "_"  +name.split("_0000.nii.gz")[0].split("_")[1]
            new_name = name.split("_0000.nii.gz")[0] + ".nii.gz"
            lb_name = os.path.join(label,new_name)
            converted_label = category
            new_name = name.split("_0000.nii.gz")[0] + ".nii.gz"
            save_path = os.path.join(save_label_path,new_name)
            lb = sitk.ReadImage(lb_name)
            lb_array = sitk.GetArrayFromImage(lb)
            lb_array[lb_array == 1] = int(msd_dict[converted_label])
            lb_unique = np.unique(lb_array)
            # ## save lb_name
            savemask = sitk.GetImageFromArray(lb_array)
            savemask.SetOrigin(lb.GetOrigin())
            savemask.SetDirection(lb.GetDirection())
            savemask.SetSpacing(lb.GetSpacing())
            # save_path = os.path.join(save_label_path,str(names[0]+"_"+new_name))
            # try:
            #     os.remove(save_path)
            #     print(f"removed {save_path}")
            # except:
            #     pass
            sitk.WriteImage(savemask,save_path)
            print(f"file:{lb_name} have been changed into {lb_unique},label: {msd_dict[converted_label], } sum:{s}")
            s+=1
            # print(f"lb_name:{lb_name},names:{names} label: {msd_dict[converted_label]} sum:{s}")
            # break
