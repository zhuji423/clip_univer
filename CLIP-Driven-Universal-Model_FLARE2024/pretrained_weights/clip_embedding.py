import os
import clip
import torch


## PAOT
# ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus', 
#                 'Liver', 'Stomach', 'Arota', 'Postcava', 'Portal Vein and Splenic Vein',
#                 'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum', 'Hepatic Vessel',
#                 'Right Lung', 'Left Lung', 'Colon', 'Intestine', 'Rectum', 
#                 'Bladder', 'Prostate', 'Left Head of Femur', 'Right Head of Femur', 'Celiac Truck',
#                 'Kidney Tumor', 'Liver Tumor', 'Pancreas Tumor', 'Hepatic Vessel Tumor', 'Lung Tumor', 
#                 'Colon Tumor', 'Kidney Cyst',"artery",'vein']

ORGAN_NAME = ['Novel Coronavirus Pneumonia',
              "Kidney lesions and Bone lesions and Pulmonary nodules and Swollen lymph nodes",
              "Kidney Tumor and Kidney Cyst",
              "Lung nodules",
              "Adrenocortical carcinoma"  ## Adrenal 肾上腺皮质癌
              "mediastinal lymph-nodes and celiac lymph node", ##  Lymph nodes 腹部和纵隔淋巴结
              "Non-small cell lung cancer and  Pleural effusion",  ## 非小细胞肺癌 胸腔积液
              "whole body cancer or tumor",
              ####msd####
              "colon Tumor",
              "pancreas Tumor",
              "Hepatic Vessel Tumor",
              "lung tumor",
              "liver tumor",
              ####msd####
              ###amos####
              "Spleen tumor, Right kidney tumor, Left kidney tumor,Gallbladder tumor, Esophagus tumor, Liver tumor, Stomach tumor, Aorta tumor,Postcava tumor, Pancreas tumor, Right Adrenal Gland tumor, \
                Left Adrenal Gland tumor, Duodenum tumor, Bladder tumor, and Prostate/Uterus tumor",
              ###amos####
              "abdominal trauma with visceral organ injury and internal bleeding, including liver, spleen, kidneys, and intestines"]
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


text_inputs = torch.cat([clip.tokenize(f'A computerized tomography of a {item}') for item in ORGAN_NAME]).to(device)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(text_features.shape, text_features.dtype)
    torch.save(text_features, '/pub/data/yangdeq/CLIP/pretrained_weight/txt_encoding_Flare2024.pth')

