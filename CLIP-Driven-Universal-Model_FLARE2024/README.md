# Language-guided Vision Model for Pan-cancer Segmentation in Abdominal CT Scans
Flare2024 solution:

## 0. Preliminary
```bash
python3 -m venv universal
source /data/zzhou82/environments/universal/bin/activate

git clone https://github.com/zhuji423/clip_univer.git
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install 'monai[all]'
pip install -r requirements.txt
cd pretrained_weights/
wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
wget wget https://www.dropbox.com/s/lh5kuyjxwjsxjpl/Genesis_Chest_CT.pt
cd ../
```

**Dataset Pre-Process**  
1. Download the dataset according to the dataset link and arrange the dataset according to the `dataset/dataset_list/PAOT.txt`.  
2. Modify [ORGAN_DATASET_DIR](https://github.com/ljwztc/CLIP-Driven-Universal-Model/blob/main/label_transfer.py#L51C1-L51C18) and [NUM_WORKER](https://github.com/ljwztc/CLIP-Driven-Universal-Model/blob/main/label_transfer.py#L53) in label_transfer.py  
3. `python -W ignore label_transfer.py`


**Current Template**
|  Index   | Organ  | Index | Organ |
|  ----  | ----  |  ----  | ----  |
| 1  | Spleen | 17 | Left Lung |
| 2  | Right Kidney | 18  | Colon |
| 3  | Left Kidney | 19  | Intestine |
| 4  | Gall Bladder | 20  | Rectum |
| 5  | Esophagus | 21  | Bladder |
| 6  | Liver | 22  | Prostate |
| 7  | Stomach | 23  | Left Head of Femur |
| 8  | Aorta | 24  | Right Head of Femur |
| 9  | Postcava | 25  | Celiac Trunk |
| 10  | Portal Vein and Splenic Vein | 26  | Kidney Tumor |
| 11  | Pancreas | 27  | Liver Tumor |
| 12  | Right Adrenal Gland | 28  | Pancreas Tumor |
| 13  | Left Adrenal Gland | 29  | Hepatic Vessel Tumor |
| 14  | Duodenum | 30  | Lung Tumor |
| 15  | Hepatic Vessel | 31  | Colon Tumor |
| 16  | Right Lung | 32  | Kidney Cyst |

**How expand to new dataset with new organ?**
1. Set the following index for new organ. (e.g. 33 for vermiform appendix)  
2. Check if there are any organs that are not divided into left and right in the dataset. (e.g. kidney, lung, etc.) The `RL_Splitd` in `label_transfer.py` is used to processed this case.  
3. Set up a new transfer list for new dataset in TEMPLATE (line 58 in label_transfer.py). (If a new dataset with Intestine labeled as 1 and vermiform appendix labeled as 2, we set the transfer list as [19, 33])  
4. Run the program `label_transfer.py` to get new post-processing labels.  

## 1. Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train.py --dist True --data_root_path /mnt/zzhou82/PublicAbdominalData/ --num_workers 12 --num_samples 4 --cache_dataset --cache_rate 0.6 --uniform_sample
```

## 2. Validation

```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --data_root_path /mnt/zzhou82/PublicAbdominalData/ --start_epoch 10 --end_epoch 40 --epoch_interval 10 --cache_dataset --cache_rate 0.6
```

## 3. Evaluation
```
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --resume ./out/epoch_61.pth --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result --cache_dataset --cache_rate 0.6
```

## 4. Pretrained weights
We have publish the pretrained weights on [Zenodo](https://zenodo.org/records/15192753).