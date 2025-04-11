import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import glob
import json
from monai.inferers import sliding_window_inference

from model.Universal_model import Universal_model
from dataset.dataloader import get_loader
from utils import loss
from utils.utils import dice_score, TEMPLATE, ORGAN_NAME, merge_label, visualize_label, get_key, NUM_CLASS
from utils.utils import extract_topk_largest_candidates, organ_post_process, threshold_organ
from tools import cut_pred_for_dice
from medpy.metric.binary import __surface_distances
from collections import defaultdict
torch.multiprocessing.set_sharing_strategy('file_system')

def normalized_surface_dice(a: np.ndarray, b: np.ndarray, threshold: float, spacing: tuple = None, connectivity=1):
    """
    This implementation differs from the official surface dice implementation! These two are not comparable!!!!!
    The normalized surface dice is symmetric, so it should not matter whether a or b is the reference image
    This implementation natively supports 2D and 3D images. Whether other dimensions are supported depends on the
    __surface_distances implementation in medpy
    :param a: image 1, must have the same shape as b
    :param b: image 2, must have the same shape as a
    :param threshold: distances below this threshold will be counted as true positives. Threshold is in mm, not voxels!
    (if spacing = (1, 1(, 1)) then one voxel=1mm so the threshold is effectively in voxels)
    must be a tuple of len dimension(a)
    :param spacing: how many mm is one voxel in reality? Can be left at None, we then assume an isotropic spacing of 1mm
    :param connectivity: see scipy.ndimage.generate_binary_structure for more information. I suggest you leave that
    one alone
    :return:
    """
    assert all([i == j for i, j in zip(a.shape, b.shape)]), "a and b must have the same shape. a.shape= %s, " \
                                                            "b.shape= %s" % (str(a.shape), str(b.shape))
    if spacing is None:
        spacing = tuple([1 for _ in range(len(a.shape))])
    a_to_b = __surface_distances(a, b, spacing, connectivity)
    b_to_a = __surface_distances(b, a, spacing, connectivity)
 
    numel_a = len(a_to_b)
    numel_b = len(b_to_a)
 
    tp_a = np.sum(a_to_b <= threshold) / numel_a
    tp_b = np.sum(b_to_a <= threshold) / numel_b
 
    fp = np.sum(a_to_b > threshold) / numel_a
    fn = np.sum(b_to_a > threshold) / numel_b
 
    dc = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + 1e-8)  # 1e-8 just so that we don't get div by 0
    return dc

def validation(model, ValLoader, args, idx):
    model.eval()
    dice_list = {}
    nsd_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
        nsd_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
    save_dice_log = defaultdict(list)
    for index, batch in enumerate(tqdm(ValLoader)):
        # print('%d processd' % (index))
        image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]
        with torch.no_grad():
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=args.overlap, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)
        pred_hard = threshold_organ(pred_sigmoid)
        pred_hard = pred_hard.cpu()
        torch.cuda.empty_cache()
        B = pred_sigmoid.shape[0]
        for b in range(B):
            content = 'case%s| '%(name[b])
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            pred_hard_post = organ_post_process(pred_hard.numpy(), organ_list, args.log_name+'/'+name[0].split('/')[0]+'/'+name[0].split('/')[-1],args)
            pred_hard_post = torch.tensor(pred_hard_post)
            for i in range(1,3):
                pred_hard_post[b,-i] = cut_pred_for_dice.compute_mask(pred_hard_post[b,-i],label[b,-i])
            # pred_hard_post = pred_hard
            for organ in organ_list:
                if torch.sum(label[b,organ-1,:,:,:]) != 0:
                    dice_organ, recall, precision = dice_score(pred_hard_post[b,organ-1,:,:,:].cuda(), label[b,organ-1,:,:,:].cuda())
                    dice_list[template_key][0][organ-1] += dice_organ.item()
                    dice_list[template_key][1][organ-1] += 1
                    content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice_organ.item())
                    save_dice_log[content] = '%s: dice %.4f, recall %.4f, precision %.4f'%(ORGAN_NAME[organ-1], dice_organ.item(), recall.item(), precision.item())
                    print('%s: dice %.4f, recall %.4f, precision %.4f'%(ORGAN_NAME[organ-1], dice_organ.item(), recall.item(), precision.item()))
            print(content)

        torch.cuda.empty_cache()
    
    ave_organ_dice = np.zeros((2, NUM_CLASS))

    with open('/pub/data/yangdeq/CLIP/data/vein/out/'+args.log_name+f'/b_val_{idx}.json', 'w') as f:
        for key in TEMPLATE.keys():
            organ_list = TEMPLATE[key]
            content = 'Task%s| '%(key)
            # content1 = 'NSD Task%s| '%(key)
            for organ in organ_list:
                dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1] ##当前这个器官的平均dice，除以这个器官出现的次数
                content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice)
                ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
                ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1]

            print(content)
            f.write(content)
            f.write('\n')
        content = 'Average | '
        for i in range(NUM_CLASS):
            content += '%s: %.4f, '%(ORGAN_NAME[i], ave_organ_dice[0][i] / ave_organ_dice[1][i])
        print(content)
        f.write(content)
        f.write('\n')

        print(np.mean(ave_organ_dice[0] / ave_organ_dice[1]))
        f.write('%s: %.4f, '%('average', np.mean([i for i in ave_organ_dice[0] / ave_organ_dice[1] if not np.isnan(i) and i != 0])))
        f.write('\n')
        f.write(json.dumps(save_dice_log))
    model_dice = np.mean([i for i in ave_organ_dice[0] / ave_organ_dice[1] if not np.isnan(i)])
    return model_dice




def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='vein_swinunetr_new_clip_400', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--start_epoch', default=490, type=int, help='Number of start epoches')
    parser.add_argument('--end_epoch', default=490, type=int, help='Number of end epoches')
    parser.add_argument('--epoch_interval', default=100, type=int, help='Number of start epoches')
    parser.add_argument('--backbone', default='swinunetr', help='backbone [swinunetr or unet]')

    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['vein_artery']) # 'PAOT', 'felix' 'PAOT_123457891213', 'PAOT_10_inner', 'PAOT_tumor'

    parser.add_argument('--data_root_path', default='/pub/data/yangdeq/CLIP/data/vein/', help='data root path')
    parser.add_argument('--data_txt_path', default='/home/yangdq/project/module/CLIP-Driven-Universal-Model/dataset/vein_list/', help='data txt path')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='validation', help='train or validation or test')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--overlap', default=0.5, type=float, help='overlap for sliding_window_inference')

    args = parser.parse_args()

    # prepare the 3D model
    model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    backbone=args.backbone,
                    encoding='word_embedding'
                    )

    #Load pre-trained weights
    store_path_root = f'/pub/data/yangdeq/CLIP/data/vein/out/{args.log_name}/epoch_***.pth'
    # store_path_root = '/pub/data/yangdeq/CLIP/data/amos/out/unet/swinunetr.pth'
    for store_path in glob.glob(store_path_root):
        if "DICE" not in store_path:
            print(f"i am  validation {store_path}")
            # if "swin" in store_path:
            # store_path = store_path_root
            store_dict = model.state_dict()
            load_dict = torch.load(store_path)['net']

            for key, value in load_dict.items():
                if 'swinViT' in key or 'encoder' in key or 'decoder' in key:
                    name = '.'.join(key.split('.')[1:])
                    # name = 'backbone.' + name
                else:
                    name = '.'.join(key.split('.')[1:])
                store_dict[name] = value

            model.load_state_dict(store_dict)
            print(f'Load {store_path} weights')

            model.cuda()

            torch.backends.cudnn.benchmark = True

            validation_loader, val_transforms = get_loader(args)
            i = int(store_path.split('_')[-1].split('.')[0])#+1
            # i = 1
            model_mean_dice = validation(model, validation_loader, args, i)
            new_name = store_path.replace(str(i), str(i)+'_DICE_AVG:'+str(np.round(model_mean_dice,4)))
            os.rename(store_path, new_name)

if __name__ == "__main__":
    main()

#python validation.py >> out/Nvidia/ablation_clip/clip2.txt