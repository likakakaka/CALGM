
import os
import time
import json
import random
import argparse
import datetime
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import sys
from utils.configv2 import get_config
from utils.utils import *
from mmpretrain import get_model
from utils.datasets_pneum import *
print(f"||{torch.multiprocessing.get_start_method()}||", end="")
torch.multiprocessing.set_start_method("spawn", force=True)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda:1')
def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #将CuDNN的性能优化模式设置为关闭
    cudnn.benchmark = False
    #将CuDNN的确定性模式设置为启用,确保CuDNN在相同的输入下生成相同的输出
    cudnn.deterministic = True
    #CuDNN加速
    cudnn.enabled = True
    print(cudnn.benchmark, cudnn.deterministic, cudnn.enabled)
def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option():
    parser = argparse.ArgumentParser('CALGM', add_help=False)
    # easy config modification
    parser.add_argument('--batch-size', type=int, default=10, help="batch size for single GPU")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='/disk3/wjr/workspace/sec_nejm/temp2', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', default=True, action='store_true', help='Perform evaluation only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    parser.add_argument('--base_lr', type=float, default=1e-4)

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--seed', default=20, type=int)
    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')
    parser.add_argument('--pin_mem', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    args, unparsed = parser.parse_known_args()
    return args
from torchvision import transforms as pth_transforms
from PIL import ImageFilter, ImageOps
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import d_transform
def build_phe_loader(args):
    pil2tensor_transfo = d_transform.Compose([d_transform.Resize((512, 512)), d_transform.ToTensor(),
                                             d_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                             ])


    dataset_test = JSRT_wmask_Dataset('/disk3/wjr/dataset/JSRT/visimg_process/jsrt_seg_rec_img/',
                                  '/disk3/wjr/dataset/JSRT/visimg_process/jsrt_sec_mask_img',
                                   txtpath='/disk3/wjr/dataset/JSRT/test_1.txt',
                                   data_transform=pil2tensor_transfo)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    return data_loader_test


from calgm_model import *
from pathlib import Path
def main(config):
    start_time = time.time()
    data_loader_test = build_phe_loader(args)
    model = calgm(pretrained=True, drop_path_rate=0.)
    model.head.fc = nn.Linear(768, 2)
    model.head.detailed_sick_fc = nn.Linear(768, 5)
    model.head.detailed_health_fc = nn.Identity()
    args.save_path = '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/jsrt/512/contra_20251013/wmask_nosubcls_nodropouttest/our/g512_l256_6cls_2cls_tree_cam_wdatamixup_trainval/8034_0.5_cam_0.5_camalign_0.3_cloc_seed0_2_lr5e-05/analyze'
    ckp_path = '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/jsrt/512/contra_20251013/wmask_nosubcls_nodropouttest/our/g512_l256_6cls_2cls_tree_cam_wdatamixup_trainval/8034_0.5_cam_0.5_camalign_0.3_cloc_seed0_2_lr5e-05/model_best_auc_val.pth'
    if os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path, map_location="cpu")
        msg = model.load_state_dict(checkpoint['model'], strict=True)
        print(msg, ckp_path)

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    model.to(device)
    total = sum([param.nelement() for param in model.parameters()])
    print(f"{args.model_name}: Number of parameters: %.5fM" % (total / 1e6))
    validate(data_loader_test, model, valdata='100_guizhou_test_chosen', save_path=args.save_path)
    end_time = time.time()
    # 总耗时（包含数据加载）
    total_inference_time = end_time - start_time
    print(f"{args.model_name}: Total inference time (including data loading): {total_inference_time:.4f} seconds")

import torch
import time

def local2global_prob(local_probs):
    # local_probs = torch.nn.functional.softmax(local_logits, dim=-1)  # [B, 6, 2]
    p0 = local_probs[..., 1]  # [B, 6]：每个肺区0级的概率

    # 2. 计算每个肺区≥1级的概率 s = 1 - p0
    s = 1.0 - p0  # [B, 6]

    # 3. 计算 P(满足数=0)：所有6个肺区都是0级的概率
    prod_p0 = torch.prod(p0, dim=1)  # [B]，沿肺区维度（6）求积

    # 4. 计算 P(满足数=1)：恰好1个肺区≥1级，其余5个为0级的概率
    # 公式：sum( s_i * (prod_p0 / p0_i) )，处理p0_i=0的情况（加eps）
    inv_p0 = 1.0 / (p0 + 1e-8)  # [B, 6]，避免除以0
    term = s * inv_p0 * prod_p0.unsqueeze(1)  # [B, 6]，prod_p0.unsqueeze(1)扩展为[B,1]
    P_k1 = torch.sum(term, dim=1)  # [B]，沿肺区维度求和

    # 5. 规则违反概率：0个或1个肺区满足1级
    V_prob = prod_p0 + P_k1  # [B]
    global_output=torch.cat((1-V_prob.unsqueeze(1), V_prob.unsqueeze(1)), dim=1)
    return global_output

@torch.no_grad()
def validate(data_loader, model,valdata='', save_path=''):
    model.eval()
    data_results = []
    all_outputs = []
    all_targets = []
    for idx, (images, masks, targets, imgname) in enumerate(data_loader):
        result = {}
        result["Image Index"] = imgname[0]
        result["Finding Labels"] = imgname[0].split('_')[0]
        samples = (images * masks).to(device)

        pred,_,_,_ = model(samples)

        prob = torch.nn.functional.softmax(pred, dim=1)
        aa = torch.argmax(pred)
        if aa == 0:
            pred_global_labels = 'JPCLN'
        else:
            pred_global_labels = 'JPCNN'
        result["Pred Labels"] = pred_global_labels
        result["Pred logits"] = '%.4f' % prob[0,aa].cpu().detach().numpy()
        data_results.append(result)
        all_outputs.extend(prob.data.cpu().numpy())
        # targets = torch.nn.functional.one_hot(targets.squeeze(-1).long(), num_classes=2)
        all_targets.extend(targets.cpu().numpy())
    df_data_results = pd.DataFrame(data_results)
    # 合并所有输出和目标
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    adaptive_calculate_metrics(
        all_outputs,
        all_targets, valdata=valdata, save_path=save_path)
    return


@torch.no_grad()
def validate_roi(data_loader, model, valdata='', save_path=''):
    model.eval()
    data_results = []
    data_results_local = []
    data_results_global2local = []

    all_outputs = []
    all_local_outputs = []
    all_local_global_outputs = []
    all_targets = []
    for idx, (images, masks, labels, sub_rois, masks_index, imgname) in enumerate(data_loader):
        # print(idx)
        result = {}
        result_local = {}
        result_global2local = {}
        result["Image Index"] = imgname[0]
        result["Finding Labels"] = imgname[0].split('_')[0]
        result_local["Image Index"] = imgname[0]
        result_local["Finding Labels"] = imgname[0].split('_')[0]
        result_global2local["Image Index"] = imgname[0]
        result_global2local["Finding Labels"] = imgname[0].split('_')[0]
        samples = images[0].to(device)
        for ii in range(len(labels)):
            labels[ii] = labels[ii].to(device)
        targets = labels[0].to(device)
        if isinstance(masks[0], list):
            sample_masks = masks[0]
            for ii in range(len(sample_masks)):
                sample_masks[ii] = sample_masks[ii].to(device)
        else:
            sample_masks = masks[0].to(device)
        subregions_imgs = []
        subregions_masks = []
        subregions_labels = []
        index = 0
        for ii in range(labels[1].shape[0]):
            index = 1
            sub_6imgs = []
            sub_6masks = []
            sub_6labels = []
            for jj in range(6):
                subimg = images[jj + 1]
                submask = masks[jj + 1]
                sublabel = labels[jj + 1]
                sub_6imgs.append(subimg[ii, ...].unsqueeze(0))
                sub_6masks.append(submask[ii, ...].unsqueeze(0))
                sub_6labels.append(sublabel[ii, ...].unsqueeze(0))
            sub_6imgs = torch.cat(sub_6imgs, dim=0).to(device)
            sub_6masks = torch.cat(sub_6masks, dim=0).to(device)
            sub_6labels = torch.cat(sub_6labels, dim=0).to(device)
            subregions_imgs.append(sub_6imgs)
            subregions_masks.append(sub_6masks)
            subregions_labels.append(sub_6labels)
        if index != 0:
            subregions_imgs = torch.cat(subregions_imgs, dim=0).to(device)
            subregions_masks = torch.cat(subregions_masks, dim=0).to(device)
            subregions_labels = torch.cat(subregions_labels, dim=0).to(device)
        # subregions_labels_2cls = subregions_labels.clone()
        # subregions_labels_2cls[(subregions_labels_2cls == 0) | (subregions_labels_2cls == 1) | (
        #         subregions_labels_2cls == 2)] = 0
        # subregions_labels_2cls[(subregions_labels_2cls == 3) | (subregions_labels_2cls == 4)] = 1

        all_rois = []
        sub_rois = torch.stack(sub_rois)
        for bb in range(sub_rois.shape[1]):
            for kk in range(sub_rois.shape[0]):
                sub_bb_roi = sub_rois[kk, bb, :]
                roi_with_batch_idx = torch.cat([torch.tensor([bb]).float(), sub_bb_roi])
                all_rois.append(roi_with_batch_idx)
        samples = (samples * sample_masks[0]).to(device)
        pred, _, _,_ = model(samples)
        if args.model_name == "dinov2":
            pred = pred[1]

        prob = torch.nn.functional.softmax(pred, dim=1)
        aa = torch.argmax(pred)
        if aa == 0:
            pred_global_labels = 'Sick'
        else:
            pred_global_labels = 'Health'
        result["Pred Labels"] = pred_global_labels
        result["Pred logits"] = '%.4f' % prob[0, aa].cpu().detach().numpy()

        data_results.append(result)

        # 收集输出和目标
        all_outputs.extend(prob.data.cpu().numpy())
        targets = torch.nn.functional.one_hot(targets.squeeze(-1).long(), num_classes=2)

        all_targets.extend(targets.cpu().numpy())
        subregions_outputs, subregions_outputs_sick, subregions_outputs_health, _  = model(
            subregions_imgs * subregions_masks)
        subregions_outputs = torch.nn.functional.softmax(subregions_outputs, dim=1)

        subregions_outputs2 = subregions_outputs.view(samples.shape[0], 6, -1)
        global_prob = local2global_prob(subregions_outputs2)
        all_local_outputs.extend(global_prob.data.cpu().numpy())
        all_local_global_outputs.extend(((global_prob + prob) / 2).data.cpu().numpy())

        aa = torch.argmax(global_prob)
        if aa == 0:
            pred_local_labels = 'Sick'
        else:
            pred_local_labels = 'Health'
        result_local["Pred Labels"] = pred_local_labels
        result_local["Pred logits"] = '%.4f' % global_prob[0, aa].cpu().detach().numpy()
        result_global2local["Pred logits"] = '%.4f' % ((global_prob + prob) / 2)[0, aa].cpu().detach().numpy()

        aa = torch.argmax((global_prob + prob) / 2)
        if aa == 0:
            pred_global2local_labels = 'Sick'
        else:
            pred_global2local_labels = 'Health'
        result_global2local["Pred Labels"] = pred_global2local_labels

        data_results_local.append(result_local)
        data_results_global2local.append(result_global2local)
    # # 合并所有输出和目标
    all_outputs = np.array(all_outputs)
    all_local_outputs = np.array(all_local_outputs)
    all_local_global_outputs = np.array(all_local_global_outputs)
    all_targets = np.array(all_targets)
    adaptive_calculate_metrics(
        all_outputs,
        all_targets, valdata=valdata+'_global', save_path=save_path
    )
    adaptive_calculate_metrics(
        all_local_outputs,
        all_targets, valdata=valdata+'_local', save_path=save_path
    )
    adaptive_calculate_metrics(
        all_local_global_outputs,
        all_targets,valdata=valdata+'_globallocal', save_path=save_path
        )
    df_data_results_global2local = pd.DataFrame(data_results_global2local)

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
def adaptive_calculate_metrics(all_outputs, all_targets_onehot, save_fpr_tpr=True, threshold=0.5, valdata='', plot_roc=True, save_path=''):
    # # 将输出通过 Sigmoid 激活函数转换为概率
    probabilities = all_outputs
    all_targets = all_targets_onehot
    # 初始化存储各项指标的列表
    auc_scores = []
    optimal_thresholds = []
    acc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    sensitivity_scores = []
    specificity_scores = []

    for i in range(1):
        # 计算 ROC 曲线
        fpr, tpr, thresholds = roc_curve(all_targets[:, i], probabilities[:, i])
        if threshold == None:
            # mccs, threshold = compute_mccs(all_targets, probabilities)
            # print(threshold)
            optimal_idx = np.argmax(tpr - fpr)
            threshold = thresholds[optimal_idx]
            # print(optimal_threshold)

        auc = roc_auc_score(all_targets[:, i], probabilities[:, i])
        if plot_roc:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'AUC = {auc * 100:.2f}' + '%')
            plt.plot([0, 1], [0, 1], 'k--')
            # 找到最接近特定 threshold 的索引
            idx = np.argmin(np.abs(thresholds - threshold))
            # 获取对应的 fpr 和 tpr 值
            specific_fpr = fpr[idx]
            specific_tpr = tpr[idx]
            # 使用 plt.scatter 标记特定点
            plt.scatter(specific_fpr, specific_tpr, marker='*', color='red', s=200, zorder=10)

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            # plt.legend(loc='lower right')
            plt.legend(loc='lower right', prop={'size': 20})
            save_path_fig = save_path + '/' + valdata + '_roc_map.pdf'
            print(save_path_fig)
            plt.savefig(save_path_fig, dpi=500, bbox_inches='tight')  # 保存图像，设置分辨率和裁剪边距
            # # 显示
            # plt.show()
            # 自动关闭图像
            plt.close()

        auc_scores.append(auc)

        # 使用最优阈值计算预测值
        y_pred = (probabilities[:, i] >= threshold).astype(int)
        # y_pred = (probabilities[:, i] >= 0.5).astype(int)
        # 计算各项指标
        acc_scores.append(accuracy_score(all_targets[:, i], y_pred))

        f1_scores.append(f1_score(all_targets[:, i], y_pred))
        precision_scores.append(precision_score(all_targets[:, i], y_pred))
        recall_scores.append(recall_score(all_targets[:, i], y_pred))
        # 计算 Sensitivity 和 Specificity
        tn, fp, fn, tp = confusion_matrix(all_targets[:, i], y_pred).ravel()
        sensitivity_scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificity_scores.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    if save_fpr_tpr:
        # 在 fpr 最后一个数后接上 specific_fpr
        fpr = np.append(fpr, specific_fpr)
        tpr = np.append(tpr, specific_tpr)
        np.save(save_path + '/' + valdata + '_fpr.npy', fpr)
        np.save(save_path + '/' + valdata + '_tpr.npy', tpr)
    # 计算平均值
    auc_scores = np.array(auc_scores)
    auc_scores = np.mean(auc_scores[~np.isnan(auc_scores)])  # 计算有效 AUC 的平均值
    acc = np.mean(acc_scores)
    f1 = np.mean(f1_scores)
    precision = np.mean(precision_scores)
    recall = np.mean(recall_scores)
    sensitivity = np.mean(sensitivity_scores)
    specificity = np.mean(specificity_scores)

    avg=(auc_scores+acc+sensitivity+specificity)/4

    print('shanxitest sensitivity:', sensitivity, ' specificity:', specificity, ' acc:', acc, 'auc:', auc_scores, '  avg:', avg)


if __name__ == '__main__':
    import gc
    import pandas as pd
    torch.set_num_threads(3)
    # model_names = ['DMALNET']
    # model_names = ['mambaout','PKA2_Net', 'pcam','DMALNET', 'our_best', 'MedMamba ']
    model_names = ['our_best']
    for model_name in model_names:
        args = parse_option()
        args.batch_size = 1
        args.model_name = model_name
        config = get_config(args)
        main(config)

