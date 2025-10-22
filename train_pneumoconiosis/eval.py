
import os
import time
import argparse
import torch.backends.cudnn as cudnn
from timm.utils import accuracy, AverageMeter
from utils.configv2 import get_config
from utils.logger import create_logger
from utils.utils import *
from mmpretrain import get_model
from utils.datasets_pneum import *
print(f"||{torch.multiprocessing.get_start_method()}||", end="")
torch.multiprocessing.set_start_method("spawn", force=True)
import utils.misc as misc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device('cuda:4')
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
    parser.add_argument('--output', default='./OUTPUT', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', default=True, action='store_true', help='Perform evaluation only')

    parser.add_argument('--cam_weight', type=float, default=0.5, help='')
    parser.add_argument('--cam_align_weight', type=float, default=0.5, help='')
    parser.add_argument('--subrank_weight', type=float, default=0.3, help='')

    parser.add_argument('--pretrained_pth',
                        default='./OUTPUT/model_ema_best_acc_shanxi_val.pth',
                        type=str, metavar='PATH',
                        help='root of pretrained model during inference')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    parser.add_argument('--base_lr', type=float, default=8e-5)


    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    parser.add_argument('--data_root', type=str, default='/dataset/pneumoconiosis_path/', help='The path of dataset')
    parser.add_argument('--seed', default=20, type=int)
    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')
    parser.add_argument('--pin_mem', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    args, unparsed = parser.parse_known_args()
    return args
from PIL import ImageFilter, ImageOps
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL save.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img1, img2, mask2=None, sub_rois=None):
        do_it = random.random() <= self.prob
        if not do_it:
            return img1, img2, mask2, sub_rois
        return img1.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        ), img2, mask2, sub_rois
import d_transform
def build_phe_loader(args):
    # 设置随机数种子
    setup_seed(args.seed)
    global_transfo2 = d_transform.Compose([
        d_transform.Resize((1024, 1024)),
    ])
    global_transfo2_subregions = d_transform.Compose([
        d_transform.Resize((256, 256)),
        d_transform.ToTensor(),
        d_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    pil2tensor_transfo = d_transform.Compose([d_transform.Resize((512, 512)), d_transform.ToTensor(),
                                             d_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                             ])

    dataset_train = Shanxi_w7masks_5Subregions_wsubroi_Dataset(args.data_root + 'seg_rec_img_1024',
                                                      args.data_root + 'seg_rec_mask_1024',
                                                      txtpath=args.data_root + 'train.txt',
                                                      csvpath=args.data_root + 'subregions_label_shanxi_all.xlsx',
                                                      data_transform=global_transfo2,
                                                      pil2tensor_transform=pil2tensor_transfo,
                                                      data_subregions_transform=global_transfo2_subregions,
                                                      sub_img_size=256, )
    dataset_val = Shanxi_w7masks_5Subregions_wsubroi_Dataset(args.data_root + 'seg_rec_img_1024',
                                                              args.data_root + 'seg_rec_mask_1024',
                                                              txtpath=args.data_root + 'val.txt',
                                                              csvpath=args.data_root + 'subregions_label_shanxi_all.xlsx',
                                                              data_transform=global_transfo2,
                                                              pil2tensor_transform=pil2tensor_transfo,
                                                              data_subregions_transform=global_transfo2_subregions,
                                                              sub_img_size=256, )
    dataset_test = Shanxi_w7masks_5Subregions_wsubroi_Dataset(args.data_root + 'seg_rec_img_1024',
                                                            args.data_root + 'seg_rec_mask_1024',
                                                            txtpath=args.data_root + 'test2.txt',
                                                            csvpath=args.data_root + 'subregions_label_shanxi_all.xlsx',
                                                            data_transform=global_transfo2,
                                                            pil2tensor_transform=pil2tensor_transfo,
                                                            data_subregions_transform=global_transfo2_subregions,
                                                            sub_img_size=256, )

    dataset_test2 = Shanxi_w7masks_5Subregions_wsubroi_Dataset('/dataset/guizhoudataset/seg_rec_img_1024','/dataset/guizhoudataset/seg_rec_mask_1024',
                                                              txtpath='/dataset/guizhoudataset/guizhou_one.txt',
                                                              csvpath='/ddataset/guizhoudataset/subregions_guizhou_all.xlsx',
                                                              data_transform=global_transfo2,
                                                              pil2tensor_transform=pil2tensor_transfo,
                                                              data_subregions_transform=global_transfo2_subregions,
                                                              sub_img_size=256, )

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank,
                                                        shuffle=True)

    sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=False)
    sampler_test = torch.utils.data.DistributedSampler(dataset_test, num_replicas=num_tasks, rank=global_rank,
                                                       shuffle=False)
    sampler_test2 = torch.utils.data.DistributedSampler(dataset_test2, num_replicas=num_tasks, rank=global_rank,
                                                       shuffle=False)


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_test2 = torch.utils.data.DataLoader(
        dataset_test2, sampler=sampler_test2,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    return data_loader_train, data_loader_val, data_loader_test, data_loader_test2


from timm.utils import ModelEma as ModelEma
from calgm_model import *
def main(config, ptname='model_ema_best_auc_shanxi_val'):
    data_loader_train, data_loader_val, data_loader_test, data_loader_test2 = build_phe_loader(args)
    model = calgm(pretrained=True, drop_path_rate=0)
    model.head.fc = nn.Linear(768, 2)
    model.to(device)
    ckp_path = args.pretrained_pth
    if os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path, map_location="cpu")
        model.load_state_dict(checkpoint['model'], strict=True)
        shanxi_val_accuracy, shanxi_val_sensitivity, shanxi_val_specificity, shanxi_val_auc, shanxi_val_f1, mcc_threshold = validate(
            data_loader_val, model,
            valdata='shanxi_val', ptnames=ptname)
        guizhou_test_accuracy, guizhou_test_sensitivity, guizhou_test_specificity, guizhou_test_auc, guizhou_test_f1, mcc_threshold = validate(
            data_loader_test2, model,
            valdata='guizhou_test', ptnames=ptname)
        shanxi_test_accuracy, shanxi_test_sensitivity, shanxi_test_specificity, shanxi_test_auc, shanxi_test_f1, mcc_threshold = validate(
            data_loader_test, model,
            valdata='shanxi_test', ptnames=ptname)

        shanxi_train_accuracy, shanxi_train_sensitivity, shanxi_train_specificity, shanxi_train_auc, shanxi_train_f1, mcc_threshold = validate(
            data_loader_train, model,
            valdata='shanxi_train',  ptnames= ptname)


import torch

def calculate_auc_metrics(all_outputs, all_targets_onehot):
    # probabilities = torch.softmax(all_outputs, dim=1).cpu().numpy()
    probabilities = torch.sigmoid(all_outputs).cpu().numpy()
    all_targets = all_targets_onehot.cpu().numpy()
    auc = roc_auc_score(all_targets[:, 0], probabilities[:, 0])
    return auc * 100

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
def validate(data_loader, model,valdata='shanxi_val', ptnames=None):
    model.eval()

    batch_time = AverageMeter()


    end = time.time()
    # 初始化输出和目标列表
    all_outputs = []
    all_local_outputs = []
    all_local_outputs5_2 = []
    all_local_global_outputs = []
    all_local_5_2_global_outputs = []
    all_local_5_2_2_global_outputs = []
    all_targets = []
    for idx, (images, masks, labels, sub_rois, masks_index, imgname) in enumerate(data_loader):
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
                # sub_6labels = torch.cat(sub_6labels, dim=0).to(device)
                subregions_imgs.append(sub_6imgs)
                subregions_masks.append(sub_6masks)
                # subregions_labels.append(sub_6labels)
        if index != 0:
            subregions_imgs = torch.cat(subregions_imgs, dim=0).to(device)
            subregions_masks = torch.cat(subregions_masks, dim=0).to(device)

        all_rois = []
        sub_rois = torch.stack(sub_rois)
        for bb in range(sub_rois.shape[1]):
            for kk in range(sub_rois.shape[0]):
                sub_bb_roi = sub_rois[kk, bb, :]
                roi_with_batch_idx = torch.cat([torch.tensor([bb]).float(), sub_bb_roi])
                all_rois.append(roi_with_batch_idx)
        samples = (samples * sample_masks[0]).to(device)
        pred,_,_,_= model(samples)

        prob = torch.nn.functional.softmax(pred, dim=1)
        # 收集输出和目标
        all_outputs.extend(prob.data.cpu().numpy())
        targets = torch.nn.functional.one_hot(targets.squeeze(-1).long(), num_classes=2)

        all_targets.extend(targets.cpu().numpy())

        subregions_outputs,_,_,_ = model(subregions_imgs * subregions_masks)
        subregions_outputs = torch.nn.functional.softmax(subregions_outputs, dim=1)
        subregions_outputs2 = subregions_outputs.view(samples.shape[0], 6, -1)

        global_prob=local2global_prob(subregions_outputs2)
        all_local_outputs.extend(global_prob.data.cpu().numpy())
        all_local_global_outputs.extend(((global_prob+prob)/2).data.cpu().numpy())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # 合并所有输出和目标
    all_outputs = np.array(all_outputs)
    all_local_outputs = np.array(all_local_outputs)
    all_local_global_outputs = np.array(all_local_global_outputs)
    all_targets = np.array(all_targets)
    acc, auc, sensitivity, specificity, f1, precision, recall, threshold_mcc = adaptive_calculate_metrics(
        all_outputs,
        all_targets,
        threshold=0.5,
        valdata=valdata,
        plot_roc=True,
        save_fpr_tpr=True)
    avg = (acc + auc + specificity + sensitivity) / 4
    logger.info(
        f'lr: {args.base_lr} {valdata} {ptnames}' + ' Global_results Average: ' + "%.4f" % avg + ' Accuracy: ' + "%.4f" % acc + ' AUC.5: ' + "%.4f" % auc + ' Sensitivity: ' + "%.4f" % sensitivity + ' Specificity.5: ' + "%.4f" % specificity)

    acc_local, auc_local, sensitivity_local, specificity_local, f1_local, precision_local, recall_local, threshold_mcc = adaptive_calculate_metrics(
        all_local_outputs,
        all_targets,
        threshold=0.5,
        valdata=valdata,
        plot_roc=True,
        save_fpr_tpr=True)
    avg_local = (acc_local + auc_local + specificity_local + sensitivity_local) / 4
    logger.info(
        f'lr: {args.base_lr} {valdata} {ptnames}' + ' Local2_results Average: ' + "%.4f" % avg_local + ' Accuracy: ' + "%.4f" % acc_local + ' AUC.5: ' + "%.4f" % auc_local + ' Sensitivity: ' + "%.4f" % sensitivity_local + ' Specificity.5: ' + "%.4f" % specificity_local)

    acc_local_global, auc_local_global, sensitivity_local_global, specificity_local_global, f1_local_global, precision_local_global, recall_local_global, threshold_mcc = adaptive_calculate_metrics(
        all_local_global_outputs,
        all_targets,
        threshold=0.5,
        valdata=valdata,
        plot_roc=True,
        save_fpr_tpr=True)
    avg_local_global = (acc_local_global + auc_local_global + specificity_local_global + sensitivity_local_global) / 4
    logger.info(
        f'lr: {args.base_lr} {valdata} {ptnames}' + ' Local2_global_results Average: ' + "%.4f" % avg_local_global + ' Accuracy: ' + "%.4f" % acc_local_global + ' AUC.5: ' + "%.4f" % auc_local_global + ' Sensitivity: ' + "%.4f" % sensitivity_local_global + ' Specificity.5: ' + "%.4f" % specificity_local_global)
    return accuracy, sensitivity, specificity, auc, f1, threshold_mcc


from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.5, axis_labels=None, valdata='shanxi_test'):
    # 利用sklearn中的函数生成混淆矩阵并归一化
    cm0 = confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    cm = cm0.astype('float') / cm0.sum(axis=1)[:, np.newaxis]  # 归一化

    class_names = ['Normal', 'Sick']  # 类名称
    # 设置字体大小
    title_font_size = 26  # 标题字体大小
    label_font_size = 24  # 坐标轴标签字体大小
    tick_font_size = 22  # 刻度字体大小
    text_font_size = 22  # 矩阵内文本字体大小
    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=plt.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'), vmin=0, vmax=1)
    # 添加 colorbar
    cbar = plt.colorbar()

    # 设置 colorbar 的字体大小
    cbar.ax.tick_params(labelsize=22)  # 设置 colorbar 刻度字体大小
    # cbar.set_label('Color Scale', fontsize=14)  # 设置 colorbar 标签字体大小

    # plt.colorbar()  # 绘制图例

    # 图像标题
    if title is not None:
        plt.title(title, fontsize=title_font_size)

    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = class_names
    plt.xticks(num_local, axis_labels, rotation=0, fontsize=tick_font_size)  # 将类名称印在x轴坐标上，并倾斜45度
    # plt.yticks(num_local, axis_labels, rotation=90, fontsize=tick_font_size)  # 将类名称印在y轴坐标上
    plt.yticks(num_local, axis_labels, fontsize=tick_font_size)
    labels = plt.gca().get_yticklabels()  # 获取 y 轴标签
    plt.setp(labels, rotation=90, va='center')  # 调整对齐方式

    # plt.ylabel('True label', fontsize=label_font_size)
    # plt.xlabel('Predicted label', fontsize=label_font_size)

    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                count = int(cm[i][j] * cm.sum(axis=1)[i])  # 样本数量
                plt.text(j, i - 0.08, f'{int(cm0[i][j])}',  # 显示样本数量
                         ha="center", va="center", fontsize=text_font_size,
                         color="white" if cm[i][j] > thresh else "black")

                plt.text(j, i + 0.08, format(float(cm[i][j] * 100), '.2f') + '%',
                         ha="center", va="center", fontsize=text_font_size,
                         color="white" if cm[i][j] > thresh else "black")

    # 设置保存路径
    save_path = config.OUTPUT + '/'+valdata+'_confusion_map.png'
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')  # 保存图像，设置分辨率和裁剪边距

    # # 显示图像
    # plt.show()
    #
    # # 自动关闭图像
    # plt.close()
def adaptive_calculate_metrics(all_outputs, all_targets_onehot,threshold=None, valdata='shanxi_test',num_classes=2, plot_roc=True, save_fpr_tpr=True):
    # # 将输出通过 Sigmoid 激活函数转换为概率
    probabilities=all_outputs
    all_targets=all_targets_onehot
    # predictions = (probabilities >= 0.5).astype(int)  # 设定阈值为 0.5
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
            plt.scatter(specific_fpr, specific_tpr, marker='*', color='red', s=200)

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            # plt.legend(loc='lower right')
            plt.legend(loc='lower right', prop={'size': 20})
            save_path = config.OUTPUT + '/' + valdata + '_roc_map.pdf'
            plt.savefig(save_path, dpi=500, bbox_inches='tight')  # 保存图像，设置分辨率和裁剪边距
            # # 显示
            # plt.show()
            # # 自动关闭图像
            # plt.close()

        auc_scores.append(auc)
        # # 找到使得 Sensitivity 和 Specificity 最接近的阈值
        # optimal_idx = np.argmax(tpr - fpr)
        # optimal_threshold = thresholds[optimal_idx]
        # print(optimal_threshold)
        # optimal_thresholds.append(optimal_threshold)
        # if args.valdata == 'shanxi':
        #     optimal_threshold=0.5
        # 使用最优阈值计算预测值
        y_pred = (probabilities[:, i] >= threshold).astype(int)
        # y_pred = (probabilities[:, i] >= 0.5).astype(int)
        # 计算各项指标
        acc_scores.append(accuracy_score(all_targets[:, i], y_pred))
        f1_scores.append(f1_score(all_targets[:, i], y_pred))
        precision_scores.append(precision_score(all_targets[:, i], y_pred))
        recall_scores.append(recall_score(all_targets[:, i], y_pred))

        # 1. 计算混淆矩阵
        plot_matrix(all_targets[:, i], y_pred, [0, 1], title=None, thresh=0.6, axis_labels=None, valdata=valdata)

        # 计算 Sensitivity 和 Specificity
        tn, fp, fn, tp = confusion_matrix(all_targets[:, i], y_pred).ravel()
        sensitivity_scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificity_scores.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    if save_fpr_tpr:
        # 在 fpr 最后一个数后接上 specific_fpr
        fpr = np.append(fpr, specific_fpr)
        tpr = np.append(tpr, specific_tpr)
        np.save(config.OUTPUT + '/' + valdata + '_fpr.npy', fpr)
        np.save(config.OUTPUT + '/' + valdata + '_tpr.npy', tpr)
    # 计算平均值
    auc_scores = np.array(auc_scores)
    auc_scores = np.mean(auc_scores[~np.isnan(auc_scores)])  # 计算有效 AUC 的平均值
    acc = np.mean(acc_scores)
    f1 = np.mean(f1_scores)
    precision = np.mean(precision_scores)
    recall = np.mean(recall_scores)
    sensitivity = np.mean(sensitivity_scores)
    specificity = np.mean(specificity_scores)

    return acc * 100, auc_scores * 100, sensitivity * 100, specificity * 100, f1 * 100, precision * 100, recall * 100, threshold



if __name__ == '__main__':
    import gc
    import pandas as pd
    torch.set_num_threads(3)

    pt_names = ['model_ema_best_acc_shanxi_val', 'model_ema_best_auc_shanxi_val']
shanxi_test_avgs=[]
shanxi_test_accs = []
shanxi_test_aucs=[]
shanxi_test_sens = []
shanxi_test_spec = []

guizhou_test_avgs=[]
guizhou_test_accs = []
guizhou_test_aucs = []
guizhou_test_sens = []
guizhou_test_spec = []

val_aucs=[]

# for ii in range(len(seeds)):
for ii in range(1):
    ii+=1
    avgs = []

    args = parse_option()
    config = get_config(args)
    config.OUTPUT = f'/disk3/wjr/workspace/sec_nejm/temp'

    # config.OUTPUT =args.output
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    folder_path = config.OUTPUT.split(config.OUTPUT.split('/')[-1])[0]
    logger = create_logger(output_dir=folder_path, name=f"{config.MODEL.NAME}",
                           txt_name='local2global_result')

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())

    for pt_name in pt_names:
        main(config, pt_name)


