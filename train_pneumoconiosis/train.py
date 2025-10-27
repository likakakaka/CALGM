import time
import json
import argparse
import datetime
import torch.backends.cudnn as cudnn
from timm.utils import accuracy, AverageMeter
from utils.configv2 import get_config
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
import numpy as np
from utils.datasets_pneum import *

print(f"||{torch.multiprocessing.get_start_method()}||", end="")
torch.multiprocessing.set_start_method("spawn", force=True)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 将CuDNN的性能优化模式设置为关闭
    cudnn.benchmark = False
    # 将CuDNN的确定性模式设置为启用,确保CuDNN在相同的输入下生成相同的输出
    cudnn.deterministic = True
    # CuDNN加速
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
    parser.add_argument('--output',
                        default='./OUTPUT',
                        type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', default=True, action='store_true', help='Perform evaluation only')

    parser.add_argument('--cam_weight', type=float, default=0.5, help='')
    parser.add_argument('--cam_align_weight', type=float, default=0.5, help='')
    parser.add_argument('--subrank_weight', type=float, default=0.3, help='')


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

    parser.add_argument('--data_root', type=str, default='/dataset/pneumoconiosis_path',
                        help='The path of dataset')
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


class GaussianBlur2(object):
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
        d_transform.RandomResizedCrop(1024, scale=(0.75, 1.), interpolation=Image.BICUBIC),
        d_transform.RandomHorizontalFlip(p=0.5),
        d_transform.RandomRotation(degrees=(-15, 15)),
        d_transform.RandomAutocontrast(p=0.3),
        d_transform.RandomEqualize(p=0.3),
        GaussianBlur2(0.3),
    ])
    global_transfo2_subregions = d_transform.Compose([
        d_transform.Resize((256, 256)),
        d_transform.RandomAutocontrast(p=0.3),
        d_transform.RandomEqualize(p=0.3),
        GaussianBlur2(0.3),
        d_transform.ToTensor(),
        d_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    pil2tensor_transfo = d_transform.Compose([d_transform.Resize((512, 512)), d_transform.ToTensor(),
                                             d_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                             ])
    global_transfo = d_transform.Compose([
        d_transform.Resize((1024, 1024)),
    ])
    global_transfo_subregions = d_transform.Compose([
        d_transform.Resize((256, 256)),
        d_transform.ToTensor(),
        d_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform = d_transform.Compose([
        d_transform.Resize((512, 512)),
        d_transform.ToTensor(),
        d_transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = Shanxi_w7masks_5Subregions_wsubroi_Mixednew_Dataset(args.data_root + 'seg_rec_img_1024',
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
                                                              data_transform=global_transfo,
                                                              pil2tensor_transform=pil2tensor_transfo,
                                                              data_subregions_transform=global_transfo_subregions,
                                                              sub_img_size=256, )
    dataset_test = Shanxi_w7masks_5Subregions_wsubroi_Dataset(args.data_root + 'seg_rec_img_1024',
                                                              args.data_root + 'seg_rec_mask_1024',
                                                              txtpath=args.data_root + 'test2.txt',
                                                              csvpath=args.data_root + 'subregions_label_shanxi_all.xlsx',
                                                              data_transform=global_transfo2,
                                                              pil2tensor_transform=pil2tensor_transfo,
                                                              data_subregions_transform=global_transfo2_subregions,
                                                              sub_img_size=256, )
    dataset_test2 = Shanxi_w7masks_5Subregions_wsubroi_Dataset(
        '/dataset/guizhoudataset/seg_rec_img_1024',
        '/dataset/guizhoudataset/seg_rec_mask_1024',
        txtpath='/dataset/guizhoudataset/guizhou_one.txt',
        csvpath='/dataset/guizhoudataset/subregions_guizhou_all.xlsx',
        data_transform=global_transfo2,
        pil2tensor_transform=pil2tensor_transfo,
        data_subregions_transform=global_transfo2_subregions,
        sub_img_size=256, )


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    data_loader_test2 = torch.utils.data.DataLoader(
        dataset_test2,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    return data_loader_train, data_loader_val, data_loader_test, data_loader_test2
def save_rng_states():
    # 保存CPU和GPU的RNG状态
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    return cpu_state, cuda_state

def restore_rng_states(cpu_state, cuda_state):
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_state)


from timm.utils import ModelEma as ModelEma
from calgm_model import *
from utils.cloc import *
from utils.utils import *


def main(config, ptname='model_ema_best_auc_shanxi_val'):
    data_loader_train, data_loader_val, data_loader_test, data_loader_test2 = build_phe_loader(args)
    logger.info(f"Creating CALGM model!")

    model = calgm(pretrained=True, drop_path_rate=0)
    model.head.fc = nn.Linear(768, 2)
    learnable_map = [
            ['fixed', 0.4],
            ['fixed', 0.4],
            ['fixed', 0.4],
            ['fixed', 0.4],
        ]
    model.margin_criterion = OrdinalContrastiveLoss_mm(
            n_classes=5,
            device=device,
            learnable_map=learnable_map
        )

    model.to(device)
    model_ema = ModelEma(
        model,
        decay=args.model_ema_decay,
        device='cpu' if args.model_ema_force_cpu else '',
        resume='')
    print("Using EMA with decay = %.8f" % args.model_ema_decay)

    ckp_path = os.path.join(config.OUTPUT, ptname + '.pth')

    optimizer = build_optimizer(config, model)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    max_acc_avg = 0.0
    max_auc_avg = 0.0
    max_acc_ema_avg = 0.0
    max_auc_ema_avg = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    logger.info("Start training")
    start_time = time.time()
    log_path = os.path.join(config.OUTPUT, 'logs')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)

    acc_ema_best_epoch = 0
    acc_best_epoch = 0
    auc_ema_best_epoch = 0
    auc_best_epoch = 0

    for epoch in range(config.TRAIN.START_EPOCH, 50):
        writer = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn=None,
                                 lr_scheduler=lr_scheduler, loss_scaler=loss_scaler, model_ema=model_ema, writer=writer)
        # 保存当前RNG状态
        cpu_state, cuda_state = save_rng_states()

        acc, _, _, auc, writer = validate(data_loader_val, model, epoch, writer,
                                              valdata='shanxi_val')
        acc_ema, _, _, auc_ema, writer = validate(data_loader_val, model_ema.ema, epoch, writer,
                                                      valdata='shanxi_val_ema')
       
        # val_avg = test_acc2_ema
        # logger.info(f"Task average of the network on the val images: {val_avg:.4f}")
        if acc > max_acc_avg:
            acc_best_epoch = epoch
            max_acc_avg = max(max_acc_avg, acc)
            save_test_checkpoint(config, epoch, model, max_acc_avg, optimizer, lr_scheduler,
                                 loss_scaler,
                                 logger, ptname='model_best_acc_shanxi_val.pth')
        else:
            max_acc_avg = max(max_acc_avg, acc)

        if acc_ema > max_acc_ema_avg:
            acc_ema_best_epoch = epoch
            max_acc_ema_avg = max(max_acc_ema_avg, acc_ema)
            save_test_checkpoint(config, epoch, model_ema.ema, max_acc_ema_avg, optimizer, lr_scheduler,
                                 loss_scaler,
                                 logger, ptname='model_ema_best_acc_shanxi_val.pth')
        else:
            max_acc_ema_avg = max(max_acc_ema_avg, acc_ema)

        if auc > max_auc_avg:
            auc_best_epoch = epoch
            max_auc_avg = max(max_auc_avg, auc)
            save_test_checkpoint(config, epoch, model, max_auc_avg, optimizer, lr_scheduler,
                                 loss_scaler,
                                 logger, ptname='model_best_auc_shanxi_val.pth')
        else:
            max_auc_avg = max(max_auc_avg, auc)

        if auc_ema > max_auc_ema_avg:
            auc_ema_best_epoch = epoch
            max_auc_ema_avg = max(max_auc_ema_avg, auc_ema)
            save_test_checkpoint(config, epoch, model_ema.ema, max_auc_ema_avg, optimizer, lr_scheduler,
                                 loss_scaler,
                                 logger, ptname='model_ema_best_auc_shanxi_val.pth')
        else:
            max_auc_ema_avg = max(max_auc_ema_avg, auc_ema)
        logger.info(
            f'Best acc epoch: {acc_best_epoch}' + ' Max acc average: ' + "%.4f" % max_acc_avg + f'Best auc epoch: {auc_best_epoch}' +  'Test auc average: ' + "%.4f" % max_auc_avg + f' Best acc_ema epoch: {acc_ema_best_epoch}' + ' Max acc_ema average: ' + "%.4f" % max_acc_ema_avg + f'Best auc_ema epoch: {auc_ema_best_epoch}' +  'Test auc_ema average: ' + "%.4f" % max_auc_ema_avg)
        restore_rng_states(cpu_state, cuda_state)
    ckp_path = os.path.join(config.OUTPUT, ptname + '.pth')
    if os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path, map_location="cpu")
        model.load_state_dict(checkpoint['model'], strict=True)


    shanxi_val_accuracy, shanxi_val_sensitivity, shanxi_val_specificity, shanxi_val_auc, shanxi_val_f1, _ = validate(
        config, data_loader_val, model,
        threshold=0.5, valdata='shanxi_val')
    shanxi_test_accuracy, shanxi_test_sensitivity, shanxi_test_specificity, shanxi_test_auc, shanxi_test_f1, _ = validate(
        config, data_loader_test,
        model, threshold=0.5,
        valdata='shanxi_test')
    guizhou_test_accuracy, guizhou_test_sensitivity, guizhou_test_specificity, guizhou_test_auc, guizhou_test_f1, _ = validate(
        config, data_loader_test2,
        model, threshold=0.5,
        valdata='guizhou_test')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    return shanxi_val_auc, shanxi_val_accuracy, shanxi_val_sensitivity, shanxi_val_specificity, shanxi_test_auc, shanxi_test_accuracy, shanxi_test_sensitivity, shanxi_test_specificity, guizhou_test_auc, guizhou_test_accuracy, guizhou_test_sensitivity, guizhou_test_specificity

class OrdinalMetricLoss(nn.Module):
    def __init__(self, pos_label="none", **kwargs):
        super(OrdinalMetricLoss, self).__init__()
        self.pos_label = pos_label
        self.crit = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.criterion = torch.nn.CrossEntropyLoss()
        self.l1_criterion = nn.SmoothL1Loss(beta=0.05, reduction='mean')
        self.bcecrit = torch.nn.BCEWithLogitsLoss()
        self.loss = torch.nn.BCELoss()

    @staticmethod
    def get_loss_names():
        return ["loss"]

    def forward(self, labelsorg, gradcammap_sick, masksorg, sub_index=False):
        if isinstance(masksorg, list):
            masks = masksorg[0]
        else:
            masks = masksorg
        gradcammap_sick = gradcammap_sick.unsqueeze(1)
        scale_factor=gradcammap_sick.shape[-1]/masks.shape[-1]
        masks=F.interpolate(masks, scale_factor=scale_factor, mode="bicubic")
        masks_new = (masks > 0)
        masks_out = (masks == 0)
        scores_back = (gradcammap_sick * masks_out).sum(dim=-1).sum(dim=-1) / (masks_out.sum(dim=-1).sum(dim=-1) + 1e-5)

        if sub_index:
            loss_diff = torch.mean(torch.max(torch.zeros_like(scores_back), scores_back - 0.1))
            sick_scores_fore = (gradcammap_sick * masks_new).sum(dim=-1).sum(dim=-1) / (
                    masks_new.sum(dim=-1).sum(dim=-1) + 1e-5)
            diff_scores = torch.max(torch.zeros_like(sick_scores_fore),
                                    (1-labelsorg) * (-sick_scores_fore + 0.7)) + torch.max(
                torch.zeros_like(sick_scores_fore), labelsorg * (sick_scores_fore - 0.2))
            loss_contra = diff_scores.mean()
            return loss_diff, loss_contra
        else:
            loss_diff1 = torch.mean(torch.max(torch.zeros_like(scores_back), scores_back - 0.1))
            maskss = []
            labelss = []
            for jj in range(masks.shape[0]):
                maskk = []
                labelkk = []
                for kk in range(6):
                    maskk.append(masksorg[kk + 1][jj, ...])
                    labelkk.append(labelsorg[kk + 1][jj, ...])
                maskk = torch.stack(maskk)
                scale_factor = gradcammap_sick.shape[-1] / maskk.shape[-1]
                maskk = F.interpolate(maskk, scale_factor=scale_factor, mode="bicubic")
                labelkk = torch.stack(labelkk)
                maskss.append(maskk)
                labelss.append(labelkk)
            maskss = torch.stack(maskss)
            labelss = torch.stack(labelss)

            sick_scores_fore = (gradcammap_sick.unsqueeze(1) * maskss).sum(dim=-1).sum(dim=-1) / (
                    maskss.sum(dim=-1).sum(dim=-1) + 1e-5)
            diff_scores = torch.relu((1 - labelss) * (-sick_scores_fore + 0.7)) + torch.relu(
                labelss * (sick_scores_fore - 0.2))
            loss_contra = diff_scores.mean()
            loss_diff = loss_diff1 + loss_contra
            return loss_diff, loss_diff1, loss_contra

class SilicosisStage1RuleLoss(nn.Module):
    def __init__(self, alpha=1.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, local_logits, global_probs):
        """
        local_logits: [B, 6, 2] 局部分支的logits（未经过softmax）
        global_probs: [B, 2] 全局分支的logits（未softmax，对应正常、壹期）
        返回：规则损失（仅针对壹期规则）
        """
        # 1. 计算每个肺区的0级概率 p0（softmax）
        local_logits=local_logits.view(global_probs.shape[0], 6, -1)
        global_probs = torch.nn.functional.softmax(global_probs, dim=-1)
        local_probs = torch.nn.functional.softmax(local_logits, dim=-1)  # [B, 6, 5]
        p0 = local_probs[..., 1]  # [B, 6]：每个肺区0级的概率

        # 2. 计算每个肺区≥1级的概率 s = 1 - p0
        s = 1.0 - p0  # [B, 6]

        # 3. 计算 P(满足数=0)：所有6个肺区都是0级的概率
        prod_p0 = torch.prod(p0, dim=1)  # [B]，沿肺区维度（6）求积

        # 4. 计算 P(满足数=1)：恰好1个肺区≥1级，其余5个为0级的概率
        # 公式：sum( s_i * (prod_p0 / p0_i) )，处理p0_i=0的情况（加eps）
        inv_p0 = 1.0 / (p0 + self.eps)  # [B, 6]，避免除以0
        term = s * inv_p0 * prod_p0.unsqueeze(1)  # [B, 6]，prod_p0.unsqueeze(1)扩展为[B,1]
        P_k1 = torch.sum(term, dim=1)  # [B]，沿肺区维度求和

        # 5. 规则违反概率：0个或1个肺区满足1级
        V_prob = prod_p0 + P_k1  # [B]

        # 6. 全局壹期的预测概率
        normal_prob = global_probs[:, 1]  # 假设索引1对应壹期

        # 7. 规则损失：batch平均后乘以权重
        rule_loss = torch.mean(- (normal_prob * torch.log(V_prob + self.eps)
                                  + (1 - normal_prob) * torch.log(1 - V_prob + self.eps)))
        return rule_loss


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler,
                    model_ema=None, writer=None):
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ruleloss = SilicosisStage1RuleLoss()
    criterion2 = OrdinalMetricLoss()
    loss_meter = AverageMeter()
    loss_cls_meter = AverageMeter()
    loss_rule_meter = AverageMeter()
    loss_sub_cls_meter = AverageMeter()
    loss_cam_align_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    start = time.time()
    end = time.time()
    for idx, (images_all, masks_all, labels_all, sub_rois_all, masks_index_all, imgname) in enumerate(data_loader):
        images = []
        masks = []
        labels = labels_all[0].copy()
        subregions_imgs = []
        subregions_masks = []
        subregions_labels = []
        sample_masks = masks_all[0][0].copy()
        sub_rois = sub_rois_all[0].copy()
        for ip in range(images_all[0][0].shape[0]):
            sampled_indices = torch.randint(0, 3, (1,))
            images.append(images_all[sampled_indices][0][ip, ...].unsqueeze(0))
            masks.append(masks_all[sampled_indices][0][0][ip, ...].unsqueeze(0))
            labels[0][ip, ...] = labels_all[sampled_indices][0][ip, ...]
            sample_masks[0][ip, ...] = masks_all[sampled_indices][0][0][ip, ...]
            sub_6imgs = []
            sub_6masks = []
            sub_6labels = []
            for jj in range(6):
                subimg = images_all[sampled_indices][jj + 1][ip, ...]
                submask = masks_all[sampled_indices][jj + 1][ip, ...]
                sublabel = labels_all[sampled_indices][jj + 1][ip, ...]
                sub_6imgs.append(subimg.unsqueeze(0))
                sub_6masks.append(submask.unsqueeze(0))
                sub_6labels.append(sublabel.unsqueeze(0))
                sub_rois[jj][ip, ...] = sub_rois_all[sampled_indices][jj][ip, ...]
                sample_masks[jj + 1][ip, ...] = masks_all[sampled_indices][0][jj + 1][ip, ...]
                labels[jj + 1][ip, ...] = labels_all[sampled_indices][jj + 1][ip, ...]
            sub_6imgs = torch.cat(sub_6imgs, dim=0).to(device)
            sub_6masks = torch.cat(sub_6masks, dim=0).to(device)
            sub_6labels = torch.cat(sub_6labels, dim=0).to(device)
            subregions_imgs.append(sub_6imgs)
            subregions_masks.append(sub_6masks)
            subregions_labels.append(sub_6labels)

        samples = torch.cat(images).to(device)
        subregions_imgs = torch.cat(subregions_imgs).to(device)
        subregions_masks = torch.cat(subregions_masks).to(device)
        subregions_labels = torch.cat(subregions_labels).to(device)
        all_rois = []
        sub_rois = torch.stack(sub_rois)
        for bb in range(sub_rois.shape[1]):
            for kk in range(sub_rois.shape[0]):
                sub_bb_roi = sub_rois[kk, bb, :]
                roi_with_batch_idx = torch.cat([torch.tensor([bb]).float(), sub_bb_roi])
                all_rois.append(roi_with_batch_idx)
        all_rois = torch.stack(all_rois).to(device)
        for ii in range(len(sample_masks)):
            sample_masks[ii] = sample_masks[ii].to(device)
            labels[ii] = labels[ii].to(device)

        samples = (samples * sample_masks[0]).to(device)
        targets = labels[0].to(device)
        targets = targets.squeeze(-1).long().to(device)
        data_time.update(time.time() - end)

        outputs, _, _, fea = model(samples)
        loss_cls = criterion(outputs, targets)
        weights = model.head.fc.weight

        # 归一化和调整大小
        selected_weights_sick = weights[0]  # 形状：(batch_size, 2048)
        # 计算 CAM
        global_cam_sick = torch.matmul(fea.permute(0, 2, 3, 1), selected_weights_sick.unsqueeze(-1)).squeeze(-1)
        global_cam_sick = torch.relu(global_cam_sick.squeeze(-1))  # 应用 ReLU（可选）

        for kp in range(6):
            label = labels[kp+1].clone()
            label[(label == 0) | (label == 1) | (
                    label == 2)] = 0
            label[(label == 3) | (label == 4)] = 1
            labels[kp + 1]=label

        cam_loss, loss_diff1, loss_contraii = criterion2(labels, global_cam_sick, sample_masks)

        subregions_outputs_0_1, subregions_outputs_sick, subregions_outputs_health, subregions_fea = model(
            subregions_imgs * subregions_masks)
        subregions_labels_0_1 = subregions_labels.squeeze(-1).clone()
        subregions_labels_0_1[(subregions_labels_0_1 == 0) | (subregions_labels_0_1 == 1) | (
                subregions_labels_0_1 == 2)] = 0
        subregions_labels_0_1[(subregions_labels_0_1 == 3) | (subregions_labels_0_1 == 4)] = 1
        loss_subregions_2cls = criterion(subregions_outputs_0_1, subregions_labels_0_1.long().to(device))

        subregions_outputs_0_1_softmax = nn.Softmax(dim=-1)(subregions_outputs_0_1)


        outputs_sick3_softmax = nn.Softmax(dim=-1)(subregions_outputs_sick) * subregions_outputs_0_1_softmax[:,
                                                                              0].unsqueeze(-1)
        outputs_health2_softmax = nn.Softmax(dim=-1)(subregions_outputs_health) * subregions_outputs_0_1_softmax[:,
                                                                                  1].unsqueeze(-1)
        subregions_outputs = torch.cat((outputs_sick3_softmax, outputs_health2_softmax), dim=-1)
        log_output = torch.log(subregions_outputs + 1e-8)  # 对softmax结果取log
        # 真实标签用类别索引形式，如 [1, 2]
        loss_subregions_5cls = nn.functional.nll_loss(log_output, subregions_labels.squeeze(-1).long())
        sub_feas = model.neck(subregions_fea).flatten(start_dim=1, end_dim=-1)
        sub_rank_clsloss = model.margin_criterion(sub_feas,
                                                  subregions_labels.squeeze(-1).long(),
                                                  step=epoch * num_steps + idx)
        loss_rule = ruleloss(subregions_outputs_0_1, outputs)
        weights = model.head.fc.weight
        selected_weights_sick = weights[0]  # 形状：(batch_size, 2048)
        # 计算 CAM
        cam_sick = torch.matmul(subregions_fea.permute(0, 2, 3, 1), selected_weights_sick.unsqueeze(-1)).squeeze(-1)
        cam_sick = torch.relu(cam_sick.squeeze(-1))  # 应用 ReLU（可选）
        subregions_cam_loss, subregions_cam_contra_loss = criterion2(subregions_labels_0_1, cam_sick,
                                                                     subregions_masks,
                                                                     sub_index=True)

        local_cam_sick = cam_sick.view(targets.shape[0], 6, cam_sick.shape[-1], -1)
        size = (local_cam_sick.shape[-1], local_cam_sick.shape[-1])
        pooled = roi_align(
            global_cam_sick.unsqueeze(1),
            all_rois,
            output_size=size,
            spatial_scale=1 / 32  # 需与坐标转换一致
        ).view(local_cam_sick.shape[0], 6, -1)
        # 将特征图展平为 [batch, channel, h*w]
        local_cam_sick = local_cam_sick.view(local_cam_sick.shape[0], 6, -1)

        loss_cam_align = F.l1_loss(pooled, local_cam_sick)


        loss = loss_cls + loss_subregions_2cls + loss_subregions_5cls + \
               args.cam_weight * (
                       cam_loss + subregions_cam_loss + subregions_cam_contra_loss) + args.cam_align_weight * (
                   loss_cam_align) + args.subrank_weight * sub_rank_clsloss

        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        # torch.cuda.synchronize()
        loss_cam_align_meter.update(loss_cam_align.item(), targets.size(0))
        loss_meter.update(loss.item(), targets.size(0))
        loss_cls_meter.update(loss_cls.item(), targets.size(0))
        loss_sub_cls_meter.update(loss_subregions_2cls.item(), targets.size(0))
        loss_rule_meter.update(loss_rule.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'loss_cls {loss_cls_meter.val:.4f} ({loss_cls_meter.avg:.4f})\t'
                f'loss_sub_cls {loss_sub_cls_meter.val:.4f} ({loss_sub_cls_meter.avg:.4f})\t'
                f'loss_rule_cls {loss_rule_meter.val:.4f} ({loss_rule_meter.avg:.4f})\t'
                f'loss_cam_align {loss_cam_align_meter.val:.4f} ({loss_cam_align_meter.avg:.4f})\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
            )
            writer.add_scalar('Shanxitrain cls loss', loss_cls.item(), epoch * num_steps + idx)
            writer.add_scalar('Shanxitrain subregions 2cls loss', loss_subregions_2cls.item(), epoch * num_steps + idx)
            writer.add_scalar('Shanxitrain subregions 5cls loss', loss_subregions_5cls.item(), epoch * num_steps + idx)
            writer.add_scalar('Shanxitrain rule loss', loss_rule.item(), epoch * num_steps + idx)
            writer.add_scalar('Shanxitrain cam_align loss', loss_cam_align.item(), epoch * num_steps + idx)
            writer.add_scalar('Shanxitrain cam_back_loss', loss_diff1.item(), epoch * num_steps + idx)
            writer.add_scalar('Shanxitrain cam_contra_sick_health_loss', loss_contraii.item(), epoch * num_steps + idx)
            writer.add_scalar('Shanxitrain subregions_cam_back_loss', subregions_cam_loss.item(),
                              epoch * num_steps + idx)
            writer.add_scalar('Shanxitrain subregions_cam_contra_loss', subregions_cam_contra_loss.item(),
                              epoch * num_steps + idx)
            writer.add_scalar('Shanxitrain subregions_cloc_rank_loss', sub_rank_clsloss.item(),
                              epoch * num_steps + idx)

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return writer


import torch





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
def calculate_metrics(all_outputs, all_targets, num_classes=4):
    # 将输出通过 Softmax 激活函数转换为概率
    probabilities = all_outputs.cpu().numpy()
    predictions = np.argmax(probabilities, axis=1)  # 选择概率最大的类别作为预测结果
    # 将 one-hot 编码的目标转换为类别索引
    all_targets_onehot = torch.nn.functional.one_hot(all_targets.long(), num_classes=num_classes)
    all_targets = all_targets.cpu().numpy()
    # 计算准确率
    accuracy = accuracy_score(all_targets, predictions)

    # 计算 F1 分数、精确率和召回率
    f1 = f1_score(all_targets, predictions, average='weighted')
    precision = precision_score(all_targets, predictions, average='weighted')
    recall = recall_score(all_targets, predictions, average='weighted')

    # 计算每个类别的敏感性和特异性
    sensitivity = []
    specificity = []
    auc_scores=[]
    class_accuracies=[]
    for i in range(num_classes):
        # 为每个类别创建二分类标签
        binary_predictions = (predictions == i).astype(int)
        binary_targets = (all_targets == i).astype(int)
        preds=binary_predictions[binary_targets==1]
        cc = preds.sum()
        dd = binary_targets.sum()
        class_accuracy = cc / dd
        # 计算每个类别的准确率
        class_accuracies.append(class_accuracy)

        auc = roc_auc_score(all_targets_onehot.cpu().numpy()[:, i], probabilities[:, i])
        auc_scores.append(auc)
        true_positive = ((predictions == i) & (all_targets == i)).sum()
        true_negative = ((predictions != i) & (all_targets != i)).sum()
        false_positive = ((predictions == i) & (all_targets != i)).sum()
        false_negative = ((predictions != i) & (all_targets == i)).sum()

        sensitivity.append(
            true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0)
        specificity.append(
            true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0)
    auc_scores = np.asarray(auc_scores)
    auc_scores = np.mean(auc_scores[~np.isnan(auc_scores)])
    sensitivity = np.asarray(sensitivity)
    sensitivity = np.mean(sensitivity[~np.isnan(sensitivity)])
    specificity = np.asarray(specificity)
    specificity = np.mean(specificity[~np.isnan(specificity)])
    # 计算每个类别的平均准确率
    class_accuracies = np.asarray(class_accuracies)
    avg_class_accuracies = np.mean(class_accuracies[~np.isnan(class_accuracies)])
    return avg_class_accuracies*100,class_accuracies*100,accuracy* 100, auc_scores* 100, sensitivity* 100, specificity* 100, f1* 100, precision* 100, recall* 100


@torch.no_grad()
def validate(data_loader, model, epoch, writer=None, valdata='shanxi_val'):
    model.eval()

    batch_time = AverageMeter()

    end = time.time()
    all_outputs = []
    all_local_outputs = []
    all_local_global_outputs = []
    all_targets = []
    all_5local_outputs = []
    all_5targets = []
    all_2local_outputs = []
    all_2targets = []
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
        subregions_labels_2cls = subregions_labels.clone()
        subregions_labels_2cls[(subregions_labels_2cls == 0) | (subregions_labels_2cls == 1) | (
                subregions_labels_2cls == 2)] = 0
        subregions_labels_2cls[(subregions_labels_2cls == 3) | (subregions_labels_2cls == 4)] = 1

        all_rois = []
        sub_rois = torch.stack(sub_rois)
        for bb in range(sub_rois.shape[1]):
            for kk in range(sub_rois.shape[0]):
                sub_bb_roi = sub_rois[kk, bb, :]
                roi_with_batch_idx = torch.cat([torch.tensor([bb]).float(), sub_bb_roi])
                all_rois.append(roi_with_batch_idx)
        samples = (samples * sample_masks[0]).to(device)
        pred, _, _, _ = model(samples)
        prob = torch.nn.functional.softmax(pred, dim=1)
        # 收集输出和目标
        all_outputs.extend(prob.data.cpu().numpy())
        targets = torch.nn.functional.one_hot(targets.squeeze(-1).long(), num_classes=2)

        all_targets.extend(targets.cpu().numpy())
        subregions_outputs, subregions_outputs_sick, subregions_outputs_health, _ = model(
            subregions_imgs * subregions_masks)
        subregions_outputs = torch.nn.functional.softmax(subregions_outputs, dim=1)
        # all_output = nn.Softmax(dim=-1)(subregions_outputs)
        outputs_sick3_softmax = nn.Softmax(dim=-1)(subregions_outputs_sick) * subregions_outputs[:,
                                                                              0].unsqueeze(-1)
        outputs_health2_softmax = nn.Softmax(dim=-1)(subregions_outputs_health) * subregions_outputs[:,
                                                                                  1].unsqueeze(-1)
        prob5 = torch.cat((outputs_sick3_softmax, outputs_health2_softmax), dim=-1)

        all_5local_outputs.append(prob5)
        all_5targets.append(subregions_labels.squeeze(-1))
        all_2local_outputs.append(subregions_outputs)
        all_2targets.append(subregions_labels_2cls.squeeze(-1))

        subregions_outputs2 = subregions_outputs.view(samples.shape[0], 6, -1)
        global_prob = local2global_prob(subregions_outputs2)
        all_local_outputs.extend(global_prob.data.cpu().numpy())
        all_local_global_outputs.extend(((global_prob + prob) / 2).data.cpu().numpy())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # 合并所有输出和目标
    all_2local_outputs = torch.cat(all_2local_outputs)
    all_2targets = torch.cat(all_2targets)
    all_5local_outputs = torch.cat(all_5local_outputs)
    all_5targets = torch.cat(all_5targets)
    avg_class_accuracies, class_accuracies, accuracy, auc, sensitivity, specificity, f1, precision, recall = calculate_metrics(
        all_2local_outputs, all_2targets, num_classes=2)
    if writer!=None:
        logger.info(
            f'2clsAccuracy: ' + "%.4f" % accuracy + ' acc0: ' + "%.4f" %
            class_accuracies[0] + ' acc1: ' + "%.4f" % class_accuracies[
                1] + ' AUC: ' + "%.4f" % auc + ' Sensitivity: ' + "%.4f" % sensitivity + ' Specificity: ' + "%.4f" % specificity)
        writer.add_scalar(valdata + ' local_sub2 acc', accuracy, epoch)
        writer.add_scalar(valdata + ' local_sub2 acc0', class_accuracies[0], epoch)
        writer.add_scalar(valdata + ' local_sub2 acc1', class_accuracies[1], epoch)


    avg_class_accuracies5, class_accuracies5, accuracy5, auc5, sensitivity5, specificity5, f15, precision5, recall5 = calculate_metrics(
        all_5local_outputs, all_5targets, num_classes=5)
    if writer!=None:
        logger.info(
            f'5clsAccuracy: ' + "%.4f" % accuracy5 + ' acc0: ' + "%.4f" %
            class_accuracies5[0] + ' acc1: ' + "%.4f" % class_accuracies5[
                1] + ' acc2: ' + "%.4f" %
            class_accuracies5[2] + ' acc3: ' + "%.4f" % class_accuracies5[
                3] + ' acc4: ' + "%.4f" %
            class_accuracies5[
                4] + ' AUC: ' + "%.4f" % auc5 + ' Sensitivity: ' + "%.4f" % sensitivity5 + ' Specificity: ' + "%.4f" % specificity5 + '\n')
        writer.add_scalar(valdata + ' local_sub5 acc', accuracy5, epoch)
        writer.add_scalar(valdata + ' local_sub5 acc0', class_accuracies5[0], epoch)
        writer.add_scalar(valdata + ' local_sub5 acc1', class_accuracies5[1], epoch)
        writer.add_scalar(valdata + ' local_sub5 acc2', class_accuracies5[2], epoch)
        writer.add_scalar(valdata + ' local_sub5 acc3', class_accuracies5[3], epoch)
        writer.add_scalar(valdata + ' local_sub5 acc4', class_accuracies5[4], epoch)

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
        f'{valdata} epoch: {epoch} ' + ' Global_results Average: ' + "%.4f" % avg + ' Accuracy: ' + "%.4f" % acc + ' AUC.5: ' + "%.4f" % auc + ' Sensitivity: ' + "%.4f" % sensitivity + ' Specificity.5: ' + "%.4f" % specificity)

    acc_local, auc_local, sensitivity_local, specificity_local, f1_local, precision_local, recall_local, threshold_mcc = adaptive_calculate_metrics(
        all_local_outputs,
        all_targets,
        threshold=0.5,
        valdata=valdata,
        plot_roc=True,
        save_fpr_tpr=True)
    avg_local = (acc_local + auc_local + specificity_local + sensitivity_local) / 4
    if writer != None:
        logger.info(
        f'{valdata} epoch: {epoch} ' + ' Local2_results Average: ' + "%.4f" % avg_local + ' Accuracy: ' + "%.4f" % acc_local + ' AUC.5: ' + "%.4f" % auc_local + ' Sensitivity: ' + "%.4f" % sensitivity_local + ' Specificity.5: ' + "%.4f" % specificity_local)

    acc_local_global, auc_local_global, sensitivity_local_global, specificity_local_global, f1_local_global, precision_local_global, recall_local_global, threshold_mcc = adaptive_calculate_metrics(
        all_local_global_outputs,
        all_targets,
        threshold=0.5,
        valdata=valdata,
        plot_roc=True,
        save_fpr_tpr=True)
    avg_local_global = (acc_local_global + auc_local_global + specificity_local_global + sensitivity_local_global) / 4
    if writer != None:
        logger.info(
        f'{valdata} epoch: {epoch} '+ ' Local2_global_results Average: ' + "%.4f" % avg_local_global + ' Accuracy: ' + "%.4f" % acc_local_global + ' AUC.5: ' + "%.4f" % auc_local_global + ' Sensitivity: ' + "%.4f" % sensitivity_local_global + ' Specificity.5: ' + "%.4f" % specificity_local_global)

    if writer != None:
        writer.add_scalar(valdata + ' global avg', avg, epoch)
        writer.add_scalar(valdata + ' global auc', auc, epoch)
        writer.add_scalar(valdata + ' global acc_threshold0.5', acc, epoch)
        writer.add_scalar(valdata + ' global sensitivity_threshold0.5', sensitivity, epoch)
        writer.add_scalar(valdata + ' global specificity_threshold0.5', specificity, epoch)

        writer.add_scalar(valdata + ' local avg', avg_local, epoch)
        writer.add_scalar(valdata + ' local auc', auc_local, epoch)
        writer.add_scalar(valdata + ' local acc_threshold0.5', acc_local, epoch)
        writer.add_scalar(valdata + ' local sensitivity_threshold0.5', sensitivity_local, epoch)
        writer.add_scalar(valdata + ' local specificity_threshold0.5', specificity_local, epoch)

        writer.add_scalar(valdata + ' local_global avg', avg_local_global, epoch)
        writer.add_scalar(valdata + ' local_global auc', auc_local_global, epoch)
        writer.add_scalar(valdata + ' local_global acc_threshold0.5', acc_local_global, epoch)
        writer.add_scalar(valdata + ' local_global sensitivity_threshold0.5', sensitivity_local_global, epoch)
        writer.add_scalar(valdata + ' local_global specificity_threshold0.5', specificity_local_global, epoch)

    return acc_local_global, sensitivity_local_global, specificity_local_global, auc_local_global, writer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, roc_curve, \
    confusion_matrix
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
    save_path = config.OUTPUT + '/' + valdata + '_confusion_map.png'
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')  # 保存图像，设置分辨率和裁剪边距



def adaptive_calculate_metrics(all_outputs, all_targets_onehot, threshold=None, valdata='shanxi_test', num_classes=2,
                               plot_roc=True, save_fpr_tpr=True):
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


        auc_scores.append(auc)

        # 使用最优阈值计算预测值
        y_pred = (probabilities[:, i] >= threshold).astype(int)

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
    pt_names = ['model_ema_best_acc_shanxi_val', 'model_ema_best_auc_shanxi_val', 'model_best_auc_shanxi_val', 'model_best_acc_shanxi_val']
    shanxi_test_avgs = []
    shanxi_test_accs = []
    shanxi_test_aucs = []
    shanxi_test_sens = []
    shanxi_test_spec = []

    guizhou_test_avgs = []
    guizhou_test_accs = []
    guizhou_test_aucs = []
    guizhou_test_sens = []
    guizhou_test_spec = []

    val_aucs = []

    # for ii in range(len(seeds)):
    for ii in range(1):
        ii += 1
        avgs = []
        args = parse_option()
        config = get_config(args)
        config.OUTPUT = os.path.join(args.output,
                                     'g512_l256_tree_5cls_2cls_cam_wdatamixup_3',
                                     str(args.cam_weight) + '_cam_' + str(args.cam_align_weight) + '_camalign_' + str(
                                         args.subrank_weight) + '_cloc' + '_lr' + str(
                                         args.base_lr))

        config.freeze()

        os.makedirs(config.OUTPUT, exist_ok=True)
        logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

        # print config
        logger.info(config.dump())
        logger.info(json.dumps(vars(args)))

        for pt_name in pt_names:
            shanxi_val_auc, shanxi_val_acc, shanxi_val_sens, shanxi_val_spec, shanxi_test_auc, shanxi_test_accuracy, shanxi_test_sensitivity, shanxi_test_specificity, guizhou_test_auc, guizhou_test_accuracy, guizhou_test_sensitivity, guizhou_test_specificity = main(
                config, pt_name)
            shanxi_val_avgs = (shanxi_val_auc + shanxi_val_acc + shanxi_val_sens + shanxi_val_spec) / 4

            shanxi_avgs = (
                                  shanxi_test_auc + shanxi_test_accuracy + shanxi_test_sensitivity + shanxi_test_specificity) / 4

            guizhou_avgs = (
                                   guizhou_test_auc + guizhou_test_accuracy + guizhou_test_sensitivity + guizhou_test_specificity) / 4

            # 移除之前添加的处理器
            for handler in logger.handlers[:]:  # 循环遍历处理器的副本
                logger.removeHandler(handler)  # 移除处理器
            gc.collect()
            folder_path = config.OUTPUT.split(config.OUTPUT.split('/')[-1])[0]
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_name = 'result.txt'
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'a', encoding='utf-8') as file:
                re = f'cam weight:{args.cam_weight} cam align weight:{args.cam_align_weight} cloc_rank_weight:{args.subrank_weight} {args.base_lr}: threshold: 0.5, {pt_name}:\n' \
                     f'shanxi_val_avgs: {shanxi_val_avgs}, shanxi_val_auc: {shanxi_val_auc}, shanxi_val_acc: {shanxi_val_acc}, shanxi_val_sensitivity: {shanxi_val_sens}, shanxi_val_specificity: {shanxi_val_spec}\n' \
                     f'shanxi_test_avg: {shanxi_avgs}, shanxi_test_auc: {shanxi_test_auc}, shanxi_test_acc: {shanxi_test_accuracy}, shanxi_test_sensitivity: {shanxi_test_sensitivity}, shanxi_test_specificity: {shanxi_test_specificity}\n' \
                     f'guizhou_test_avg: {guizhou_avgs}, guizhou_test_auc: {guizhou_test_auc}, guizhou_test_acc: {guizhou_test_accuracy}, guizhou_test_sensitivity: {guizhou_test_sensitivity}, guizhou_test_specificity: {guizhou_test_specificity}\n\n'
                file.write(re)
