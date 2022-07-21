import logging
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from utils.evaluate_utils import plot_predictions

from models.spin import spin

affine_par = True


class BasicResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, padding=1, downsample=None):
        super(BasicResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, group=1):
        super(DecoderBlock, self).__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1, groups=group)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            groups=group,
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1, groups=group)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class HourglassModuleMTL(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(HourglassModuleMTL, self).__init__()
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual1(self, block, num_blocks, planes):
        layers = [block(planes * block.expansion, planes) for _ in range(num_blocks)]
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = [self._make_residual1(block, num_blocks, planes) for _ in range(4)]

            if i == 0:
                res.extend(
                    (
                        self._make_residual1(block, num_blocks, planes),
                        self._make_residual1(block, num_blocks, planes),
                    )
                )

            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        rows = x.size(2)
        cols = x.size(3)

        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2, ceil_mode=True)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2_1, low2_2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2_1 = self.hg[n - 1][4](low1)
            low2_2 = self.hg[n - 1][5](low1)
        low3_1 = self.hg[n - 1][2](low2_1)
        low3_2 = self.hg[n - 1][3](low2_2)
        up2_1 = self.upsample(low3_1)
        up2_2 = self.upsample(low3_2)
        out_1 = up1 + up2_1[:, :, :rows, :cols]
        out_2 = up1 + up2_2[:, :, :rows, :cols]

        return out_1, out_2

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


################################################################
############### StackHOurGlassNet with SPIN ################
################################################################
# We added Dual GCN module after the first downsampling layer and only at segmentation branch
# Added Dual GCN at multiple locations (256 x 256 scale) and (128 x 128 scale)
class HourglassNet(nn.Module):
    def __init__(
        self,
        task1_classes=2,
        task2_classes=37,
        block=BasicResnetBlock,
        in_channels=3,
        num_stacks=2,
        num_blocks=1,
        hg_num_blocks=3,
        depth=3,
    ):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, num_blocks)
        self.layer3 = self._make_residual(block, self.num_feats, num_blocks)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # build hourglass modules
        ch = self.num_feats * block.expansion
        hg = []
        res_1, fc_1, score_1, _fc_1, _score_1 = [], [], [], [], []
        res_2, fc_2, score_2, _fc_2, _score_2 = [], [], [], [], []

        for i in range(num_stacks):
            hg.append(HourglassModuleMTL(block, hg_num_blocks, self.num_feats, depth))

            res_1.append(self._make_residual(block, self.num_feats, hg_num_blocks))
            res_2.append(self._make_residual(block, self.num_feats, hg_num_blocks))

            fc_1.append(self._make_fc(ch, ch))
            fc_2.append(self._make_fc(ch, ch))

            score_1.append(nn.Conv2d(ch, task1_classes, kernel_size=1, bias=True))
            score_2.append(nn.Conv2d(ch, task2_classes, kernel_size=1, bias=True))
            if i < num_stacks - 1:
                _fc_1.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                _fc_2.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                _score_1.append(nn.Conv2d(task1_classes, ch, kernel_size=1, bias=True))
                _score_2.append(nn.Conv2d(task2_classes, ch, kernel_size=1, bias=True))

        self.hg = nn.ModuleList(hg)
        self.res_1 = nn.ModuleList(res_1)
        self.fc_1 = nn.ModuleList(fc_1)
        self.score_1 = nn.ModuleList(score_1)
        self._fc_1 = nn.ModuleList(_fc_1)
        self._score_1 = nn.ModuleList(_score_1)

        self.res_2 = nn.ModuleList(res_2)
        self.fc_2 = nn.ModuleList(fc_2)
        self.score_2 = nn.ModuleList(score_2)
        self._fc_2 = nn.ModuleList(_fc_2)
        self._score_2 = nn.ModuleList(_score_2)

        # Final Classifier
        self.decoder1 = DecoderBlock(self.num_feats, self.inplanes)
        self.decoder1_score = nn.Conv2d(self.inplanes, task1_classes, kernel_size=1, bias=True)
        self.finaldeconv1 = nn.ConvTranspose2d(self.inplanes, 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, task1_classes, 2, padding=1)

        # Final Classifier
        self.angle_decoder1 = DecoderBlock(self.num_feats, self.inplanes)
        self.angle_decoder1_score = nn.Conv2d(
            self.inplanes, task2_classes, kernel_size=1, bias=True
        )
        self.angle_finaldeconv1 = nn.ConvTranspose2d(self.inplanes, 32, 3, stride=2)
        self.angle_finalrelu1 = nn.ReLU(inplace=True)
        self.angle_finalconv2 = nn.Conv2d(32, 32, 3)
        self.angle_finalrelu2 = nn.ReLU(inplace=True)
        self.angle_finalconv3 = nn.Conv2d(32, task2_classes, 2, padding=1)

        # Spin module
        self.dgcn_seg_l41 = spin(planes=32, ratio=1)
        self.dgcn_seg_l42 = spin(planes=32, ratio=1)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                )
            )

        layers = [block(self.inplanes, planes, stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        out_1 = []
        out_2 = []

        rows = x.size(2)
        cols = x.size(3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        for i in range(self.num_stacks):
            y1, y2 = self.hg[i](x)
            y1, y2 = self.res_1[i](y1), self.res_2[i](y2)
            y1, y2 = self.fc_1[i](y1), self.fc_2[i](y2)
            score1, score2 = self.score_1[i](y1), self.score_2[i](y2)
            out_1.append(score1[:, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))])
            out_2.append(score2[:, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))])
            if i < self.num_stacks - 1:
                _fc_1, _fc_2 = self._fc_1[i](y1), self._fc_2[i](y2)
                _score_1, _score_2 = self._score_1[i](score1), self._score_2[i](score2)
                x = x + _fc_1 + _score_1 + _fc_2 + _score_2

        # Final Classifications
        d1 = self.decoder1(y1)[
            :, :, : int(math.ceil(rows / 2.0)), : int(math.ceil(cols / 2.0))
        ]  # d1 = 128, 128,128
        d1_score = self.decoder1_score(d1)
        out_1.append(d1_score)
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f2 = self.dgcn_seg_l41(f2)  # Graph reasoning at LEVEL 4 - 257 x 257 - SPIN layer 1
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f4 = self.dgcn_seg_l42(f4) #Graph reasoning at LEVEL 4 - 257 x 257 - SPIN layer 2
        f5 = self.finalconv3(f4)
        out_1.append(f5)

        # Final Classification
        a_d1 = self.angle_decoder1(y2)[
            :, :, : int(math.ceil(rows / 2.0)), : int(math.ceil(cols / 2.0))
        ]
        a_d1_score = self.angle_decoder1_score(a_d1)
        out_2.append(a_d1_score)
        a_f1 = self.angle_finaldeconv1(a_d1)
        a_f2 = self.angle_finalrelu1(a_f1)
        a_f3 = self.angle_finalconv2(a_f2)
        a_f4 = self.angle_finalrelu2(a_f3)
        a_f5 = self.angle_finalconv3(a_f4)
        out_2.append(a_f5)

        return out_1, out_2


def criterion(num_stacks, loss_fn, pred_mask, pred_vec, label, vecmap_angles, epoch, args):
    miou = loss_fn[0]
    ce = loss_fn[1]
    topo = loss_fn[2]

    if not args.multi_scale_pred:
        loss1 = miou(pred_mask, label.to(args.device))
        loss2 = ce(pred_vec, vecmap_angles.to(args.device))
        loss3 = topo(pred_vec, vecmap_angles.to(args.device))
        loss = args.miou_weight * loss1 + args.ce_weight * loss2 + loss3
        return loss, loss1, loss2, loss3

    loss1 = miou(pred_mask[0], label[0].to(args.device), False)
    for idx in range(num_stacks - 1):
        loss1 += miou(pred_mask[idx + 1], label[0].to(args.device), False)

    for idx, output in enumerate(pred_mask[-2:]):
        loss1 += miou(output, label[idx + 1].to(args.device), False)

    loss2 = ce(pred_vec[0], vecmap_angles[0].to(args.device))
    for idx in range(num_stacks - 1):
        loss2 += ce(pred_vec[idx + 1], vecmap_angles[0].to(args.device))
    for idx, pred_vecmap in enumerate(pred_vec[-2:]):
        loss2 += ce(pred_vecmap, vecmap_angles[idx + 1].to(args.device))

    if epoch > 2000:
        loss3 = topo(torch.amax(pred_mask[-1][0], dim=0), label[-1][0].to(args.device))
        loss = args.weight_miou * loss1 + args.weight_vec * loss2 + loss3
        return loss, loss1, loss2, loss3 

    loss = args.weight_miou * loss1 + args.weight_vec * loss2
    return loss, loss1, loss2, 0


def train_one_epoch(train_loader, model, criterion, optimizer, metric_fns, epoch, args):
    metrics = {
        "loss": [],
        "val_loss": [],
        "road_loss": [],
        "val_road_loss": [],
        "angle_loss": [],
        "val_angle_loss": [],
        "topo_loss": [],
        "val_topo_loss": [],
    }
    for k in list(metric_fns):
        metrics[k] = []
        metrics[f"val_{k}"] = []

    # pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
    # pbar.set_postfix({k: 0 for k in metrics})

    model.train()
    for (inputsBGR, labels, vecmap_angles) in tqdm(train_loader):
        # if len(metrics["loss"]) > 0:
        #     continue

        inputsBGR = inputsBGR.to(args.device)
        metrics, road_loss, angle_loss, topo_loss = train_step(
            model, criterion, optimizer, metric_fns, metrics, inputsBGR, labels, vecmap_angles, epoch, args
        )
        metrics["road_loss"].append(road_loss)
        metrics["angle_loss"].append(angle_loss)
        metrics["topo_loss"].append(topo_loss)

        # pbar.set_postfix({k: sum(v) / len(v) for k, v in metrics.items() if len(v) > 0})

    logging.info(f"\tTrain Loss: {sum(metrics['loss']) / len(metrics['loss']):.4f}")
    logging.info(f"\tTrain Road Loss: {sum(metrics['road_loss']) / len(metrics['road_loss']):.4f}")
    logging.info(
        f"\tTrain Angle Loss: {sum(metrics['angle_loss']) / len(metrics['angle_loss']):.4f}"
    )
    logging.info(
        f"\tTrain Topo Loss: {sum(metrics['topo_loss']) / len(metrics['topo_loss']):.4f}"
    )
    for k in list(metric_fns):
        logging.info(f"\tTrain {k}: {sum(metrics[k]) / len(metrics[k]):.4f}")

    return metrics


def train_step(model, loss_fn, optimizer, metric_fns, metrics, x, y, y_vec, epoch, args):
    optimizer.zero_grad()

    pred_mask, pred_vec = model(x)

    loss, road_loss, angle_loss, topo_loss = criterion(
        num_stacks=model.num_stacks,
        loss_fn=loss_fn,
        pred_mask=pred_mask,
        pred_vec=pred_vec,
        label=y,
        vecmap_angles=y_vec,
        epoch=epoch,
        args=args,
    )

    loss.backward()
    optimizer.step()

    metrics["loss"].append(loss.item())
    if args.multi_scale_pred:
        pred_mask = pred_mask[-1]
        y = y[-1]

    for k, fn in metric_fns.items():
        metrics[k].append(fn(pred_mask.argmax(dim=1).float(), y.to(device=args.device)).item())

    return metrics, road_loss, angle_loss, topo_loss


def evaluate_model(val_loader, model, loss_fn, metric_fns, epoch, metrics, args):
    model.eval()
    with torch.no_grad():  # do not keep track of gradients
        show = (epoch % 5) == 0
        for (inputsBGR, y, y_vec) in tqdm(val_loader):
            inputsBGR = inputsBGR.to(device=args.device)
            pred_mask, pred_vec = model(inputsBGR)  # forward pass

            loss, road_loss, angle_loss, topo_loss = criterion(
                num_stacks=model.num_stacks,
                loss_fn=loss_fn,
                pred_mask=pred_mask,
                pred_vec=pred_vec,
                label=y,
                vecmap_angles=y_vec,
                epoch=epoch,
                args=args,
            )

            if args.multi_scale_pred:
                pred_mask = pred_mask[-1]
                pred_vec = pred_vec[-1]
                y = y[-1]

            if show:
                n = 4
                images = [img.permute(1, 2, 0).cpu().numpy()[:, :, ::-1] for img in inputsBGR]
                images = [(img * 255).astype(np.uint8) for img in images]
                images = [Image.fromarray(img) for img in images]

                masks = [mask.numpy() for mask in y]
                masks = [(mask * 255).astype(np.uint8) for mask in masks]
                masks = [Image.fromarray(mask).convert("L") for mask in masks]

                pred_masks = [mask.argmax(dim=0).cpu().numpy() for mask in pred_mask]
                pred_masks = [(mask * 255).astype(np.uint8) for mask in pred_masks]
                pred_masks = [Image.fromarray(mask).convert("L") for mask in pred_masks]

                plot_predictions(
                    images[:n],
                    masks[:n],
                    pred_masks[:n],
                    os.path.join("./checkpoints", "predictions_" + args.model_name + f"_{epoch + 1 }.pdf"),
                    epoch,
                )
                show = False

            metrics["val_loss"].append(loss.item())
            metrics["val_road_loss"].append(road_loss)
            metrics["val_angle_loss"].append(angle_loss)
            metrics["val_topo_loss"].append(topo_loss)

            for k, fn in metric_fns.items():
                metrics[f"val_{k}"].append(
                    fn(pred_mask.argmax(dim=1).float(), y.to(device=args.device))
                )

            # summarize metrics, log to tensorboard and display
    logging.info(f"\tValidation Loss: {sum(metrics['val_loss']) / len(metrics['val_loss']):.4f}")
    logging.info(
        f"\tValidation Road Loss: {sum(metrics['val_road_loss']) / len(metrics['val_road_loss']):.4f}"
    )
    logging.info(
        f"\tValidation Angle Loss: {sum(metrics['val_angle_loss']) / len(metrics['val_angle_loss']):.4f}"
    )
    logging.info(
        f"\tValidation Topo Loss: {sum(metrics['val_topo_loss']) / len(metrics['val_topo_loss']):.4f}"
    )
    for k in list(metric_fns):
        logging.info(f'\tValidation {k}: {sum(metrics[f"val_{k}"]) / len(metrics[f"val_{k}"]):.4f}')

    return metrics
