import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.evaluate_utils import log_metrics, plot_predictions

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
        ]  # d1 = 128, 128, 128
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


def criterion(num_stacks, loss_fn, pred_mask, pred_vec, mask, vecmap_angles, args):
    miou_func = loss_fn[0]
    vec_func = loss_fn[1]
    bce_func = loss_fn[2]
    topo_func = loss_fn[3]
    dice_func = loss_fn[4]

    if not args.multi_scale_pred:
        miou_loss = miou_func(pred_mask, mask.to(args.device))
        vec_loss = vec_func(pred_vec, vecmap_angles.to(args.device))
        bce_loss = bce_func(pred_mask, mask.to(args.device))

        label_one_hot = torch.stack([1 - mask, mask], dim=1).to(args.device)
        topo_loss = topo_func(label_one_hot, pred_mask.softmax(dim=1))
        dice_loss = dice_func(label_one_hot, pred_mask.softmax(dim=1))

        loss = (
            args.weight_miou * miou_loss
            + args.weight_vec * vec_loss
            + args.weight_bce * bce_loss
            + args.weight_topo * topo_loss
            + args.weight_dice * dice_loss
        )

        return loss, [miou_loss, vec_loss, bce_loss, topo_loss, dice_loss]

    # get average loss over all stacks
    miou_loss = miou_func(pred_mask[0], mask[0].to(args.device))
    vec_loss = vec_func(pred_vec[0], vecmap_angles[0].to(args.device))

    pm_logits = pred_mask[0].softmax(dim=1)[:, 1, :, :]
    bce_loss = bce_func(pm_logits, mask[0].to(args.device))

    label_one_hot = torch.stack([1 - mask[0], mask[0]], dim=1).to(args.device)
    topo_loss = topo_func(label_one_hot, pred_mask[0].softmax(dim=1))
    dice_loss = dice_func(label_one_hot, pred_mask[0].softmax(dim=1))

    for i in range(1, num_stacks):
        miou_loss += miou_func(pred_mask[i], mask[i - 1].to(args.device))
        vec_loss += vec_func(pred_vec[i], vecmap_angles[i - 1].to(args.device))

        pm_logits = pred_mask[i].softmax(dim=1)[:, 1, :, :]
        bce_loss += bce_func(pm_logits, mask[i - 1].to(args.device))

        label_one_hot = torch.stack([1 - mask[i - 1], mask[i - 1]], dim=1).to(args.device)
        topo_loss += topo_func(label_one_hot, pred_mask[i].softmax(dim=1))
        dice_loss += dice_func(label_one_hot, pred_mask[i].softmax(dim=1))

    # add loss of final classification for higher weighting
    miou_loss = miou_loss / num_stacks + miou_func(pred_mask[-1], mask[-1].to(args.device))
    vec_loss = vec_loss / num_stacks + vec_func(pred_vec[-1], vecmap_angles[-1].to(args.device))

    pm_logits = pred_mask[-1].softmax(dim=1)[:, 1, :, :]
    bce_loss = bce_loss / num_stacks + bce_func(pm_logits, mask[-1].to(args.device))

    label_one_hot = torch.stack([1 - mask[-1], mask[-1]], dim=1).to(args.device)
    topo_loss = topo_loss / num_stacks + topo_func(label_one_hot, pred_mask[-1].softmax(dim=1))
    dice_loss = dice_loss / num_stacks + dice_func(label_one_hot, pred_mask[-1].softmax(dim=1))

    loss = (
        args.weight_miou * miou_loss
        + args.weight_vec * vec_loss
        + args.weight_bce * bce_loss
        + args.weight_topo * topo_loss
        + args.weight_dice * dice_loss
    )

    return loss, [miou_loss, vec_loss, bce_loss, topo_loss, dice_loss]


def train_one_epoch(
    train_loader, model, criterion, optimizer, lr_scheduler, metric_fns, epoch, args
):
    metrics = {
        "loss": [],
        "val_loss": [],
        "road_loss": [],
        "val_road_loss": [],
        "angle_loss": [],
        "val_angle_loss": [],
        "topo_loss": [],
        "val_topo_loss": [],
        "bce_loss": [],
        "val_bce_loss": [],
        "dice_loss": [],
        "val_dice_loss": [],
    }
    for k in list(metric_fns):
        metrics[k] = []
        metrics[f"val_{k}"] = []

    model.train()
    for (img, mask, vecmap_angles) in tqdm(train_loader):
        img = img.to(args.device)

        # for _ in range(100):
        metrics = train_step(
            model=model,
            loss_fn=criterion,
            optimizer=optimizer,
            metric_fns=metric_fns,
            metrics=metrics,
            img=img,
            mask=mask,
            y_vec=vecmap_angles,
            args=args,
        )

    lr_scheduler.step()

    return metrics


def train_step(model, loss_fn, optimizer, metric_fns, metrics, img, mask, y_vec, args):
    optimizer.zero_grad()

    pred_mask, pred_vec = model(img)

    loss, losses = criterion(
        num_stacks=model.num_stacks,
        loss_fn=loss_fn,
        pred_mask=pred_mask,
        pred_vec=pred_vec,
        mask=mask,
        vecmap_angles=y_vec,
        args=args,
    )

    loss.backward()
    optimizer.step()

    metrics["loss"].append(loss.item())
    metrics["road_loss"].append(losses[0].cpu().detach().numpy())
    metrics["angle_loss"].append(losses[1].cpu().detach().numpy())
    metrics["topo_loss"].append(losses[2].cpu().detach().numpy())
    metrics["bce_loss"].append(losses[3].cpu().detach().numpy())
    metrics["dice_loss"].append(losses[4].cpu().detach().numpy())

    if args.multi_scale_pred:
        pred_mask = pred_mask[-1]
        mask = mask[-1]

    for k, fn in metric_fns.items():
        metrics[k].append(fn(pred_mask.argmax(dim=1).float(), mask.to(device=args.device)).item())

    return metrics


def evaluate_model(val_loader, model, loss_fn, metric_fns, epoch, metrics, checkpoint_path, args):
    model.eval()
    with torch.no_grad():  # do not keep track of gradients
        show = True
        for (img, mask, y_vec) in tqdm(val_loader):
            img = img.to(device=args.device)
            pred_mask, pred_vec = model(img)  # forward pass

            loss, losses = criterion(
                num_stacks=model.num_stacks,
                loss_fn=loss_fn,
                pred_mask=pred_mask,
                pred_vec=pred_vec,
                mask=mask,
                vecmap_angles=y_vec,
                args=args,
            )

            if args.multi_scale_pred:
                pred_mask = pred_mask[-1]
                pred_vec = pred_vec[-1]
                mask = mask[-1]

            if show and (epoch + 1) % 5 == 0:
                n = 5

                plot_predictions(
                    img[:n],
                    mask[:n],
                    pred_mask[:n],
                    os.path.join(checkpoint_path, "plots", f"predictions_{epoch + 1}.pdf"),
                    epoch,
                )
                show = False

            metrics["val_loss"].append(loss.item())
            metrics["val_road_loss"].append(losses[0].cpu().detach().numpy())
            metrics["val_angle_loss"].append(losses[1].cpu().detach().numpy())
            metrics["val_topo_loss"].append(losses[2].cpu().detach().numpy())
            metrics["val_bce_loss"].append(losses[3].cpu().detach().numpy())
            metrics["val_dice_loss"].append(losses[4].cpu().detach().numpy())

            for k, fn in metric_fns.items():
                metrics[f"val_{k}"].append(
                    fn(pred_mask.argmax(dim=1).float(), mask.to(device=args.device)).item()
                )

    log_metrics(metrics)

    return metrics
