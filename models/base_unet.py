import os

import torch
from torch import nn
from tqdm import tqdm
from utils.evaluate_utils import log_metrics, plot_predictions

PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road


class Block(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder
        self.enc_blocks = nn.ModuleList(
            [Block(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])]
        )  # encoder blocks
        self.pool = nn.MaxPool2d(2)  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
                for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
            ]
        )  # deconvolution
        self.dec_blocks = nn.ModuleList(
            [Block(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])]
        )  # decoder blocks
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[-1], 1, 1),  # output is a single channel
        )  # 1x1 convolution for producing the output

    def forward(self, x):
        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)
        # decode
        for block, upconv, feature in zip(self.dec_blocks, self.upconvs, enc_features[::-1]):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block
        return self.head(x).squeeze(1)  # reduce to 1 channel


def criterion(loss_fn, pred_mask, mask, args):
    miou_func = loss_fn[0]
    vec_func = loss_fn[1]
    bce_func = loss_fn[2]
    topo_func = loss_fn[3]
    dice_func = loss_fn[4]

    pred_mask = pred_mask.sigmoid()

    bce_loss = bce_func(pred_mask, mask.to(args.device))

    label_one_hot = torch.stack([1 - mask, mask], dim=1).to(args.device)
    pred_one_hot = torch.stack([1 - pred_mask, pred_mask], dim=1).to(args.device)
    topo_loss = topo_func(label_one_hot, pred_one_hot)
    dice_loss = dice_func(label_one_hot, pred_one_hot)

    loss = args.weight_bce * bce_loss + args.weight_topo * topo_loss + args.weight_dice * dice_loss

    return loss, [torch.tensor(0), torch.tensor(0), topo_loss, bce_loss, dice_loss]


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
    for (img, mask) in tqdm(train_loader):
        img = img.to(args.device)

        metrics = train_step(
            model=model,
            loss_fn=criterion,
            optimizer=optimizer,
            metric_fns=metric_fns,
            metrics=metrics,
            img=img,
            mask=mask,
            args=args,
        )

    lr_scheduler.step()

    return metrics


def train_step(model, loss_fn, optimizer, metric_fns, metrics, img, mask, args):
    optimizer.zero_grad()

    pred_mask = model(img)

    loss, losses = criterion(
        loss_fn=loss_fn,
        pred_mask=pred_mask,
        mask=mask,
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

    for k, fn in metric_fns.items():  # TODO: make pred 0, 1
        metrics[k].append(fn(pred_mask.sigmoid(), mask.to(device=args.device)).cpu())

    return metrics


def evaluate_model(val_loader, model, loss_fn, metric_fns, epoch, metrics, checkpoint_path, args):
    model.eval()
    with torch.no_grad():  # do not keep track of gradients
        show = True
        for (img, mask) in tqdm(val_loader):
            img = img.to(device=args.device)
            pred_mask = model(img)  # forward pass

            loss, losses = criterion(
                loss_fn=loss_fn,
                pred_mask=pred_mask,
                mask=mask,
                args=args,
            )

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
                    fn(pred_mask.sigmoid(), mask.to(device=args.device)).cpu()
                )
            # break

    log_metrics(metrics)

    return metrics
