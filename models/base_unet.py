import torch
from PIL import Image
from torch import nn
from tqdm import tqdm
from sklearn.metrics import f1_score

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
            nn.Conv2d(dec_chs[-1], 1, 1), nn.Sigmoid()
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


def patch_accuracy_fn(y_hat, y):
    # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
    h_patches = y.shape[-2] // PATCH_SIZE
    w_patches = y.shape[-1] // PATCH_SIZE
    patches_hat = (
        y_hat.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    )
    patches = y.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    return (patches == patches_hat).float().mean()


def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()
#FIXME: f1 score is only calculated for the first sample
def f1_fn(y_hat, y):
    return f1_score(y[0].cpu().int(), y_hat[0].cpu().int(), average='weighted')

def patch_F1_fn(y_hat, y):
    h_patches = y.shape[-2] // PATCH_SIZE
    w_patches = y.shape[-1] // PATCH_SIZE
    patches_hat = torch.squeeze(
        y_hat.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    )
    patches = torch.squeeze(y.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF)
    return f1_score(patches[0].cpu().int(), patches_hat[0].cpu().int(), average='weighted')


def train_one_epoch(train_loader, model, criterion, optimizer, metrics, epoch, args):
    metrics = {"loss": [], "val_loss": []}
    for k in metrics:
        metrics[k] = []
        metrics["val_" + k] = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
    pbar.set_postfix({k: 0 for k in metrics})

    model.train()
    for (x, y) in pbar:
        x = x.to(args.device)
        y = y.to(args.device)

        train_step(model, criterion, optimizer, metrics, metrics, x, y, args)

        pbar.set_postfix({k: sum(v) / len(v) for k, v in metrics.items() if len(v) > 0})

    return metrics


def train_step(model, loss_fn, optimizer, metric_fns, metrics, x, y, args):

    optimizer.zero_grad()  # zero out gradients
    y_hat = model(x)  # forward pass

    loss = loss_fn(y, y_hat)  # compute loss
    loss.backward()  # backward pass

    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()  # optimize weights

    # log partial metrics
    metrics["loss"].append(loss.item())
    for k, fn in metric_fns.items():
        metrics[k].append(fn(y_hat, y).item())


def evaluate_model(val_loader, model, loss_fn, metric_fns, epoch, metrics, args):
    model.eval()
    with torch.no_grad():  # do not keep track of gradients
        for (x, y) in tqdm(val_loader):
            x = x.to(args.device)
            y = y.to(args.device)

            y_hat = model(x)  # forward pass

            loss = loss_fn(y, y_hat)

            # log partial metrics
            metrics["val_loss"].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[f"val_{k}"].append(fn(y_hat, y).item())

            # summarize metrics, log to tensorboard and display
    return metrics
