import torch
from torchmetrics import F1Score
from datasets import road_dataset
from models.hourglas_spin import HourglassNet
from datasets.road_dataset import RoadDataset

from utils import utils, evaluate_utils
from tqdm import tqdm

from PIL import Image

import numpy as np
import pandas as pd

from topoloss.topoloss import getCriticalPoints

import os

from sklearn.metrics import f1_score

from main import load_data_info_with_split


def main(args):
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  train_df, val_df = load_data_info_with_split(args)
  # Specify a path
  PATH = "SPI_120epochs_WEIGHTED"

  # Load
  model = HourglassNet()
  model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
  model.eval()

  target_size = (400, 400)

  # Test data
  test_dataset = road_dataset.RoadDataset(
      dataframe=val_df,
      target_size=target_size,
      multi_scale_pred=args.multi_scale_pred,
      is_train=False,
      args=args
  )
  test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_workers,
  )

  df = []
  idx = 0
  with torch.no_grad():
    for (inputsBGR, labels, vecmap_angles) in tqdm(test_loader): 
      inputsBGR = inputsBGR.to(args.device)
      pred_mask, pred_vec = model(inputsBGR)
      
      name = val_df["filename"].to_list()[idx]
      label = labels[-1][0].numpy()

      pred_mask = pred_mask[-1][0].argmax(dim=0).cpu().numpy()
      f1score = f1_score(label, pred_mask, average='weighted')

      images = [img.permute(1, 2, 0).cpu().numpy()[:, :, ::-1] for img in inputsBGR]
      images = [(img * 255).astype(np.uint8) for img in images]
      images = [Image.fromarray(img) for img in images]

      mask = (label * 255).astype(np.uint8)
      masks = [Image.fromarray(mask).convert("L")]

      pred_mask = (pred_mask * 255).astype(np.uint8)
      pred_masks = [Image.fromarray(pred_mask).convert("L")]


      evaluate_utils.plot_predictions(images, masks, pred_masks, "./eval_imgs/" + name + ".pdf", 1)

      imb = getCriticalPoints(pred_mask)[-2]
      gtb = getCriticalPoints(mask)[-2]
      imBetti = len(imb) if isinstance(imb, np.ndarray) else 1
      gtBetti = len(gtb) if isinstance(gtb, np.ndarray) else 1
      df.append((idx, name, imBetti, gtBetti, f1score))
      idx += 1
    
  df = pd.DataFrame(df, columns=['id', 'name', 'imBetti', 'gtBetti', 'f1'])
  df.to_csv(os.path.join("topoData3.csv"), index=False)




args = utils.parse_arguments()
main(args)



