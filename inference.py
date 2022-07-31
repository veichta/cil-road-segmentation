import torch
from datasets import road_dataset
from models.hourglas_spin import HourglassNet
from datasets.road_dataset import RoadDataset

from utils import utils
from tqdm import tqdm

from PIL import Image

import numpy as np

from main import load_data_info_with_split


def main(args):
    args.inference = True
    assert(args.inference)  # Use this module only for inference

    test_df = load_data_info_with_split(args)

    # Specify a path to trained model
    PATH = "SPI_120epochs_WEIGHTED_NonNormalized"

    # Load
    model = HourglassNet()
    model.load_state_dict(torch.load(
        PATH, map_location=torch.device(args.device)))
    model.eval()

    target_size = (400, 400)

    # Test data
    test_dataset = road_dataset.RoadDataset(
        dataframe=test_df,
        target_size=target_size,
        multi_scale_pred=args.multi_scale_pred,
        is_train=False,
        args=args,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    idx = 144
    with torch.no_grad():
        for (inputsBGR, labels, vecmap_angles) in tqdm(test_loader):
            inputsBGR = inputsBGR.to(args.device)
            pred_mask, pred_vec = model(inputsBGR)
            prediction = pred_mask[-1][0].argmax(dim=0).cpu().numpy()
            prediction = (prediction * 255).astype(np.uint8)
            prediction = Image.fromarray(prediction)
            prediction.save(
                "data/CIL-dataset/test/predictions/" + str(idx) + ".jpg")

            idx += 1


if __name__ == '__main__':
    args = utils.parse_arguments()
    main(args)
