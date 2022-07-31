# Road Segmentation
**This repository is part of the course "Computational Intelligence Lab" at ETH Zurich.**

In this project we aim to improve the [SPIN](https://github.com/wgcban/SPIN_RoadMapper) module by complex image augmentation, the ResNeXt backbone architecture and a topology-aware loss function.


**Team:**
| Name             |  ethz id  |
| :--------------- | :-------: |
| Talu Karagöz     | tkaragoez |
| András Strausz   |  stausza  |
| Alexander Veicht |  veichta  |

---

## Repository structure
Main code can be found under [main.py](main.py), the training and evaluation loops are in the respective scripts of the modules under [models](models/). [utils](utils/) contains the loss functions used (including topoloss) and several logging and helper scripts.
## Credits

The main model is mostly based on the implementation from the SPIN reposotory, with slight modifications. The ResNeXt module has been taken from [ResNeXt](https://github.com/prlz77/ResNeXt.pytorch).

## Reconstruction of results

To create the desired data structure run the [prepare-datasets.ipynb](notebooks/prepare-datasets.ipynb) notebook.

The following scripts can be used to reconstruct results reported in the paper:

- SPIN baseline
  ```python main.py --model spin --device cuda --num_epochs 200 --batch_size 6 --num_workers 6 --min_pixels 50000 --datasets cil --lr 0.01 --weight_miou 1 --weight_vec 1 --weight_topo 0 --topo_after 200 --backbone resnext```
- UNET baseline
  ```python main.py --model unet --device cuda --num_epochs 200 --batch_size 6 --num_workers 6 --min_pixels 50000 --datasets cil --lr 0.01 --weight_miou 1 --weight_vec 1 --weight_topo 0 --topo_after 200 --backbone resnext```
- Augmentation
  ```python main.py --model spin --device cuda --num_epochs 200 --batch_size 6 --num_workers 6 --min_pixels 50000 --datasets cil --lr 0.01 --weight_miou 1 --weight_vec 1 --weight_topo 0 --topo_after 200 --backbone resnext --```
- Backbone
  ```python main.py --model spin --device cuda --num_epochs 200 --batch_size 6 --num_workers 6 --min_pixels 50000 --datasets cil --lr 0.01 --weight_miou 1 --weight_vec 1 --weight_topo 0 --topo_after 200 --backbone resnext --```
- Topology
  ```python main.py --model spin --device cuda --num_epochs 200 --batch_size 6 --num_workers 6 --min_pixels 50000 --datasets cil --lr 0.01 --weight_miou 1 --weight_vec 1 --weight_topo 0 --topo_after 200 --backbone resnext --```

**Note:** In case the runs produce only noisy predictions, please clear pycache before execution.
