# Road Segmentation
This project is part of the cours "Computational Intelligence Lab" at ETH Zurich.

**Team:**
| Name             |  ethz id  |
| :--------------- | :-------: |
| Talu Karagöz     | tkaragoez |
| András Strausz   |  stausza  |
| Alexander Veicht |  veichta  |

---
## Notes (FIXME)

To prepare data first create file structure like in [create_dir_structure.sh](create_dir_structure.sh) and then run [create_crops.sh](utils/create_crops.py). For the latter, use:
```
python create_crops.py --base_dir ./CIL_data --crop_size 256 --crop_overlap 10 --im_suffix .png --gt_suffix .png --im_prefix satimage_ --gt_prefix satimage_
```
- [ ] Determine crop size and overlap!

