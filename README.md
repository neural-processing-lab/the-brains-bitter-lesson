# The Brain's Bitter Lesson
This repository contains the code for "The Brain's Bitter Lesson: Scaling Speech Decoding With Self-Supervised Learning", appearing at ICML 2025. Find the paper on ArXiv [here](https://arxiv.org/abs/2406.04328). If you find this code helpful in your work, please cite the paper:
```
@inproceedings{jayalath2024brain,
  title={{The Brain's Bitter Lesson: Scaling Speech Decoding With Self-Supervised Learning}},
  author={Jayalath, Dulhan and Landau, Gilad and Shillingford, Brendan and Woolrich, Mark and Parker Jones, Oiwi},
  booktitle={Forty-second International Conference on Machine Learning, {ICML} 2025, Vancouver, Canada, July 13-19, 2025},
  year={2025},
  organization={PMLR}
}
```

# Quick start

## Prerequisites
1. Install requirements with `pip install -r requirements.txt`.
2. Download the [Armeni](https://data.ru.nl/collections/di/dccn/DSC_3011085.05_995) and [Gwilliams](https://osf.io/ag3kj/) datasets (and optionally [CamCAN](https://cam-can.mrc-cbu.cam.ac.uk/dataset/) and [MOUS](https://data.ru.nl/collections/di/dccn/DSC_3011020.09_236) if pre-training).
3. Modify `datasets.yaml` to point `bids_root` to your dataset's root directory, and change `preproc_root` accordingly.

## Pre-training a model
→ `pretrain.py`

e.g. run `python pretrain.py --datasets camcan mous` to pretrain a model on the CamCAN and MOUS datasets together. A checkpoint will be saved in `checkpoints/`

## Probing a pre-trained model
→ `probe.py`

e.g. run `python probe.py --task speech --pretrained_ckpt checkpoints/<>.ckpt --datasets armeni2022` to probe a pretrained model for speech detection on the Armeni dataset.

## Pre-trained checkpoints
We provide checkpoints for models pretrained with CamCAN in `checkpoints/`.

# TODO
- [ ] Provide code examples for novel subject generalisation experiments
- [ ] Provide code examples for scaling experiments