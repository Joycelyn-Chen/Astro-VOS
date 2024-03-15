# Astro-VOS: Tracking Supernovae Evolution Using Video Object Segmentation

[Joycelyn Chen](https://github.com/Joycelyn-Chen)

[[arXiv]]() [[PDF]]() [[Project Page]](https://github.com/Joycelyn-Chen/Astro-VOS.git) 



### Table of Contents

1. [Introduction](#introduction)
2. [Results](docs/RESULTS.md)
3. [Training/inference](#traininginference)
4. [Citation](#citation)

### Introduction
Understanding the influence of supernovae on the interstellar medium (ISM) is crutial for unraveling the complexities of our Galaxy. Traditional methods, however, are inadequate in accurately capturing the three-dimensional structures of superbubbles formed by supernovae, thus constraining detailed quantitative analysis. To bridge this gap, we utilize 3D magnetohydrodynamic numerical simulations to construct a tailored dataset. Moreover, we develop a video object segmentation model to precisely depict the contours of superbubbles within our 3D dataset, offering an in-depth view of superbubble evolution. Our findings, verified against the principles of Sedov-Taylor theories, highlight the effectiveness of our innovative approach in delivering accurate and comprehensive insights into the ISM dynamics, significantly outperforming traditional astrophysical methods.

![Teaser figure](imgs/teaser.png?raw=true)

### Training/inference

First, install the required python packages and datasets following [GETTING_STARTED.md](docs/GETTING_STARTED.md).

For training, see [TRAINING.md](docs/TRAINING.md).


For inference, see [INFERENCE.md](docs/INFERENCE.md).

### Citation

Please cite our paper if you find this repo useful!

```bibtex

```

and the reference model:

```bibtex
@inproceedings{cheng2022xmem,
  title={{XMem}: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model},
  author={Cheng, Ho Kei and Alexander G. Schwing},
  booktitle={ECCV},
  year={2022}
}
```