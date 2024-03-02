# Astro-VOS: Tracking Supernovae Evolution Using Video Object Segmentation

[Joycelyn Chen](https://github.com/Joycelyn-Chen)

[[arXiv]]() [[PDF]]() [[Project Page]](https://github.com/Joycelyn-Chen/Astro-VOS.git) 

## Training
- `python -m torch.distributed.run --master_port 25763 --nproc_per_node=2 train.py --exp_id retrain --stage 4 --num_workers 16 --loader_shuffle False --s4_batch_size 6`

## Evaluation
- `python eval.py --output ../output/astro --mem_every 3 --dataset G --generic_path astro_test_root --save_scores --size -1 --model ./saves/<>.pth `
    - not the checkpoint one

- `python eval.py --output ../Evaluation/output/astro_0219 --dataset A --astro_path /home/joy0921/Desktop/XMEM/astro-davis --size -1 --model ./saves/Feb12_19.26.55_retrain_s4/Feb12_19.26.55_retrain_s4_150000.pth`


## Features

* 


### Table of Contents

1. [Introduction](#introduction)
2. [Results](docs/RESULTS.md)
3. [Training/inference](#traininginference)
4. [Citation](#citation)

### Introduction




### Training/inference

First, install the required python packages and datasets following [GETTING_STARTED.md](docs/GETTING_STARTED.md).

For training, see [TRAINING.md](docs/TRAINING.md).

For inference, see [INFERENCE.md](docs/INFERENCE.md).

### Citation

Please cite our paper if you find this repo useful!

```bibtex

```