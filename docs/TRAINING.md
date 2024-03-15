# Training

First, set up the datasets following [GETTING STARTED.md](./GETTING_STARTED.md).

The model is trained progressively with different stages (0: static images; 1: BL30K; 2: longer main training; 3: shorter main training; 4: fine-tune on MHD-VOS dataset). After each stage finishes, the XMem team start the next stage by loading the latest trained weight.
For example, the base model is pretrained with static images followed by the shorter main training (s03).

To train the base model on two GPUs, you can use:

```bash
python -m torch.distributed.run --master_port 25763 --nproc_per_node=2 train.py --exp_id retrain --stage 03
```
(**NOTE**: Unexplained accuracy decrease might occur if you are not using two GPUs to train. See https://github.com/hkchengrex/XMem/issues/71.)

`master_port` needs to point to an unused port. 
`nproc_per_node` refers to the number of GPUs to be used (specify `CUDA_VISIBLE_DEVICES` to select which GPUs to use).
`exp_id` is an identifier you give to this training job.

See other available command line arguments in `util/configuration.py`.
**Unlike the training code of STCN, batch sizes are effective. You don't have to adjust the batch size when you use more/fewer GPUs.**

The XMem team implemented automatic staging in this code base. You don't have to train different stages by yourself like in STCN (but that is still supported).
`stage` is a string that the XMem team split to determine the training stages. Examples include `0` (static images only), `03` (base training), `012` (with BL30K), `2` (main training only), `4` (training on MHD-VOS dataset).

You can use `tensorboard` to visualize the training process.

## Outputs

The model files and checkpoints will be saved in `./saves/[name containing datetime and exp_id]`.

`.pth` files with `_checkpoint` store the network weights, optimizer states, etc. and can be used to resume training (with `--load_checkpoint`).

Other `.pth` files store the network weights only and can be used for inference. The XMem team note that there are variations in performance across different training runs and across the last few saved models. For the base model, the XMem team most often note that main training at 107K iterations leads to the best result (full training is 110K).

[The original post](https://github.com/hkchengrex/XMem/blob/main/docs/TRAINING.md) measure the median and std scores across five training runs of the base model:

| Dataset |  median | std |
| --- | :--:|:--:|
| DAVIS J&F | 86.2 | 0.23 |
| YouTubeVOS 2018 G | 85.6 | 0.21

## Pretrained models

You can start training from scratch, or use any of our pretrained models for fine-tuning. For example, you can load our stage 0 model to skip main training:

```bash
python -m torch.distributed.launch --master_port 25763 --nproc_per_node=2 train.py --exp_id retrain_stage3_only --stage 3 --load_network saves/XMem-s0.pth
```

To fine-tuning the pretrained `XMem-s0.pth` on MHD-VOS dataset:
```bash
python -m torch.distributed.run --master_port 25763 --nproc_per_node=2 train.py --exp_id retrain --stage 4 --num_workers 16 --loader_shuffle False --s4_batch_size 6
```

Download them from [[GitHub]](https://github.com/hkchengrex/XMem/releases/tag/v1.0) or [[Google Drive]](https://drive.google.com/drive/folders/1QYsog7zNzcxGXTGBzEhMUg8QVJwZB6D1?usp=sharing).

