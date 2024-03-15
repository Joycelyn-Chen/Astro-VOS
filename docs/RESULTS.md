# Results

## Pretrained models

THe XMem team provide four pretrained models for download:

1. XMem.pth (Default)
2. XMem-s012.pth (Trained with BL30K)
3. XMem-s2.pth (No pretraining on static images)
4. XMem-no-sensory (No sensory memory)

The model without pretraining is for reference. The model without sensory memory might be more suitable for tasks without spatial continuity, like mask tracking in a multi-camera 3D reconstruction setting, though you could try the base model as well.

Download them from [[GitHub]](https://github.com/hkchengrex/XMem/releases/tag/v1.0) or [[Google Drive]](https://drive.google.com/drive/folders/1QYsog7zNzcxGXTGBzEhMUg8QVJwZB6D1?usp=sharing).

## MHD-VOS Dataset

| Model | J&F | J | J-Recall | F | F-Recall |
| --- | :--:|:--:|:---:|:---:|:---:|
| [TAM](https://github.com/gaomingqi/Track-Anything) | 72.6 | 72.3 | 80.9 | 72.9 | 75.7 |
| [SAM](https://github.com/facebookresearch/segment-anything) | 79.5 | 79.8 | 83.9 | 79.2 | 81.8 |
| [XMem](https://github.com/hkchengrex/XMem/tree/main) | 79.6 | 79.6 | 84.6 | 79.5 | 81.6 |
| Ours | 97.7 | 96.8 | 100 | 98.7 | 100 |

