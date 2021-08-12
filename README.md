# Inbetweening
### [Paper](https://arxiv.org/abs/2008.04149)

The source code of Deep Sketch-guided Cartoon Video Inbetweening by Xiaoyu Li, Bo Zhang, Jing Liao, Pedro V. Sander, IEEE Transactions on Visualization and Computer Graphics, 2021.

<img src='figures/teaser.png' align="center" width=875>

## Prerequisites
- Linux or Windows
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Use a Pre-trained Model
You can download the pre-trained model [here](https://drive.google.com/file/d/1wn4MfKv2bTH_EWobZB-nE875s9l913rq/view?usp=sharing).

Run the following commands for evaluting the frame synthesis model and full model.
```bash
python eval_synthesis.py
python eval_full.py
```
The frame synthesis model takes img_0, img_1, ske_t as inputs and synthesize img_t.

The full model takes img_0, img_1, ske_t as inputs and interpolate five frames between img_0 and img_1.
