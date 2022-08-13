# Trans-INR

This repository contains the official implementation for the following paper:

[**Transformers as Meta-Learners for Implicit Neural Representations**](https://arxiv.org/abs/2208.02801)
<br>
[Yinbo Chen](https://yinboc.github.io/), [Xiaolong Wang](https://xiaolonw.github.io/)
<br>
ECCV 2022

<img src="https://user-images.githubusercontent.com/10364424/183021009-b0d15bf4-70ec-4402-8f17-0b26ecacc3f9.png" width="400">

Project page: https://yinboc.github.io/trans-inr/.

```
@inproceedings{chen2022transinr,
  title={Transformers as Meta-Learners for Implicit Neural Representations},
  author={Chen, Yinbo and Wang, Xiaolong},
  booktitle={European Conference on Computer Vision},
  year={2022},
}
```

## Reproducing Experiments

### Environment
- Python 3
- Pytorch 1.12.0
- pyyaml numpy tqdm imageio TensorboardX wandb einops

### Data

`mkdir data` and put different dataset folders in it.

- **CelebA**: download ([from kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)), extract, and rename the folder as `celeba` (so that images are in `data/celeba/img_align_celeba/img_align_celeba`).

- **Imagenette**: [download](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz), extract, and rename the folder as `imagenette`.

- **View synthesis**: download from [google drive](https://drive.google.com/drive/folders/1lRfg-Ov1dd3ldke9Gv9dyzGGTxiFOhIs) (provided by [learnit](https://www.matthewtancik.com/learnit)) and put them in a folder named `learnit_shapenet`, unzip the category folders and rename them as `chairs`, `cars`, `lamps` correspondingly.

### Training

Run `CUDA_VISIBLE_DEVICES=[GPU] python run_trainer.py --cfg [CONFIG]`, configs are in `cfgs/`.

To enable [wandb](https://wandb.ai/home), complete `wandb.yaml` (in root) and add `-w` to the training command.

When running multiple multi-gpu training processes, specify `-p` with different values (0,1,2...) for different ports.

### Evaluation

For image reconstruction, test PSNR is automatically evaluated in the training script.

For view synthesis, run in a single GPU with configs in `cfgs/nvs_eval`. To enable test-time optimization, uncomment (remove `#`) `tto_steps` in configs.
