# MMFNet Image Fusion: Training and Testing

This project fuses registered infrared and visible-light images. Training also requires pixel-aligned semantic segmentation labels. All commands below must be run from the project root containing `train.py`, `training.yml`, and `split_dataset.py`.

The package contains small demonstration datasets, not complete benchmark datasets. No pretrained checkpoint is included. Train a model first, or supply a checkpoint compatible with the current `MMFNet.py` and its segmentation-head class count.

## 1. Environment Setup

Software versions: Windows, Python 3.12.7, PyTorch 2.6.0 with CUDA 12.6, and torchvision 0.21.0. GPU configuration example: NVIDIA RTX 4090. Runtime dependencies are pinned in `requirements.txt`.

```bash
conda create -n mmfnet python=3.12 -y
conda activate mmfnet
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

For CPU use, replace the PyTorch installation command with:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

Training and testing accept `--device auto`, `--device cuda`, or `--device cpu`; the default is `auto`. CPU execution is slower. Check the environment with:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 2. Check the Complete Workflow with the Bundled Samples

The bundled training images are already 320x320. They can be used directly to check the environment:

```bash
python train.py --epochs 1 --batch_size 1 --max_steps 1 --save_dir ./checkpoints_smoke
```

This runs one training batch, saves a checkpoint, calls `test_mfnet.py`, and then calls `metric/test_metric.py`. It checks the workflow; a checkpoint trained for one step is not a benchmark result.

```text
checkpoints_smoke/mfnet/
|-- models/model_latest.pth
|-- results_auto/mfnet/*.png
`-- logs/
    |-- training_config.yml
    |-- train_console_log.txt
    |-- loss_history.csv
    |-- loss_history.json
    |-- loss_total_curve.png
    |-- loss_reconstruction_edge_terms_curve.png
    |-- lr_curve.png
    |-- metric_mfnet.csv
    `-- metric_mfnet.xlsx
```

## 3. Prepare Your Own Data and Run Offline Tiling First

### 3.1. Prepare paired source files

For each training sample, provide infrared input/target, visible-light input/target, and a segmentation label. Under the current reconstruction setup, each modality's input and target can be copies of the same image; the segmentation label must be a genuine corresponding annotation.

```text
dataset/raw/mfnet/
|-- ir/
|   |-- input/000001.png
|   `-- target/000001.png
|-- rgb/
|   |-- input/000001.png
|   `-- target/000001.png
`-- seg/000001.png
```

Use the same layout for `dataset/raw/potsdam/` and `dataset/raw/whu/`. All five folders must contain matching filename stems, and all five images within a sample must have identical dimensions and spatial alignment. Each image must be at least 320 pixels wide and high. Extract the required bands and prepare supported 8-bit image/label files before tiling.

Split scenes or original large images into training and test sets before tiling. Adjacent patches from one original image should not be shared across the training and test sets.

### 3.2. Segmentation label encoding

The existing loaders convert labels to RGB and match exact pixel values. Prepare labels using the following encoding:

| Dataset | Classes | Encoding required by the current loader |
|---|---:|---|
| mfnet | 9 | Class IDs 0 through 8 as grayscale values 0 through 8; RGB equivalents `(0,0,0)` through `(8,8,8)` |
| whu | 8 | Class IDs 0 through 7 as grayscale values `0,10,20,30,40,50,60,70` |
| potsdam | 6 | RGB colors listed below |

| potsdam class ID | RGB label color |
|---:|---|
| 0 | `(255,255,255)` |
| 1 | `(0,0,255)` |
| 2 | `(0,255,255)` |
| 3 | `(0,255,0)` |
| 4 | `(255,255,0)` |
| 5 | `(255,0,0)` |

Save labels as lossless PNG. For mfnet/whu, use grayscale `L` images or RGB images with the required equal channel values. A palette PNG may display unrelated colors when converted to RGB; its numeric indices alone do not guarantee compatibility. Unrecognized colors are mapped to class 0 by the current loaders, not automatically ignored. Do not introduce new colors by interpolation or JPEG compression.

### 3.3. Run `split_dataset.py` before training

Use paired mode to apply identical crop boxes and output names to all five training folders:

```bash
python split_dataset.py --dataset_dir ./dataset/raw/mfnet --output_dir ./dataset/train_320/mfnet
python split_dataset.py --dataset_dir ./dataset/raw/potsdam --output_dir ./dataset/train_320/potsdam
python split_dataset.py --dataset_dir ./dataset/raw/whu --output_dir ./dataset/train_320/whu
```

Run the command for the dataset you are preparing. Each output root contains `ir/input`, `ir/target`, `rgb/input`, `rgb/target`, and `seg`, with matching names such as `000001.png`. `patch_index.csv` records source stems and crop coordinates.

The output directory must be new or empty, and must not contain or be contained by the input directory. Do not mix generated patches with the bundled samples. Tiling does not resize or pad images. Boundary tiles align with the right/bottom edge and can overlap preceding tiles; duplicate crop positions are omitted. Images smaller than 320 in either dimension are rejected. Label values and image mode are preserved during tiling.

The CLI accepts only `--size 320 320`, which is also the default. Single-folder mode is available, but paired mode is recommended for training data:

```bash
python split_dataset.py --input_dir ./images --output_dir ./images_320
python split_dataset.py --input_dir ./labels --output_dir ./labels_320 --keep_mode
```

### 3.4. Prepare test pairs

```text
dataset/test/
|-- mfnet/
|   |-- ir/000001.png
|   `-- rgb/000001.png
|-- potsdam/
|   |-- ir/000001.png
|   `-- rgb/000001.png
|-- whu/
|   |-- ir/254.png
|   `-- rgb/254.png
|-- LLVIP/
|   |-- ir/190064.jpg
|   `-- rgb/190064.jpg
`-- M3FD/
    |-- ir/00000.png
    `-- rgb/00000.png
```

The names above illustrate the layout. Test folders need only `ir` and `rgb`, without `input`, `target`, or `seg` subfolders. Keep paired images aligned, equal in size, and consistently named. The original test loaders pair sorted lists, so missing files can misalign samples. Test widths/heights must be divisible by four; the test scripts do not automatically crop or pad them.

The package includes 10 training samples each for mfnet, potsdam, and whu; test samples comprise 5 pairs each for mfnet, potsdam, LLVIP, and M3FD, and 3 pairs for whu. `dataset/unpaired/whu/ir/` contains unmatched source images retained separately and is not used by the loaders.

## 4. Train on the Prepared Patches

After offline tiling, explicitly select the prepared directory:

```bash
python train.py --dataset mfnet --train_dir ./dataset/train_320/mfnet --num_classes 9
python train.py --dataset potsdam --train_dir ./dataset/train_320/potsdam --num_classes 6
python train.py --dataset whu --train_dir ./dataset/train_320/whu --num_classes 8
```

Run only the command for the dataset you intend to train. CLI arguments override `training.yml`. Selecting `--dataset` alone does not automatically update the training path or class count; keep all three consistent.

| Dataset | `Datasets.data` | Prepared `TRAINING.TRAIN_DIR` | `TRAINING.NUM_CLASSES` |
|---|---|---|---:|
| mfnet | `mfnet` | `./dataset/train_320/mfnet` | 9 |
| potsdam | `potsdam` | `./dataset/train_320/potsdam` | 6 |
| whu | `whu` | `./dataset/train_320/whu` | 8 |

The unchanged `training.yml` defaults to the bundled `./dataset/train/mfnet`, 9 classes, 50 epochs, batch size 4, initial learning rate `2e-5`, and output root `./checkpoints1`. Running `python train.py` without overrides uses that configuration; it does not automatically select `dataset/train_320/`.

For full training, omit `--max_steps`. Reduce memory use with `--batch_size 1`. Windows data loading defaults to `--num_workers 0`. The training loader uses `drop_last=True`, so the dataset must contain at least one full batch. The synchronized 320x320 training crop remains active: prepared 320x320 patches pass through without a spatial change.

Training saves checkpoints under `<save_dir>/<dataset>/models/` and logs under `<save_dir>/<dataset>/logs/`. `model_latest.pth` is the latest completed epoch, not a validation-selected best checkpoint. `VAL_AFTER_EVERY` controls periodic checkpoint saving; there is no separate validation loop.

After training, the script automatically calls the corresponding `test_<dataset>.py` on `./dataset/test/<dataset>/`, followed by `metric/test_metric.py`. Fusion images are saved under `<save_dir>/<dataset>/results_auto/<dataset>/`; CSV/XLSX metrics are saved in the same run's `logs/` directory. Prepare test pairs in advance, or add `--no_auto_test` to skip this step.

To resume an interrupted run, repeat its dataset, path, output directory and total epoch settings, adding `--resume`. For example:

```bash
python train.py --dataset whu --train_dir ./dataset/train_320/whu --num_classes 8 --resume
```

With the default save root, this reads `checkpoints1/whu/models/model_latest.pth` and continues from the next epoch, restoring model, optimizer and scheduler states when present. If the run used `--save_dir`, supply the same value when resuming. A run that has already completed its configured total epochs does not need resuming.

## 5. Standalone Fusion Testing

After default mfnet training:

```bash
python test_mfnet.py --weights ./checkpoints1/mfnet/models/model_latest.pth --num_classes 9
```

This saves images to `results/mfnet/`. For the quick workflow in Section 2, use `./checkpoints_smoke/mfnet/models/model_latest.pth` instead.

| Script | Input below `--input_dir` | Output below `--result_dir` | Color space |
|---|---|---|---|
| `test_mfnet.py` | `mfnet/` | `mfnet/` | YCbCr |
| `test_potsdam.py` | `potsdam/` | `potsdam/` | HSV |
| `test_whu.py` | `whu/` | `whu/` | HSV |
| `test_LLVIP.py` | `LLVIP/` | `LLVIP/` | YCbCr |
| `test_M3FD.py` | `M3FD/` | `M3FD/` | YCbCr |

All five scripts accept `--weights`, `--num_classes`, `--input_dir`, `--result_dir`, `--gpus`, `--device`, and `--num_workers`. Each script's `--dataset` accepts only its own dataset name. Defaults are `--input_dir ./dataset/test`, `--result_dir ./results`, `--gpus 0`, `--device auto`, and `--num_workers 0`. All test batches contain one pair.

The class count must match the checkpoint's segmentation head, not the test dataset name. For example, use the mfnet 9-class checkpoint for LLVIP as follows:

```bash
python test_LLVIP.py --weights ./checkpoints1/mfnet/models/model_latest.pth --num_classes 9
```

LLVIP/M3FD provide fusion test pairs in this package, without segmentation labels for this supervised training entry. Their higher-resolution images require more memory. Existing test processing, including per-image min-max normalization, is retained. Record and use consistent preprocessing when comparing methods.

## 6. Compute Fusion Metrics

After standalone testing:

```bash
python metric/test_metric.py --dataset mfnet --fus_dir ./results/mfnet --save_dir ./results/metrics
```

To recompute metrics for automatic-test outputs:

```bash
python metric/test_metric.py --dataset mfnet --fus_dir ./checkpoints1/mfnet/results_auto/mfnet --save_dir ./checkpoints1/mfnet/logs
```

Supported datasets are `mfnet`, `potsdam`, `whu`, `LLVIP`, and `M3FD`. `--input_dir` defaults to `./dataset/test`; use it when the source test pairs are elsewhere. `--fus_dir` points directly to the fused-image folder. Results are written to `metric_<dataset>.csv` and `.xlsx`, with per-image rows and a `mean` row for `EN`, `SD`, `SF`, `AG`, `MI`, `SCD`, `VIFF`, `Qabf`, and `SSIM`.

Evaluation follows the existing `Evaluator.py`: all source/fused images are converted to grayscale and resized to 224x224 before computing metrics. The SSIM column is the sum of comparisons with the two source images and can exceed 1. MI uses the existing post-resize values and may produce a scikit-learn continuous-input warning. Undefined/non-finite metrics cause an error. Source and output files are matched by filename stem, allowing JPG sources and PNG fusion outputs.

## 7. Dataset Download Locations
The following author/official download pages were checked on November 16, 2026. The complete datasets were not downloaded during this update.





| Role | Dataset | Official download information | Download location |


|---|---|---|---|


| Training | MFNet RGB-Thermal | [Official repository](https://github.com/haqishen/MFNet-pytorch), [project page: semantic segmentation dataset](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) | [Google Drive](https://drive.google.com/drive/folders/18BQFWRfhXzSuMloUmtiBRFrr6NSrf8Fw?usp=sharing) |


| Training | ISPRS Potsdam | [Dataset description](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx), [official download instructions](https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/default.aspx) | [Seafile](https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/); obtain the access password from the official instructions |


| Training | WHU-OPT-SAR | [Official repository and label encoding](https://github.com/AmberHen/WHU-OPT-SAR-dataset) | [Google Drive](https://drive.google.com/drive/folders/1W3iMpkehng7ADXmhz9pPvmgFQBayq22t?usp=sharing), [Baidu](https://pan.baidu.com/s/1sIGsD3lBEogSCqzbDOaclA), code `i51o` |


| Generalization testing | LLVIP | [Official download instructions](https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md) | [Google Drive](https://drive.google.com/file/d/1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1eQO1Is2NPyd-mgmv1Csbfg), code `14lc` |


| Generalization testing | M3FD | [Author repository: M3FD downloads](https://github.com/JinyuanLiu-CV/TarDAL#download) | [Google Drive](https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6?usp=sharing), [Baidu](https://pan.baidu.com/s/1GoJrrl_mn2HNQVDSUdPCrw?pwd=M3FD), code `M3FD` |





Prepare the raw downloads before placing them in the directories described above:





- MFNet images contain four channels. Separate RGB (the first three channels) from thermal infrared (the fourth), and provide the corresponding segmentation labels and official splits.


- Potsdam provides RGBIR, IRRG, and other products. RGBIR contains RGB followed by NIR; the first IRRG channel is NIR. Export the required bands from the large TIFF tiles to the supported 8-bit image format and preserve label alignment.


- The whu label values 0/10/.../70 match WHU-OPT-SAR. This dataset includes RGB, NIR, and SAR. A folder named `ir` does not establish whether an existing sample is NIR or SAR. Select and document the actual modality: NIR/RGB for infrared-visible fusion, or explicitly report SAR/optical fusion when using SAR.


- Place LLVIP visible/infrared pairs under `dataset/test/LLVIP/rgb/` and `ir/`. For M3FD, map the visible `vi` images to `dataset/test/M3FD/rgb/` and the infrared images to `ir/`, using matching filenames.


- Keep LLVIP and M3FD out of training and parameter selection when evaluating generalization. Split scenes/original tiles before cropping training images.


