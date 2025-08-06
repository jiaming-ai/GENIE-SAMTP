# SAM for Traversability Prediction (SAM-TP)

This repository provides inference and visualization code for **SAM-TP**, a customized version of SAM2 tailored for pixel-wise traversability prediction. SAM-TP is the perception core of [GeNIE: A Generalizable Navigation System for In-the-Wild Environments](https://arxiv.org/abs/2506.17960), where it enables robust terrain understanding across diverse environments.


### 📄 Paper

**GeNIE: A Generalizable Navigation System for In-the-Wild Environments**

Jiaming Wang*, Diwen Liu*, Jizhuo Chen*, Jiaxuan Da, Nuowen Qian, Tram Minh Man, [Harold Soh](https://haroldsoh.com/)

---

## 🔧 Setup

### 1. Create a virtual environment or activate an existing one

```bash
# create a new one
conda env create -n sam_tp python=3.10
# or activate the existing one
conda activate sam_tp
```

### 2. Make sure PyTorch is installed.
If it's not installed in the virtual environment, you should install it according to your cuda version.
For example, for the **nightly build with CUDA 12.8**, run:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

> 🔗 Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) to find the correct install command for your setup.

### 3. Install the SAM-TP package

```bash
pip install git+https://github.com/StevenLiudw/SAM-TP-GENIE

# or git clone the code and install it with editable mode
git clone 
pip install -e.
```

---

## 📦 Model Configuration and Checkpoint

The script uses the following paths:

* Model config:
  `sam2/configs/sam2.1_inference_tiny/sam2.1_custom2.yaml`
* Checkpoint:
  `sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/checkpoint_2.pt`
  
🔔 Note: The checkpoint file is not included in the repository.
To use the model, please download the checkpoint manually from the link below and place it in the expected directory.

### 📥 Download Checkpoint

Manually download the model checkpoint from:

🔗 [Google Drive – SAM2 Checkpoint](https://drive.google.com/drive/folders/190yHH-TcfQVoByZeB1809sPIR62CsBD1?dmr=1&ec=wgc-drive-hero-goto)

Then place the `checkpoint_2.pt` file at the following location:

```
sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/checkpoint_2.pt
```

---

### ✅ Usage

#### To use it in your code
```python
from sam2.sam_tp import SAM_TP

model_cfg = "path/to/config.yaml"
checkpoint = "path/to/checkpoint.pt"

sam_tp = SAM_TP(model_cfg, checkpoint)
image = ...  # your (H, W, 3) RGB image as a NumPy array

result = sam_tp.run_sam2_inference(image)

heatmap = result["heatmap"]        # heatmap for visualization
score_map = result["logits"]       # (H, W) raw logits
```

#### To test in CLI

```bash
python visualize_heatmap.py \
  --input_path /path/to/image.jpg \
  --output_dir /path/to/output/
```

* `--input_path`: Path to the input image (JPG or PNG)
* `--output_dir`: Directory where the output heatmap and score map will be saved

---

### 🧪 Example Inference Output

We run **SAM-TP** on an input image to predict traversable areas. The result is visualized as a color-coded heatmap, where:

> 🔴 **Red** regions are **easier to navigate**
> 🔵 **Blue** regions are **harder to navigate**

<div align="center">

<table>
  <tr>
    <td align="center"><strong>Input Image</strong></td>
    <td align="center"><strong>Traversability Heatmap</strong></td>
  </tr>
  <tr>
    <td><img src="example/0001.jpeg" width="300"></td>
    <td><img src="example/heatmap.jpg" width="300"></td>
  </tr>
</table>

</div>

---


### 🔖 Citation

If you use **SAM-TP** or the **GeNIE system** in your research, please cite our paper:

**[GeNIE: A Generalizable Navigation System for In-the-Wild Environments](https://arxiv.org/abs/2506.17960)**
Jiaming Wang\*, Diwen Liu\*, Jizhuo Chen\*, Jiaxuan Da, Nuowen Qian, Tram Minh Man, Harold Soh

```bibtex
@article{wang2024genie,
  title={GeNIE: A Generalizable Navigation System for In-the-Wild Environments},
  author={Wang, Jiaming and Liu, Diwen and Chen, Jizhuo and Da, Jiaxuan and Qian, Nuowen and Man, Tram Minh and Soh, Harold},
  journal={arXiv preprint arXiv:2506.17960},
  year={2024}
}
```


---
