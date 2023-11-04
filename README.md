# [FastHuman: Reconstructing High-Quality Clothed Human in Minutes](https://arxiv.org/abs/2211.14485)

https://user-images.githubusercontent.com/25956606/203796037-a64faa77-a84b-497c-b2d3-9f65a748abfc.mp4


# Installation

clone the repository
```bash
git clone https://github.com/l1346792580123/FastHuman.git
cd FastHuman
```
Step 1: requirements:
```bash
pip install -r requirements.txt
```

Step 2: install PyTorch (The PyTorch version should be higher than 1.7.1.):

```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Step 3: install nvdiffrast
```bash
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
```

## Data Preparation

You can download NHR data and DTU data from NHR(https://wuminye.github.io/NHR/datasets.html) and DTU(https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhangbs_connect_ust_hk/EazyGuwPC5tIkbI3fgeERgEBBUXBV16_hIkud_dhS34wVw?e=CWjJGP) respectively.



# Reconstruction Results of [NHR dataset](https://wuminye.github.io/NHR/)


[//]: https://user-images.githubusercontent.com/25956606/203795898-6b40fb93-7873-4d4f-b93d-66b51fa0cfe9.mp4


[//]: https://user-images.githubusercontent.com/25956606/203795977-98f697ec-96bc-46e5-a40c-7f6eff6d00ec.mp4



https://user-images.githubusercontent.com/25956606/203796001-7b143df8-02ea-47a4-988b-eaf3d0382e19.mp4




https://user-images.githubusercontent.com/25956606/203796021-b0a99641-df83-4fef-ab84-3ef259ff4052.mp4




## Citation
```
@inproceedings{fasthuman,
  author={Lin, Lixiang, Peng Songyou, Gan Qijun and Zhu, Jianke},
  booktitle={International Conference on 3D Vision, 3DV}, 
  title={FastHuman: Reconstructing High-Quality Clothed Human in Minutes}, 
  year={2024},
  }
```

# Related Works

[Multiview Textured Mesh Recovery by Differentiable Rendering](https://github.com/l1346792580123/diff)

