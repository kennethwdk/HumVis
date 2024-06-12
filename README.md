# HumVis: Human-Centric Visual Analysis System

[[`Paper`](https://dl.acm.org/doi/pdf/10.1145/3581783.3612663)]

> [HumVis: Human-Centric Visual Analysis System](https://dl.acm.org/doi/10.1145/3581783.3612663)  
> Dongkai Wang, Shiliang Zhang, Yaowei Wang, Yonghong Tian, Tiejun Huang, Wen Gao  
> ACM MM 2023 *Demo*

## Installation
Please refer to INSTALL.md

## Usage

Clone model weights by
```shell
    git lfs install
    git clone https://huggingface.co/d0ntcare/HumVis
    mv HumVis model_files
```

Run demo by

```python
CUDA_VISIBLE_DEVICES=0 python app.py
```

## Citations
If you find this code useful for your research, please cite our paper:

```
@inproceedings{10.1145/3581783.3612663,
author = {Wang, Dongkai and Zhang, Shiliang and Wang, Yaowei and Tian, Yonghong and Huang, Tiejun and Gao, Wen},
title = {HumVis: Human-Centric Visual Analysis System},
year = {2023},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {9396–9398}
}
```

```
@ARTICLE{10040902,
  author={Wang, Dongkai and Zhang, Shiliang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Contextual Instance Decoupling for Instance-Level Human Analysis}, 
  year={2023},
  volume={45},
  number={8},
  pages={9520-9533},
  doi={10.1109/TPAMI.2023.3243223}}
```