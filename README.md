# SuperMini-seg

# 0.Preface

This repo holds the implementation code of the paper.

*This repository provides code for "SuperMini-seg: An ultra lightweight network for COVID-19 lung infection segmentation from CT images" .



# 1.Introduction

If you have any questions about our paper, feel free to contact us ([Contact](#5-Contact)). 

And if you are using COVID-SemiSeg Dataset for your research, please cite this paper ([BibTeX](#4-citation)).



## 2. Proposed Methods

- **Preview:**



#### 2.1.Usage

1. Train
   - In the first step, you can directly run the  `MyTrain_LungInf_Spuermini.py` and just run it! 
2. Test
   - When training is completed, the weights will be saved in `./Snapshots/save_weights/SuperMini-Seg/`. 
   - Assign the path `--pth_path` of trained weights and `--save_path` of results save and in `MyTest_LungInf_Spuermini_for.py`.
   - Just run it and results will be saved in `./Results/Lung infection segmentation/SuperMini-Seg-'

### 2.2.  Semi-Supervised SuperMini-seg
The experimental code of semi-supervised learning is in the "SSL4MIS-COVID" directory.
#### 2.2.1. Usage

1. Train
   - Just run `/code/train_entropy_minimization_COVID_supermini.py`
   
  

# 3. Citation

Please cite our paper if you find the work useful: 

```
@article{YANG2023104896,
title = {SuperMini-seg: An ultra lightweight network for COVID-19 lung infection segmentation from CT images},
journal = {Biomedical Signal Processing and Control},
doi = {https://doi.org/10.1016/j.bspc.2023.104896},
author = {Yuan Yang and Lin Zhang and Lei Ren and Longfei Zhou and Xiaohan Wang},
}

```

## 4. Citation

Please cite this paper: 

```
@article{fan2020inf,
```

  	title={Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Images},
  	author={Fan, Deng-Ping and Zhou, Tao and Ji, Ge-Peng and Zhou, Yi and Chen, Geng and Fu, Huazhu and Shen, Jianbing and Shao, Ling},
  	journal={IEEE TMI},
  	year={2020}
	}



### 5. Contact

If you have any questions about our paper,  Feel free to email me(yangyuan@buaa.edu.cn) . 