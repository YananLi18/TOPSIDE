# TOPSIDE

TOPSIDE, a novel meta-learning-based cloud-edge collaborative QoE estimation framework, solves several practical challenges in real-world scenarios.
The core idea behind TOPSIDE is to skillfully apply meta-learning and supervised contrastive regression loss to a layer-aware encoder, to achieve an effective, efficient, and adaptive data preprocessing in an automated manner.
Specifically, we aim to solve the following practical challenges:
1. Generalization capabilities
2. Interpretability
3. Accuracy and time efficiency
4. Robustness
5. Scalability

We provide the artifact for our paper, including:
- Setup
- Datasets
- Experiments

## 1. Setup

Required software dependencies are listed below:
```bash
catboost==1.2.2
matplotlib==3.7.3
numpy==1.24.4
pandas==1.2.5
scikit-learn==1.3.2
scipy==1.10.1
python==3.8.8
pytorch==1.8.1
xgboost==2.0.1
```
The installation of GPU version LighGBM refers to this [link](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#build-cuda-version) for more details.

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## 2. Datasets
The datasets for comparison consist of:
- **SNESet**: The first large-scale QoS and QoE dataset on public edge platform-Alibaba ENS. Refer to [[paper]](https://xumengwei.github.io/files/SIGMOD24-EdgeQoE.pdf) and [[dataset]](https://github.com/YananLi18/SNESet) for more details.
Our cleaned version is `<repo>/datasets/training_2nd_dataset_1.csv`.
- **Huawei Dataset**: A QoS and QoE dataset was collected from a simulated testbed. See http://jeremie.leguay.free.fr/qoe/index.html for more details. Our cleaned version is `<repo>/datasets/ICC_cleaned.csv`.

## 3. Experiments
<!-- Our meta-training pipeline is built on https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch and https://github.com/kaiwenzha/Rank-N-Contrast.   -->

- To train corresponding models in the cloud-only paradigm, run:
```bash
cd <repo>/baselines/cloud
python <model file> --name_of_args_json_file XXX 
```

- To train corresponding models in the edge-only paradigm, run:
```bash
cd <repo>/baselines/edge
python <model file> --name_of_args_json_file XXX 
```

- To train corresponding models in our cloud-edge collaborative paradigm, run:
```bash
cd <repo>/baselines/collaborative
python <model file> --name_of_args_json_file XXX --top_n_models 5 
```

<!-- ## Citation
If you use this code for your research, please cite our paper:
```bibtex
@inproceedings{zha2023rank,
    title={Rank-N-Contrast: Learning Continuous Representations for Regression},
    author={Zha, Kaiwen and Cao, Peng and Son, Jeany and Yang, Yuzhe and Katabi, Dina},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
}
``` -->