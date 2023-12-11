# TOPSIDE

TOPSIDE, a novel cloud-edge collaborative QoE estimation framework, solves several practical challenges in real-world scenarios.
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

The dataset we use is **SNESet**, which is located in  `<repo>/datasets/training_2nd_dataset_1.csv`.
For more details of data collection, refer to the SIGMOD'24 [[paper]](https://xumengwei.github.io/files/SIGMOD24-EdgeQoE.pdf) [[dataset]](https://github.com/YananLi18/SNESet).
