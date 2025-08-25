# ConvTransformer variants — My Implementation

> Implementation of **Transformers with convolutional context for ASR** for end-to-end speech recognition:
> 📄 Paper : https://arxiv.org/pdf/1904.11660

## 🔗 Repository
- GitHub: https://github.com/itsmekhoathekid/ConvTransformer

## 🚀 Quickstart

### 1) Clone & Setup
```
bash
git clone https://github.com/itsmekhoathekid/ConvTransformer
cd ConvTransformer
```

### 2) Download & Prepare Dataset
This will download the datasets configured inside the script and generate manifests/features as needed.
```
bash
bash ./prep_data.sh
```

### 3) Train
Train with a YAML/JSON config of your choice.
```
bash
python train.py --config path/to/train_config.yaml
```

### 4) Inference (example)
```
bash
python infererence.py --config path/to/train_config.yaml --epoch num_epoch
```

## 📦 Project Layout (typical)
```
Conformer/
├── prep_data.sh                 # dataset download & preprocessing
├── train.py                     # training entry point
├── inference.py                     # inference script (optional)
├── configs/                     # training configs (yaml/json)
├── models/                    # model, losses, data, utils
│   ├── model.py
│   ├── encoder.py
│   ├── decoder.py
│   └── ...
├── utils/ 
└── README.md
```

