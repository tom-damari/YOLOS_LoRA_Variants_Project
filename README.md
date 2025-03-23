# Enhancing Object Detection in YOLOS-Small through Advanced LoRA Methods

## Overview

This project examines the integration of Low-Rank Adaptation (LoRA) methods into YOLOS-Small, a Vision Transformer-based object detection model. The goal is to explore whether fine-tuning techniques can improve computational efficiency while maintaining detection accuracy on the COCO 2017 dataset.

The YOLOS-Small model, sourced from [Hugging Face's YOLOS-Small repository](https://huggingface.co/hustvl/yolos-small), was used as the baseline. The study evaluates various LoRA-based approaches, including LoRA, AdaLoRA, LoHa, and LoKr, to determine their impact on model performance.

This project was developed as part of the Deep Learning course at Ben-Gurion University (2025).

## Dataset

- The dataset used in this project is COCO Validation 2017, accessed through Hugging Face Datasets (`detection-datasets/coco`).
- The dataset underwent preprocessing using the YOLOS image processor to ensure compatibility with the model.
- A subset of 2,100 training and validation images and 900 test images was selected for experimentation.

## LoRA Variants Implemented

| LoRA Variant | Rank (r) | Scaling (α) | Dropout | Special Features        |
| ------------ | -------- | ----------- | ------- | ----------------------- |
| LoRA         | 16       | 8           | 0.5     | Standard LoRA           |
| AdaLoRA      | 16 → 12  | Adaptive    | None    | Dynamic rank allocation |
| LoHa         | 12       | 8           | 0.3     | Rank + module dropout   |
| LoKr         | 16       | 8           | 0.4     | Kronecker decomposition |

## Evaluation & Results

The results indicate that the improvements in mAP over the baseline YOLOS-S model were minimal across all fine-tuned models. This suggests that while LoRA-based methods provide some benefits, their impact on YOLOS-S performance remains limited under the current experimental settings.
The following table summarizes the overall mAP results across different confidence thresholds:

| Model                  | mAP@50 | mAP@75 | mAP@[0.5:0.95] |
|------------------------|--------|--------|----------------|
| Baseline YOLOS-S       | 51.2714 | 34.2762 | 33.2379       |
| YOLOS-S with LoRA      | 51.6249 | 34.3871 | 33.3389       |
| YOLOS-S with AdaLoRA   | 51.5363 | 34.4044 | **33.3447**   |
| YOLOS-S with LoHa      | **51.6466** | 34.2496 | 33.2727       |
| YOLOS-S with LoKr      | 51.4323 | **34.5513** | 33.3272       |

Despite the marginal differences, AdaLoRA achieved the highest overall mAP@[0.5:0.95], indicating that its adaptive rank allocation may contribute to a slight enhancement in detection accuracy. However, detecting small objects remains a challenge across all fine-tuned models, with no significant improvements in mAP for small-object categories. 

A more significant improvement may require additional optimizations, such as configuration tuning, loss balancing, and model integration, to enhance adaptation and performance. 

## Project Files

This repository contains the following files and directories:

- **`Enhancing_YOLOS_S_through_Advanced_LoRA_Methods.py`** – The main script containing the implementation of the LoRA-based YOLOS-Small fine-tuning. This script is fully runnable and includes all necessary steps from data preprocessing to model training and evaluation.
- **`Models with LoRA Variants/`** – Contains the trained YOLOS-Small models after fine-tuning:
  - `lora_model.pth`
  - `adalora_model.pth`
  - `loha_model.pth`
  - `lokr_model.pth`
  - These models can be loaded directly in the corresponding sections of the main script.
- **`Visualizations/`** – Graphs and visualizations generated during training and evaluation.
- **`Enhancing YOLOS-S through Advanced LoRA Methods.pdf`** – The final project report summarizing the findings.
- **`Part1_setup_and_data_understanding.ipynb` → `Part7_models_comparison.ipynb`**  
  - These **seven Jupyter Notebook files** are structured **for display purposes only** and include outputs of different steps in the process.  
  - They **are not independently runnable**, as each part depends on the previous ones.  
  - The full executable code is available in **`Enhancing_YOLOS_S_through_Advanced_LoRA_Methods.py`**.


## Authors

- Orin Cohen ([orincoh@post.bgu.ac.il](mailto:orincoh@post.bgu.ac.il))
- Tom Damari ([damarit@post.bgu.ac.il](mailto:damarit@post.bgu.ac.il))
