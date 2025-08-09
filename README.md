# T2I LDM Model Training 🎨

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

This repository contains a robust and feature-rich PyTorch script for training a **Text-to-Image Latent Diffusion Model (LDM)** from scratch. It's built on the Hugging Face ecosystem and designed for efficiency, scalability, and modern best practices in generative AI.

The pipeline is heavily commented (in Italian) and structured for easy modification and experimentation.

---

## ✨ Key Features

* **Core Frameworks**:
    * Built with **PyTorch** and **Hugging Face `diffusers`** for core model components (VAE, UNet, Scheduler).
    * Seamless multi-GPU and mixed-precision training powered by **`accelerate`**.
    * Uses **`transformers`** for the CLIP text encoder and tokenizer.

* **Advanced Training Techniques**:
    * **V-Prediction**: Implements the `v-prediction` objective for improved training stability.
    * **Exponential Moving Average (EMA)**: Maintains an EMA of the UNet's weights to generate higher-quality final models.
    * **Classifier-Free Guidance (CFG)**: Implements CFG during training by randomly dropping text conditioning, improving sample guidance.

* **Performance & Efficiency**:
    * 🚀 **Memory Optimization**: Integrates **`xformers`** for memory-efficient attention layers in the UNet.
    * 💾 **Mixed Precision**: Full support for `bf16` to speed up training and reduce memory footprint.
    * **Streaming Dataset**: Uses a custom `IterableDataset` to stream data directly from the Hugging Face Hub, making it perfect for massive datasets without requiring local storage.

* **Workflow & MLOps**:
    * 📊 **Experiment Tracking**: Deep integration with **Weights & Biases (`wandb`)** for logging metrics, model configuration, and final model artifacts.
    * 🔄 **Resumable Training**: Robust checkpointing that saves and loads the complete state (optimizer, scheduler, EMA model, step count), allowing you to seamlessly resume interrupted runs.
    * 🖼️ **Live Sampling**: Generates image samples with fixed prompts during training to visually monitor model progress.

---

## 🚧 Challenges & Future Improvements

This project went through several iterations to achieve a stable and efficient state. Here are some of the key challenges faced and potential upgrades for the future.

### Development Challenges

* **CPU Bottleneck in Data Pipeline**: Initially, the data preprocessing (image resizing, tokenization) created a significant CPU bottleneck, starving the GPUs of data. While using a streaming `IterableDataset` with multiple workers helped, the pipeline's throughput is still heavily dependent on CPU performance.
* **`torch.compile()` Instability**: An attempt to use `torch.compile()` for a potential speed-up led to instability. It caused CUDA graph errors when used in conjunction with other components of the training loop, forcing a fallback to the eager-mode execution to ensure stability.
* **Attention Mechanism Conflicts**: Integrating newer attention mechanisms like PyTorch 2.0's `scaled_dot_product_attention` (`F.sdpa`) proved difficult due to conflicts with other libraries like `xformers` and the `diffusers` UNet structure.
* **`xformers` Integration**: Getting `xformers` to work correctly was not straightforward. Initial attempts failed due to dependency mismatches or build errors, which is a common hurdle when setting up highly optimized training environments.

### 🌟 Future Directions

* **Pre-sharded Dataset**: To eliminate the CPU bottleneck entirely, the next step would be to pre-process and shard the dataset into a format like **WebDataset** or Parquet files.
* **Direct Preference Optimization (DPO)**: Implement DPO or similar alignment techniques to fine-tune the trained model based on human preferences for aesthetics or prompt faithfulness.
* **Adaptation for SDXL**: Upgrade the script to support training more complex architectures like **Stable Diffusion XL**, which involves handling two text encoders and a more sophisticated UNet.
* **Automated Hyperparameter Tuning**: Integrate tools like **Optuna** or **Ray Tune** to automatically search for the optimal set of hyperparameters.
* **Full Inference Pipeline**: Add a dedicated script that loads the final trained components into a standard `diffusers.StableDiffusionPipeline` for easy image generation.

---

## 🚀 Getting Started

1.  **Clone the repository and install dependencies.**
    ```bash
    git clone [https://github.com/your-username/T2I-LDM-model-training.git](https://github.com/your-username/T2I-LDM-model-training.git)
    cd T2I-LDM-model-training
    pip install -r requirements.txt
    ```

2.  **Configure `accelerate`.**
    ```bash
    accelerate config
    ```
    Set up your training environment (single GPU, multi-GPU, etc.).

3.  **Customize the Script.**
    Open `your_training_script.py` and adjust the configuration variables at the top to set your desired hyperparameters, model IDs, dataset, and output paths.

4.  **Launch Training.**
    ```bash
    accelerate launch your_training_script.py
    ```
    The script will handle the rest, from downloading models and streaming data to logging progress and saving checkpoints.
