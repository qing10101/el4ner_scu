# EL4NER: A Quantitative Implementation of Training-Free NER

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Paper](https://img.shields.io/badge/arXiv-2505.23038-b31b1b.svg)

This repository provides a faithful, step-by-step implementation and quantitative evaluation of the research paper **"EL4NER: Ensemble Learning for Named Entity Recognition via Multiple Small-Parameter Large Language Models"**.

This project not only replicates the core EL4NER pipeline but also includes a rigorous, fair comparison against a state-of-the-art standalone LLM, all designed to run on high-VRAM GPUs like the NVIDIA RTX A6000.

## ✨ Key Features

- **🧠 Ensemble Learning**: Aggregates outputs from multiple sLLMs (Phi-3, Mistral, Qwen2) to enhance accuracy and robustness.
- **🚀 Training-Free**: Relies entirely on **In-Context Learning (ICL)**, eliminating the need for expensive and time-consuming fine-tuning.
- **🔬 Fair & Quantitative Evaluation**: Includes a script to calculate Precision, Recall, and F1-score against the WNUT17 test set, ensuring all models have access to the same ICL examples.
- **🖥️ VRAM-Efficient Scripts**: Intelligently loads and unloads models sequentially, allowing complex comparisons with multiple large models to run on a single 48GB GPU.
- **📦 Professional Structure**: Code is organized into a clean, scalable Python package.

---

## ⚙️ How It Works: The EL4NER Pipeline

The entire process is designed to intelligently guide a committee of small language models to perform a specialized NER task at inference time.

```markdown

The entire process is designed to intelligently guide general-purpose sLLMs to perform a specialized NER task at inference time.

Input: "The Bush administration blamed trial lawyers..."
                           |
                           v
+--------------------------------------------------------------------------+
| 🎯 Stage 1: Demonstration Retrieval                                      |
|                                                                          |
|   Receives: `Input Text` & `Source Pool` (e.g., the WNUT17 training set)   |
|                                                                          |
|   🧠 Process: Intelligently searches the entire source pool to find the    |
|   `k` most relevant examples. Instead of comparing whole sentences, it    |
|   uniquely compares the potential entities (spans) within them, giving   |
|   more weight to important nouns.                                        |
|                                                                          |
|   📤 Output: A small, highly relevant 'cheat sheet' of `k` demonstration  |
|   sentences with their correct entities.                                 |
+--------------------------------------------------------------------------+
                           |
                           |  [ Top-k Demonstrations ]
                           v
+--------------------------------------------------------------------------+
| 🔍 Stage 2: Span Extraction (Ensemble)                                   |
|                                                                          |
|   Receives: `Input Text` & `Demonstrations`                                |
|                                                                          |
|   🧠 Process: Each sLLM (Phi-3, Mistral, Qwen2) is prompted with the         |
|   demonstrations and asked to identify all possible named entities in    |
|   the text. This leverages the unique strengths of each model.            |
|                                                                          |
|   📤 Output: A combined list (union) of all unique spans identified by   |
|   the models. This maximizes recall, ensuring no potential entity is     |
|   missed.                                                                |
+--------------------------------------------------------------------------+
                           |
                           |  [ Combined List of Spans ]
                           v
+--------------------------------------------------------------------------+
| 🏷️ Stage 3: Span Classification (Ensemble)                               |
|                                                                          |
|   Receives: `Spans` & `Demonstrations`                                     |
|                                                                          |
|   🧠 Process: The ensemble of sLLMs is prompted again, this time to assign|
|   an entity type (e.g., `person`, `location`) to each span. The final    |
|   type is chosen by a majority vote among the models.                    |
|                                                                          |
|   📤 Output: A dictionary of spans mapped to their most likely entity    |
|   types.                                                                 |
+--------------------------------------------------------------------------+
                           |
                           |  [ Classified Entities ]
                           v
+--------------------------------------------------------------------------+
| ✅ Stage 4: Type Verification                                            |
|                                                                          |
|   Receives: `Classified Entities`                                          |
|                                                                          |
|   🧠 Process: A single, designated LLM (the 'verifier') cross-checks     |
|   each result with a simple true/false question (e.g., "Is 'Bush' a      |
|   'person'?"). This acts as a crucial quality filter.                     |
|                                                                          |
|   📤 Output: A final, high-precision list of verified named entities,    |
|   with noise and incorrect classifications removed.                      |
+--------------------------------------------------------------------------+
                           |
                           v
     Final Output: {'Bush administration': 'organization', 'trial lawyers': 'person'}

---

```
## 🚀 Getting Started: A Step-by-Step Guide

Follow these steps in order to set up and run the project.

### Step 0: Prerequisites

- **Hardware:** A high-end NVIDIA GPU with **at least 24 GB of VRAM** (e.g., RTX 3090, RTX 4090, RTX A5000) is **required** to run the full evaluation with Qwen3-30B.
- **Software:** Python 3.9+ and Git.
- **Permissions:** **No root or `sudo` access is required** for any part of this process. The entire project runs in a standard user environment.

### Step 1: Clone the Repository

Clone this repository to your local machine or lab server.

```bash
git clone https://github.com/qing10101/el4ner_scu
cd el4ner_scu
```

### Step 2: Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```


### Step 4: Prepare the Dataset

This implementation is pre-configured to use the **WNUT17** dataset from Hugging Face. Run the provided script to automatically download, process, and format the data into a source pool file required by the pipeline.

This will create a `data/wnut17_source_pool.json` file.

```bash
python prepare_data.py
```

---

## 💻 Usage

All commands should be run from the project's root directory (el4ner_scu/).

### Option 1: Quantitative Evaluation (Recommended Final Goal)

This is the main script. It runs a fair, data-driven comparison against **Qwen3-30B** and calculates Precision, Recall, and F1-scores.

**How to Run:**

```bash
# Run a quick test on 10 samples to ensure everything is working
python fair_evaluation.py --num_samples 10 --output_file fair_test_run.json

# Run a full evaluation on a larger sample (e.g., 50). This will take several hours.
python fair_evaluation.py --num_samples 50 --output_file fair_evaluation_results.json```
```
The results will be printed to the console and a detailed JSON file will be saved.

### Option 2: Qualitative Side-by-Side Comparison

This script is useful for quick demos on a few specific sentences, printing a clean comparison table against **Qwen3-30B**.

**How to Run:**

```bash
python compare_ner_methods.py
```

### Option 3: Running the Core EL4NER Pipeline

If you only want to test the EL4NER pipeline itself, use this command.

**How to Run:**

```bash
python -m el4ner.main --text "Apple is set to announce the new iPhone in California next week."
```

---

## 📂 Project Structure

```
el4ner_scu/
├── data/                    # Stores the prepared dataset (created by script)
├── el4ner/                  # The core Python source package
│   ├── __init__.py
│   ├── main.py              # Entry point for the core EL4NER tool
│   ├── models.py
│   ├── pipeline.py
│   └── prompts.py
├── .gitignore
├── LICENSE
├── README.md
├── compare_ner_methods.py   # Script for qualitative demos
├── fair_evaluation.py       # Script for quantitative F1-score evaluation
├── prepare_data.py          # Script to download and format data
├── requirements.txt
└── utils.py                 # Central toolbox for shared functions
```

---

## 📜 Citation

This project is an implementation of the original research paper. If you find this code useful in your research, please consider citing the original authors.

> Yuzhen Xiao, Jiahe Song, Yongxin Xu, et al. (2025). *EL4NER: Ensemble Learning for Named Entity Recognition via Multiple Small-Parameter Large Language Models*. arXiv:2505.23038 [cs.CL].

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.