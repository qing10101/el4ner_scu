### `README.md`

# EL4NER: A Lightweight, Training-Free NER Pipeline

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Paper](https://img.shields.io/badge/arXiv-2505.23038-b31b1b.svg)

This repository provides a faithful implementation of the research paper **"EL4NER: Ensemble Learning for Named Entity Recognition via Multiple Small-Parameter Large Language Models"**. It showcases a novel, training-free approach to NER that leverages the collective intelligence of multiple open-source small LLMs (sLLMs) to achieve state-of-the-art performance with high parameter efficiency.

## ✨ Key Features

- **🧠 Ensemble Learning**: Aggregates outputs from multiple sLLMs (Phi-3, GLM-4, Qwen2) to enhance accuracy and robustness.
- **🚀 Training-Free**: Relies entirely on **In-Context Learning (ICL)**, eliminating the need for expensive and time-consuming fine-tuning.
- **🎯 Advanced Retrieval**: Implements a novel span-level sentence similarity algorithm to find the most relevant examples for ICL prompts.
- **🔍 Task Decomposition**: Breaks down the complex NER task into a four-stage pipeline: Demonstration Retrieval, Span Extraction, Span Classification, and Type Verification.
- ** modular and Customizable**: Easily configure the number of demonstrations, the verifier model, and the input data.

---

## ⚙️ How It Works: The EL4NER Pipeline

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
|   🧠 Process: Each sLLM (Phi-3, GLM-4, Qwen2) is prompted with the         |
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
### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

This implementation is pre-configured to use the **WNUT17** dataset from Hugging Face. Run the provided script to automatically download, process, and format the data into a source pool file required by the pipeline.

This will create a `data/wnut17_source_pool.json` file.

```bash
python prepare_data.py
```

---

## 💻 Usage

You can run the full NER pipeline on any text using the `main.py` script.

### Quick Start

Here's how to run the pipeline on the example text from the paper:

Go to the el4ner-implementation folder.
```bash
python -m el4ner.main --text "The Bush administration blamed trial lawyers for undermining their authority in the country."
```

### Command-Line Arguments

Customize the pipeline's behavior with these arguments:

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--text` | **(Required)** The input text string to perform NER on. | `None` |
| `--data_path`| Path to the JSON source pool file for demonstration retrieval. | `data/wnut17_source_pool.json` |
| `--k` | The number of demonstrations to retrieve from the source pool. | `5` |
| `--verifier` | The backbone model to use for the final verification step. | `glm` |

**Example with custom arguments:**

```bash
python -m el4ner.main \
  --text "Apple is set to announce the new iPhone in California next week." \
  --k 10 \
  --verifier phi
```

---

## 📂 Project Structure

```
el4ner_scu/          <-- This is your main project folder and Git repository root
├── .git/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── prepare_data.py      <-- A utility script can live at the root
└── el4ner/              <-- This is your source code package
    ├── __init__.py      <-- Makes this a Python package (can be empty)
    ├── main.py
    ├── models.py
    ├── pipeline.py
    └── prompts.py
```

---

## 📜 Citation

This project is an implementation of the original research paper. If you find this code useful in your research, please consider citing the original authors.

> Yuzhen Xiao, Jiahe Song, Yongxin Xu, et al. (2025). *EL4NER: Ensemble Learning for Named Entity Recognition via Multiple Small-Parameter Large Language Models*. arXiv:2505.23038 [cs.CL].

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```