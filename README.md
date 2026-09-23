# Preference-Aligned TinyLlama for Pharmaceutical Domain QA

A step-by-step training pipeline for adapting **TinyLlama 1.1B** to pharmaceutical-domain question answering using:

**Base Model → Domain Adaptation → Instruction Fine-Tuning → Preference Alignment with DPO**

The project explores how a small language model can progressively improve its ability to understand pharmaceutical content, follow instructions, and produce preferred responses through **LoRA/PEFT** and **Direct Preference Optimization (DPO)**.

---

## 📌 Project Overview

Large language models are powerful but can be expensive to train and deploy. This project experiments with a lightweight **TinyLlama 1.1B** model and progressively adapts it to pharmaceutical-domain tasks.

The training pipeline consists of multiple stages:

```text
TinyLlama 1.1B
      │
      ▼
Domain Adaptation
(LoRA / PEFT)
      │
      ▼
Instruction Fine-Tuning
(New LoRA)
      │
      ▼
Preference Alignment
(DPO + New LoRA)
      │
      ▼
Preference-Aligned Model
```

Each stage serves a different purpose:

| Stage                   | Purpose                                                  |
| ----------------------- | -------------------------------------------------------- |
| Base Model              | Original pretrained TinyLlama                            |
| Domain Adaptation       | Expose the model to pharmaceutical-domain knowledge      |
| Instruction Fine-Tuning | Teach the model to respond to instructions and questions |
| Preference Alignment    | Optimize the model toward preferred responses using DPO  |

The project specifically uses **LoRA adapters** to reduce the number of trainable parameters and make fine-tuning more practical on limited GPU resources.

---

# 🎯 Objectives

The main objectives of this project are to:

* Adapt TinyLlama to pharmaceutical-domain content.
* Improve its ability to answer domain-specific questions.
* Teach the model to follow instruction-style prompts.
* Apply preference alignment using **Direct Preference Optimization (DPO)**.
* Compare the behavior of:

  * Non-instruction/domain-adapted model
  * Instruction-fine-tuned model
  * DPO preference-aligned model
* Experiment with parameter-efficient fine-tuning using **LoRA/PEFT**.
* Understand how different training stages affect model behavior.

---

# 🧠 Model

This project uses:

**TinyLlama 1.1B**

```text
TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```

TinyLlama is a relatively small causal language model with approximately **1.1 billion parameters**, making it useful for experimentation with fine-tuning and alignment on limited hardware.

---

# 🔄 Training Pipeline

## Stage 1 — Base Model

The starting point is the pretrained TinyLlama model.

```text
TinyLlama 1.1B
```

At this stage, the model has general language knowledge but has not been specifically adapted to the pharmaceutical domain.

The tokenizer is loaded from the same base model.

```python
tokenizer = AutoTokenizer.from_pretrained(base_model)
```

If a padding token is not available, the EOS token is used as the padding token:

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

---

# Stage 2 — Pharmaceutical Domain Adaptation

The first fine-tuning stage adapts the model toward pharmaceutical content.

The project uses **LoRA (Low-Rank Adaptation)** through the PEFT library.

LoRA avoids updating all parameters of the original language model. Instead, small trainable adapter matrices are added to selected model layers.

Conceptually:

```text
Original Model
      │
      ├── Frozen parameters
      │
      └── Trainable LoRA adapters
```

This significantly reduces the number of parameters that need to be updated during training.

The resulting model is referred to in the project as the **non-instruction model**.

---

# Stage 3 — Instruction Fine-Tuning

After domain adaptation, the model is further trained to better follow instructions and answer pharmaceutical questions.

The important part of the pipeline is that the previous LoRA adapter is **merged into the base model before attaching a new LoRA adapter**.

The process is:

```text
Base Model
    +
Stage 1 LoRA
    │
    ▼
Merge LoRA
    │
    ▼
Instruction-Tuned Base
    +
Stage 2 LoRA
```

The instruction checkpoint is loaded using:

```python
model = PeftModel.from_pretrained(
    model,
    instruction_checkpoint
)

model = model.merge_and_unload()
```

A new LoRA adapter can then be attached for the next training stage.

### Why merge before adding another LoRA?

The goal is to avoid continuously stacking adapters on top of previous adapters.

Instead of:

```text
Base
 ↓
LoRA
 ↓
LoRA on LoRA
 ↓
LoRA on LoRA on LoRA
```

the pipeline follows:

```text
Base
 +
Merged Stage 1 LoRA
 ↓
New Base State
 +
New LoRA
```

This makes each training stage easier to manage and keeps the adapter structure cleaner.

---

# Stage 4 — Preference Alignment with DPO

The final training stage applies **Direct Preference Optimization (DPO)**.

DPO is used to train the model using preference data containing a preferred response and an alternative response.

Conceptually:

```text
Question / Prompt
       │
       ├──────────────► Preferred Response
       │
       └──────────────► Less Preferred Response
                              │
                              ▼
                         DPO Training
                              │
                              ▼
                    Preference-Aligned Model
```

Instead of explicitly training a separate reward model, DPO directly optimizes the language model using preference comparisons.

The project uses the TRL library:

```python
from trl import DPOTrainer, DPOConfig
```

---

# 📊 Preference Dataset

The preference dataset is loaded from:

```text
pharma_preference_data.csv
```

using Hugging Face Datasets:

```python
dataset = load_dataset(
    "csv",
    data_files="/content/pharma_preference_data.csv"
)["train"]
```

The dataset is expected to contain preference-based examples suitable for DPO training.

A typical DPO example conceptually looks like:

```text
Prompt:
Explain how Metformin works in the human body.

Chosen:
A clear, accurate, well-structured pharmaceutical explanation.

Rejected:
An incomplete, inaccurate, or less useful explanation.
```

The actual dataset should follow the format expected by the installed TRL/DPOTrainer version.

---

# ⚙️ LoRA Configuration

The project uses the following LoRA configuration:

```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)
```

### Configuration

| Parameter        |              Value | Purpose                                |
| ---------------- | -----------------: | -------------------------------------- |
| `r`              |                  8 | Rank of LoRA matrices                  |
| `lora_alpha`     |                 16 | LoRA scaling factor                    |
| `lora_dropout`   |               0.05 | Dropout applied to LoRA layers         |
| `target_modules` | `q_proj`, `v_proj` | Attention projections modified by LoRA |
| `bias`           |             `none` | No bias parameters are trained         |
| `task_type`      |        `CAUSAL_LM` | Causal language modeling               |

The project focuses LoRA on the **query and value projections** of the attention layers.

---

# 🏋️ DPO Training Configuration

The DPO training configuration is:

```python
DPOConfig(
    output_dir="./tinyllama-preference-alignment",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    beta=0.1,
    report_to=None,
    logging_dir=None,
    loss_type="sigmoid",
    remove_unused_columns=False
)
```

### Training Parameters

| Parameter             |             Value |
| --------------------- | ----------------: |
| Learning Rate         |            `2e-5` |
| Batch Size            |               `1` |
| Gradient Accumulation |               `8` |
| Effective Batch Size  | Approximately `8` |
| Epochs                |               `1` |
| DPO Beta              |             `0.1` |
| Loss                  |           Sigmoid |
| W&B Logging           |          Disabled |

The small batch size combined with gradient accumulation helps reduce GPU memory requirements.

---

# 🔧 Technologies Used

## Machine Learning

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* LoRA
* TRL
* DPO
* BitsAndBytes

## Model

* TinyLlama 1.1B

## Training Techniques

* Parameter-Efficient Fine-Tuning (PEFT)
* LoRA
* Instruction Fine-Tuning
* Direct Preference Optimization

---

# 📁 Project Structure

A recommended project structure is:

```text
preference-aligned-tinyllama/
│
├── README.md
│
├── notebooks/
│   └── Preference_Aligned_Training_DPO_final.ipynb
│
├── data/
│   └── pharma_preference_data.csv
│
├── models/
│   ├── non-instruction/
│   ├── instruction/
│   └── preference-aligned/
│
├── requirements.txt
│
└── results/
    └── evaluation/
```

The current notebook uses paths such as:

```text
/content/checkpoint-3
/content/checkpoint-5
/content/tinyllama-preference-alignment/checkpoint-1
```

These are Google Colab-style paths and should be adjusted when running the project locally.

---

# 🚀 Installation

Install the required libraries:

```bash
pip install -U transformers datasets peft trl bitsandbytes accelerate torch
```

For the notebook workflow, the following packages are particularly important:

```text
transformers
datasets
peft
trl
bitsandbytes
torch
accelerate
```

---

# ▶️ Running the Project

## 1. Load the Base Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)
```

---

## 2. Load the Instruction-Fine-Tuned Model

After the instruction fine-tuning stage, load the generated checkpoint:

```python
instruction_model = AutoModelForCausalLM.from_pretrained(
    "/content/checkpoint-3",
    device_map="auto"
)
```

You can then test it with a pharmaceutical question:

```python
question = """
Explain how Metformin works in the human body and why some researchers
believe it could have benefits beyond diabetes treatment.
"""
```

---

## 3. Run DPO Training

The preference dataset is loaded:

```python
dataset = load_dataset(
    "csv",
    data_files="/content/pharma_preference_data.csv"
)["train"]
```

The instruction-trained model is merged and a fresh LoRA adapter is attached:

```text
Base Model
    ↓
Instruction LoRA
    ↓
Merge
    ↓
New LoRA
    ↓
DPO Training
```

Then training is started:

```python
trainer.train()
```

The resulting model is saved under:

```text
./tinyllama-preference-alignment/
```

---

# 🧪 Model Comparison

The project evaluates three versions of the model using the same pharmaceutical question.

## 1. Non-Instruction Model

```text
checkpoint-5
```

This represents the earlier domain-adapted model.

---

## 2. Instruction-Fine-Tuned Model

```text
checkpoint-3
```

This represents the model after instruction fine-tuning.

---

## 3. DPO Preference-Aligned Model

```text
tinyllama-preference-alignment/checkpoint-1
```

This represents the model after preference optimization using DPO.

---

# 🔬 Example Evaluation Prompt

The models are tested using:

```text
Explain how Metformin works in the human body and why some researchers
believe it could have benefits beyond diabetes treatment.
```

Each model receives the same prompt and generates a response.

This allows qualitative comparison of:

* Instruction following
* Response structure
* Relevance
* Pharmaceutical terminology
* Completeness
* Clarity
* Response quality

A stronger evaluation should additionally use a fixed evaluation dataset and quantitative metrics rather than relying only on a single example.

---

# 📈 Evaluation Strategy

The current notebook primarily demonstrates **model generation and qualitative comparison** between training stages.

For a more rigorous evaluation, the following can be measured:

### Relevance

Does the generated answer directly address the question?

### Accuracy

Does the answer contain scientifically and medically correct information?

### Completeness

Does the response cover the important aspects of the question?

### Instruction Following

Does the model follow the requested format, scope, and task?

### Preference Win Rate

For a collection of prompts, compare responses from different model versions and measure how frequently one model's response is preferred.

### Human Evaluation

Domain experts can evaluate responses for:

* Factual correctness
* Scientific quality
* Clarity
* Completeness
* Safety
* Usefulness

For pharmaceutical or medical applications, expert review is particularly important because fluent language does not guarantee factual correctness.

---

# 💾 Loading the Preference-Aligned Model

After DPO training:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./tinyllama-preference-alignment/checkpoint-1"

tokenizer = AutoTokenizer.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16
)

model.to("cuda")
```

Generate a response:

```python
inputs = tokenizer(
    question,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1
)

print(tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
))
```

---

# 🧩 Important Implementation Detail: LoRA Stages

One of the important design decisions in this project is how LoRA adapters are handled between stages.

The intended workflow is:

```text
                 ┌─────────────────────┐
                 │   TinyLlama 1.1B    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Domain Adaptation   │
                 │     + LoRA #1       │
                 └──────────┬──────────┘
                            │
                         Merge
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Instruction Tuning  │
                 │     + LoRA #2       │
                 └──────────┬──────────┘
                            │
                         Merge
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Preference Tuning   │
                 │     + LoRA #3       │
                 │        DPO           │
                 └─────────────────────┘
```

The key principle is:

> **Merge the previous adapter before attaching a new adapter for the next training stage.**

This avoids unnecessarily creating a chain of nested LoRA adapters.

---

# ⚠️ Limitations

This project is an experimental fine-tuning and preference-alignment pipeline rather than a production pharmaceutical or medical system.

Important limitations include:

* TinyLlama 1.1B has limited model capacity compared with larger language models.
* The quality of the final model depends heavily on the quality of the training and preference datasets.
* DPO does not guarantee factual correctness.
* Preference optimization can improve response style without necessarily improving domain knowledge.
* The current notebook does not provide a comprehensive quantitative benchmark.
* Testing a model with a single prompt is not sufficient to establish overall model improvement.
* Pharmaceutical and medical information requires careful validation before real-world use.
* Model outputs should not be treated as professional medical advice.

---

# 🔮 Future Improvements

Possible improvements include:

### 1. Build a Proper Evaluation Dataset

Create a held-out pharmaceutical question-answer dataset that is never used during training.

### 2. Add Quantitative Evaluation

Measure:

```text
Accuracy
Relevance
Completeness
Preference Win Rate
Perplexity
```

and, where appropriate, use LLM-based evaluation with human validation.

### 3. Experiment with DPO Hyperparameters

Test different values of:

```text
beta
learning rate
LoRA rank
LoRA alpha
batch size
number of epochs
```

### 4. Compare LoRA Configurations

Experiment with different target modules:

```text
q_proj
v_proj
k_proj
o_proj
```

and compare their effect on model performance.

### 5. Compare Different Alignment Methods

Future experiments could compare DPO with other preference-optimization approaches such as:

* ORPO
* SimPO
* GRPO
* Reward-model-based approaches

### 6. Add Automated Evaluation

Integrate evaluation frameworks to automatically measure response quality across a fixed test set.

### 7. Refactor the Notebook

The current workflow can eventually be converted into reusable modules:

```text
src/
├── data.py
├── model.py
├── training.py
├── dpo.py
└── evaluation.py
```

This would make experiments easier to reproduce and maintain.

---

# 📚 Key Concepts Demonstrated

This project demonstrates practical experience with:

* Causal Language Models
* Domain Adaptation
* Instruction Fine-Tuning
* Parameter-Efficient Fine-Tuning
* LoRA
* PEFT
* Model checkpoint management
* Adapter merging
* Preference datasets
* Direct Preference Optimization
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face TRL
* GPU-aware model loading
* 8-bit model loading
* Model generation and comparison

---

# 🏁 Final Pipeline

The complete workflow can be summarized as:

```text
                 TinyLlama 1.1B
                       │
                       ▼
             Pharmaceutical Data
                       │
                       ▼
              LoRA Domain Tuning
                       │
                       ▼
                 Merge LoRA
                       │
                       ▼
            Instruction Fine-Tuning
                       │
                       ▼
                 Merge LoRA
                       │
                       ▼
             Preference Dataset
                       │
                       ▼
                  DPO + LoRA
                       │
                       ▼
          Preference-Aligned TinyLlama
                       │
                       ▼
             Evaluation & Comparison
```

---

# 👤 Author

**Abdullah Kamal**

AI Engineer | GenAI & LLM Systems

Focused on:

```text
LLMs • RAG • AI Agents • Fine-Tuning
Python • LangChain • LangGraph • FastAPI
PEFT • LoRA • QLoRA • DPO
```

---

# 📄 License

This repository is intended for educational and research purposes. Check the licenses of TinyLlama and all third-party libraries and datasets before redistributing or using the resulting model commercially.
