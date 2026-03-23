<div align="center">

# 🚗 VehicleMemBench
### An Executable Benchmark for Multi-User Long-Term Memory in In-Vehicle Agents


[**📖 Paper**](https://arxiv.org/abs/xxxx.xxxx) | [**🚀 GitHub**](https://github.com/isyuhaochen/VehicleMemBench) | [**🤗 Dataset**](https://huggingface.co/datasets/callalilya/VehicleMemBench)

</div>

---

Official codebase of **VehicleMemBench**, a benchmark for evaluating whether agents can recover multi-user preferences from long interaction histories, resolve preference conflicts, and invoke vehicle tools to reach the correct final environment state.

<p align="center"><img src="figure/framework.png" alt="VehicleMemBench pipeline" width="100%"></p>

## 📊 Evaluation Protocols

### A. Model Evaluation

This setting evaluates the backbone model under basic memory constructions, such as:

- Raw History (`none`): Gives the entire interaction history as plain text to evaluate native long-context processing.
- Gold Memory (`gold`): Provides ground-truth latest user preferences directly, representing the theoretical performance upper bound.
- Recursive Summarization (`summary`): Compresses history into hierarchical summaries to test reasoning over distilled information.
- Key-Value Store (`key_value`): Organizes preferences into structured attribute-value pairs to assess precise, indexed retrieval.

### B. Memory-System Evaluation

This setting first ingests dialogue history into a memory system and then evaluates whether the agent can retrieve useful memory and execute the correct vehicle actions. The evaluated memory systems include:

- Gold Memory
- Recursive Summarization
- Key-Value Store
- [Memobase](https://github.com/memodb-io/memobase)
- [LightMem](https://github.com/zjunlp/LightMem)
- [Mem0](https://github.com/mem0ai/mem0)
- [MemOS](https://github.com/MemTensor/MemOS)
- [Supermemory](https://github.com/supermemoryai/supermemory)

### C. Evaluation Metrics
Main metrics used in this repo include:

- Exact State Match: A strict binary success metric that requires the final environment state of models to perfectly match the ground truth.
- Field- and Value-level Metrics: Fine-grained Precision, Recall, and F1 scores that assess whether the correct system fields were modified and if their values were predicted accurately.
- Average Prediction/Tool-call Statistics: The average number of tool calls that measures the execution cost and system overhead per task.

## 🗂️ Repository Structure

```text
VehicleMemBench/
├── benchmark/      # QA files and history logs
├── environment/    # Vehicle simulator and module definitions
├── evaluation/     # Model and memory-system evaluation code
├── figure/         # README figure assets
├── scripts/        # Example runner scripts
├── requirements.txt
└── README.md
```

## ⚙️ Setup

### 1) Environment

- Python 3.12 recommended

### 2) Install Dependencies

```bash
conda create -n VehicleMemBench python=3.12
conda activate VehicleMemBench
pip install -r requirements.txt
```

### 3) Configure API Keys

Set the base model variables before evaluation:

```bash
export LLM_API_BASE="..."
export LLM_API_KEY="..."
export LLM_MODEL="..."
```

For memory-system evaluation, set the backends you want to use:

```bash
export MEM0_API_KEY="..."
export MEMOS_API_URL="..."
export MEMOS_API_KEY="..."
export SUPERMEMORY_API_KEY="..."
export MEMOBASE_API_URL=""
export MEMOBASE_API_KEY="..."
```

## 🚀 Quick Start

Run commands from the repository root.

### A. Model Evaluation

```bash
bash scripts/model_test.sh
```

### B. Memory-System Evaluation

```bash
bash scripts/memorysystem_test.sh
```

## 🧾 Benchmark Data

Released benchmark inputs in this repo mainly include:

- long interaction histories in `benchmark/history`
- executable QA files in `benchmark/qa_data`

Typical outputs are written to `log/` or `memory_system_log/` during evaluation.

## 🙏 Acknowledgments

This project references and adapts the [`VehicleWorld`](https://github.com/OpenMOSS/VehicleWorld) repository in building the executable in-vehicle environment.

## 📝 Citation

If you use VehicleMemBench in your research, you can temporarily use the following placeholder citation until the final bibliographic entry is released:

```bibtex
@article{vehiclemembench2026,
  title={VehicleMemBench: An Executable Benchmark for Multi-User Long-Term Memory in In-Vehicle Agents},
  author={TBD},
  journal={TBD},
  year={2026}
}
```

## 📄 License

This project is licensed under the Apache-2.0 License.
