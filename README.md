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

This setting evaluates the backbone model under different memory constructions such as:

- `none`
- `gold`
- `summary`
- `key_value`

### B. Memory-System Evaluation

This setting first ingests dialogue history into a memory system and then evaluates whether the agent can retrieve useful memory and execute the correct vehicle actions.

Main metrics used in this repo include:

- `Exact State Match`
- field-level and value-level metrics
- average prediction/tool-call statistics
- output token statistics

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
