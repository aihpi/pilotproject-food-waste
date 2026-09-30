# Food Waste Project

Estimate how much of each ingredient comes back uneaten from a single photo of a plate after the meal.

This repository has the code to build the dataset, fine-tune a vision-language model on it, and evaluate it:

- **Dataset**: [AI-ServicesBB/food-waste-dataset](https://huggingface.co/datasets/AI-ServicesBB/food-waste-dataset) on Hugging Face
- **Model**: [aihpi/food-waste-vlm](https://huggingface.co/aihpi/food-waste-vlm), a LoRA fine-tune of Qwen2-VL-7B-Instruct

## 🍽️ Dataset

Photos of plates **after** a meal, each labelled with how many grams of every ingredient came back uneaten. The dataset was built to train and evaluate models that estimate plate waste from a single photo.

All meals were served and photographed in a test kitchen. Each receipt (`bonid`) is one served meal, photographed 12–40 times after eating. There are no before-meal photos.

| Split | Images | Receipts (`bonid`) | Photos per receipt |
|---|---|---|---|
| train | 215 | 12 | 12–23 |
| test | 160 | 4 | 40 |

The [dataset card](https://huggingface.co/datasets/AI-ServicesBB/food-waste-dataset) is the reference for the fields, the prediction target, known issues, licence and citation.

## 🗂️ Repository structure

| Path | What it does |
|---|---|
| `dataset/data_processor.py` | Joins the kitchen's CSV exports with the plate photos and uploads the result to Hugging Face. |
| `baseline/sharegpt.py` | Downloads the dataset and converts it into ShareGPT conversations for fine-tuning (`food_conversations.json`). |
| `train/` | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (git submodule) with the training config and dataset registration in `train/llama_factory_yaml/`. |
| `baseline/*.ipynb` | Zero-shot baselines and inference against an OpenAI-compatible endpoint. |
| `baseline/metrics.py` | MAE, RMSE, MAPE, bias and per-ingredient MAE for every `llm_inference_results*.json`. |

## 📦 Installation

1. Clone the repository with its submodule:
```bash
git clone --recurse-submodules https://github.com/aihpi/food-waste
```

2. Create and activate a virtual environment:

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install .
```

For training, install LLaMA-Factory as described in its own README.

## 🛠️ Usage

### 1. Build and upload the dataset

```bash
python dataset/data_processor.py \
    --image-dir /path/to/images \
    --csv-dir /path/to/csv \
    --hf-repo-id username/dataset-name \
    --split train \
    --combine-mode ignore \
    --private
```

- `--image-dir`: Path to directory containing images. File names start with the receipt ID, e.g. `480654.00001.jpg`.
- `--csv-dir`: Path to directory containing CSV files. These are the kitchen's `Bon Zutaten`, `Bon Nährwerte` and `Mengen Rückläufer` exports, which are not part of this repository.
- `--hf-repo-id`: Hugging Face repository ID (e.g., "username/dataset-name")
- `--split`: Upload to a specific split [train, validation, test]
- `--private`: Make the repository private (default: True)
- `--combine-mode`: Use ignore for new datasets, overwrite if you want to overwrite the dataset and append if you want to add to a dataset (default: append)

After each upload the script sets the dataset licence to CC BY 4.0.

### 2. Create the training conversations

```bash
cd baseline
python sharegpt.py
```

This downloads the train split, saves the photos to `baseline/images/`, and writes `food_conversations.json`.

### 3. Fine-tune

Copy `train/llama_factory_yaml/data/` into LLaMA-Factory's `data/` folder and `train/llama_factory_yaml/examples/qwen2vl_food_waste.yaml` into its `examples/` folder, then run:

```bash
llamafactory-cli train examples/qwen2vl_food_waste.yaml
```

Image paths in `food_conversations.json` are relative (`images/meal_0.jpg`). Set `media_dir` in the YAML to the folder that contains `images/`.

### 4. Evaluate

Copy `baseline/.env.example` to `baseline/.env` and set `API_KEY` and `API_BASE_URL` for your OpenAI-compatible endpoint. Then run `baseline/test_model.ipynb` and compute the metrics:

```bash
python baseline/metrics.py
```

Evaluate on the dataset's **test** split. All training images come from the train split, so scores on train images measure fit, not generalisation.

## 📝 License

- **Code**: [Apache License 2.0](LICENSE)
- **Dataset** ([AI-ServicesBB/food-waste-dataset](https://huggingface.co/datasets/AI-ServicesBB/food-waste-dataset)), photographs and structured records: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Commercial use, ML training and publishing trained models or other derivatives are allowed with attribution.

## 💶 Funding

The AI Service Centre Berlin-Brandenburg is funded by the German Federal Ministry of Research, Technology and Space under the funding code "KI-Servicezentrum Berlin-Brandenburg" 16IS22092.

## 📖 Citation

If you use the dataset, please cite it as:

```bibtex
@misc{aisc_food_waste_dataset_2025,
  author       = {{AI Service Centre Berlin-Brandenburg, Hasso-Plattner-Institut für Digital Engineering gGmbH}},
  title        = {Food Waste Dataset: Photos of Plate Leftovers with Per-Ingredient Returned Weights},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/AI-ServicesBB/food-waste-dataset}},
  note         = {Licensed under CC BY 4.0}
}
```
