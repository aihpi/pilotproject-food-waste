# Food Waste Project

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/aihpi/food-waste
cd food-waste
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

3. Install the package:
```bash
pip install .
```

## 🛠️ Usage

Process and upload your dataset:

```bash
python dataset_processor.py \
    --image-dir /path/to/images \
    --csv-dir /path/to/csv \
    --hf-repo-id username/dataset-name \
    --private
```

### Arguments

- `--image-dir`: Path to directory containing images
- `--csv-dir`: Path to directory containing CSV files
- `--hf-repo-id`: Hugging Face repository ID (e.g., "username/dataset-name")
- `--private`: Make the repository private (default: True)

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]