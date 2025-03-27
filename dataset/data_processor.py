import os
import pandas as pd
from PIL import Image
import glob
from pathlib import Path
import io
import base64
import re
import logging
from datasets import concatenate_datasets, Dataset, Features, Value, Image as HFImage, Sequence
from huggingface_hub import HfApi
from datasets import load_dataset
from datetime import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(self, csv_dir=None, image_dir=None):
        self.csv_dir = csv_dir
        self.image_dir = Path(image_dir) if image_dir else None
        self.csv_data = None
        self.merged_data = None
        self.gericht_df = None

    def load_csv_data(self):
        """Load CSV files from the specified directory."""
        csv_data = []
        csv_files = glob.glob(os.path.join(self.csv_dir, '*.csv'))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            csv_data.append(df)
            print(f"Loaded CSV file: {csv_file}")
        self.csv_data = csv_data
        return self

    def _get_dataframe_by_key(self, key_column):
        for df in self.csv_data:
            if key_column in df.columns:
                return df
        raise ValueError(f"No dataframe found with {key_column}")

    def process_data(self):
        """Process the CSV data to merge and update DataFrame."""
        # Assuming 'Bon_ID' is a key column to identify the correct dataframes
        zutaten_df = self._get_dataframe_by_key('Artikel').copy()
        df2 = self._get_dataframe_by_key('Artikelnr').copy()
        self.gericht_df = self._get_dataframe_by_key('Gericht').copy()
        df2['Menge_Rückläufer'] = df2['Menge_Rückläufer'].apply(lambda x: float(str(x).replace(',', '.')))
        df2['Prozent_Rückläufer'] = df2['Prozent_Rückläufer'].apply(lambda x: float(str(x).replace(',', '.')))

        zutaten_df['Menge_Rückläufer'] = 0
        zutaten_df['Prozent_Rückläufer'] = 0
        zutaten_df['Menge_Rückläufer'] = zutaten_df['Menge_Rückläufer'].astype(float)
        zutaten_df['Prozent_Rückläufer'] = zutaten_df['Prozent_Rückläufer'].astype(float)

        for _, row in df2.iterrows():
            bon_id = int(row['Bon_ID'])
            artikelnr = row['Artikelnr']
            menge_ruecklaeufer = row['Menge_Rückläufer']
            prozent_ruecklaeufer = row['Prozent_Rückläufer']
            
            if pd.notna(artikelnr):
                zutaten_df.loc[(zutaten_df['Bon_ID'] == bon_id) & (zutaten_df['Artikelnummer'] == artikelnr), 'Menge_Rückläufer'] += menge_ruecklaeufer
                zutaten_df.loc[(zutaten_df['Bon_ID'] == bon_id) & (zutaten_df['Artikelnummer'] == artikelnr), 'Prozent_Rückläufer'] += prozent_ruecklaeufer

        for _, row in df2.iterrows():
            bon_id = int(row['Bon_ID'])
            artikelnr = row['Artikelnr']
            menge_ruecklaeufer = row['Menge_Rückläufer']
            prozent_ruecklaeufer = row['Prozent_Rückläufer']
            
            if pd.isna(artikelnr):
                df1_subset = zutaten_df[(zutaten_df['Bon_ID'] == bon_id) & (zutaten_df['Menge_Rückläufer'] == 0) & (zutaten_df['Prozent_Rückläufer'] == 0)]
                num_existing_artikelnr = len(df1_subset)
                if num_existing_artikelnr > 0:
                    zutaten_df.loc[(zutaten_df['Bon_ID'] == bon_id) & (zutaten_df['Menge_Rückläufer'] == 0), 'Menge_Rückläufer'] += menge_ruecklaeufer / num_existing_artikelnr
                    zutaten_df.loc[(zutaten_df['Bon_ID'] == bon_id) & (zutaten_df['Prozent_Rückläufer'] == 0), 'Prozent_Rückläufer'] += prozent_ruecklaeufer / num_existing_artikelnr

        self.merged_data = zutaten_df
        return self
    
    def image_to_base64(self, image_path):
        try:
            with Image.open(image_path) as img:
                # Convert image to RGB if it's not
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Create a bytes buffer
                buffer = io.BytesIO()
                # Save the image as JPEG to the buffer
                img.save(buffer, format='JPEG')
                # Get the bytes from the buffer and encode to base64
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return img_str
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
        
    def add_images(self):
        """Add image data to the merged dataframe."""
        if not self.image_dir:
            raise ValueError("image_dir not specified during initialization")
        
        if self.gericht_df is None:
            raise ValueError("No gericht_df available. Call process_data() first.")

        # Create a dictionary to group images by Bon_ID
        image_groups = {}
        # Use regex pattern to match jpg/jpeg files (case insensitive)
        pattern = re.compile(r'\.(jpg|jpeg)$', re.IGNORECASE)
        
        # Check if images directory exists
        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")
            
        image_files = list(self.image_dir.iterdir())
        logger.info(f"Found {len(image_files)} files in image directory")
        
        for img in image_files:
            if pattern.search(img.name):
                try:
                    # Extract Bon_ID safely
                    bon_id_str = img.name.split('.')[0]
                    bon_id = int(bon_id_str)
                    if bon_id not in image_groups:
                        image_groups[bon_id] = []
                    image_groups[bon_id].append(img.name)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not extract Bon_ID from filename {img.name}: {e}")
                    continue
        
        logger.info(f"Found images for {len(image_groups)} Bon_IDs")
        
        # Create expanded DataFrame
        expanded_rows = []
        for _, row in self.gericht_df.iterrows():
            bon_id = row['Bon_ID']
            if bon_id in image_groups:
                for img_filename in image_groups[bon_id]:
                    # Create a copy of the row
                    new_row = row.copy()
                    # Add image filename
                    new_row['image_filename'] = img_filename
                    # Add base64 image
                    img_path = self.image_dir / img_filename
                    img_base64 = self.image_to_base64(img_path)
                    if img_base64 is not None:
                        new_row['image_base64'] = img_base64
                        expanded_rows.append(new_row)
                    else:
                        logger.warning(f"Skipping image {img_path} due to processing error")

        if not expanded_rows:
            logger.warning("No valid images were processed")
            return self
            
        # Create final DataFrame
        self.gericht_df = pd.DataFrame(expanded_rows)
        logger.info(f"Created DataFrame with {len(self.gericht_df)} rows")
        
        # Set image filename as index and rename to image_id
        self.gericht_df.index = self.gericht_df['image_filename'].str.replace('\.(jpg|jpeg)', '', regex=True, case=False)
        self.gericht_df.index.name = 'image_id'
        self.gericht_df = self.gericht_df.drop('image_filename', axis=1)
        return self

    def preprocess(self):
        """Run the full preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline")
        self.load_csv_data()
        self.process_data()
        if self.image_dir:
            self.add_images()
        else:
            logger.warning("No image directory specified, skipping image processing")
        return self.merged_data, self.gericht_df

    def save_dataframe(self, output_path):
        """Save the final dataframe to CSV."""
        if self.merged_data is not None:
            self.merged_data.to_csv(output_path, index=True)
        else:
            raise ValueError("DataFrame has not been created yet. Run preprocess() first.")

    def get_dataframe_info(self):
        """Print information about the final dataframe."""
        if self.merged_data is not None:
            print("Final DataFrame Info:")
            print(self.merged_data.info())
            print(f"\nNumber of rows in final DataFrame: {len(self.merged_data)}")
            print("\nSample of final DataFrame (without base64 string):")
            if 'image_base64' in self.merged_data.columns:
                print(self.merged_data.drop('image_base64', axis=1).head())
            else:
                print(self.merged_data.head())
        else:
            print("DataFrame has not been created yet. Run preprocess() first.")


    def create_dataset(self):
        """Create a Huggingface dataset from the processed data."""
        if self.merged_data is None or self.gericht_df is None:
            raise ValueError("Datasets have not been created yet. Run preprocess() first.")

        logger.info("Creating Hugging Face dataset")
        
        # Initialize dataset dictionary dynamically based on DataFrame columns
        sequence_columns = self.merged_data.columns.tolist()
        single_columns = [col for col in self.gericht_df.columns if col not in ['Bon_ID', 'image_base64']]
        
        dataset_dict = {
            'bonid': [],
            'image': []
        }
        # Add sequence columns (from merged_data)
        dataset_dict.update({col: [] for col in sequence_columns})
        # Add single value columns (from gericht_df)
        dataset_dict.update({col: [] for col in single_columns})

        # Group by Bon_ID and populate dictionary
        for bon_id, bon_group in self.gericht_df.groupby('Bon_ID'):
            ingredients = self.merged_data[self.merged_data['Bon_ID'] == bon_id]
            
            if ingredients.empty:
                logger.warning(f"No ingredients found for Bon_ID {bon_id}, skipping")
                continue
                
            for idx, image_row in bon_group.iterrows():
                try:
                    # Convert base64 to PIL Image
                    image_data = base64.b64decode(image_row['image_base64'])
                    image = Image.open(io.BytesIO(image_data))

                    # Add base data
                    dataset_dict['bonid'].append(bon_id)
                    dataset_dict['image'].append(image)

                    # Add sequence data from ingredients
                    for col in sequence_columns:
                        dataset_dict[col].append(ingredients[col].tolist())

                    # Add single values from image_row
                    for col in single_columns:
                        dataset_dict[col].append(image_row[col])
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {e}")
                    continue

        if not dataset_dict['bonid']:
            raise ValueError("No valid data to create dataset")
            
        logger.info(f"Created dataset dictionary with {len(dataset_dict['bonid'])} samples")
            
        # Create features description dynamically
        features = {
            'bonid': Value('int64'),
            'image': HFImage(),
        }
        
        # Add sequence features
        features.update({
            col: Sequence(Value('string' if ingredients[col].dtype == 'object' else 
                               'int64' if ingredients[col].dtype == 'int64' else 
                               'float32'))
            for col in sequence_columns
        })
        
        # Add single value features
        features.update({
            col: Value('string' if self.gericht_df[col].dtype == 'object' else 
                      'int64' if self.gericht_df[col].dtype == 'int64' else 
                      'float32')
            for col in single_columns
        })

        features = Features(features)

        # Create dataset
        dataset = Dataset.from_dict(dataset_dict, features=features)
        logger.info(f"Created Hugging Face dataset with {len(dataset)} samples")
        return dataset

    def _create_readme(self, dataset):
        """Create a README for the dataset."""
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get dataset features
        features_list = "\n".join([f"- **{k}**: {v}" for k, v in dataset.features.items()])
        
        # Create README content
        readme = f"""# Food Waste Dataset

## Dataset Information
- **Created/Updated**: {current_date}
- **Number of samples**: {len(dataset)}
- **License**: [Add license information]

## Description
This dataset contains food waste data with images and nutritional information.

## Features
{features_list}

## Usage
```python
from datasets import load_dataset

dataset = load_dataset("{dataset.name}")
```

## Citation
[Add citation information]
"""
        return readme

    def upload_to_huggingface(self, dataset, repo_id, private=True):
        """Upload the dataset to Huggingface, appending to existing dataset if it exists."""
        api = HfApi()
        
        try:
            # Try to load existing dataset
            existing_dataset = load_dataset(repo_id)
            logger.info(f"Found existing dataset with {len(existing_dataset['train'])} samples")
            
            # Get current version from existing dataset's metadata
            current_version = existing_dataset['train'].info.version
            try:
                version_info = json.loads(current_version) if current_version else {"version": 0, "history": []}
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not parse existing version info: {current_version}")
                version_info = {"version": 0, "history": []}
            
            # Increment version number
            version_info["version"] += 1
            version_info["history"].append({
                "version": version_info["version"],
                "timestamp": datetime.now().isoformat(),
                "samples_added": len(dataset),
                "total_samples": len(existing_dataset['train']) + len(dataset)
            })
            
            # Combine existing and new datasets
            combined_dataset = concatenate_datasets([existing_dataset['train'], dataset])
            logger.info(f"Combined dataset has {len(combined_dataset)} samples")
            
            # Update dataset info with new version
            combined_dataset.info.version = json.dumps(version_info)
            
            # Push combined dataset to hub
            combined_dataset.push_to_hub(repo_id, private=private)
            
            # Create version tag
            try:
                api.create_tag(
                    repo_id=repo_id,
                    tag=f"v{version_info['version']}",
                    message=f"Version {version_info['version']}: Added {len(dataset)} samples"
                )
            except Exception as e:
                logger.warning(f"Failed to create version tag: {e}")
            
            logger.info(f"Successfully uploaded combined dataset to {repo_id}")
            logger.info(f"Dataset version: {version_info['version']}")
            logger.info(f"Version history: {json.dumps(version_info['history'], indent=2)}")
            
            return combined_dataset
            
        except Exception as e:
            logger.warning(f"No existing dataset found or error loading it: {e}")
            logger.info("Creating new dataset...")
            
            # Create new repository if it doesn't exist
            try:
                api.create_repo(repo_id=repo_id, repo_type="dataset", private=private)
            except Exception as e:
                logger.warning(f"Repository might already exist: {e}")
            
            # Initialize version info for new dataset
            version_info = {
                "version": 1,
                "history": [{
                    "version": 1,
                    "timestamp": datetime.now().isoformat(),
                    "samples_added": len(dataset),
                    "total_samples": len(dataset)
                }]
            }
            
            # Update dataset info with version
            dataset.info.version = json.dumps(version_info)
            
            # Push new dataset to hub
            dataset.push_to_hub(repo_id, private=private)
            
            # Create initial version tag
            try:
                api.create_tag(
                    repo_id=repo_id,
                    tag="v1",
                    message="Initial version"
                )
            except Exception as e:
                logger.warning(f"Failed to create version tag: {e}")
            
            # Create and upload README
            readme_content = self._create_readme(dataset)
            with open("README.md", "w", encoding="utf-8") as f:
                f.write(readme_content)
            
            try:
                api.upload_file(
                    path_or_fileobj="README.md",
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset"
                )
            except Exception as e:
                logger.warning(f"Failed to upload README: {e}")
            
            logger.info(f"Successfully uploaded new dataset to {repo_id}")
            logger.info(f"Dataset version: 1")
            logger.info(f"Version history: {json.dumps(version_info['history'], indent=2)}")
            
            return dataset


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process and upload food waste dataset')
  
    parser.add_argument('--image-dir', type=str, required=True,
                      help='Path to the image directory')
    parser.add_argument('--csv-dir', type=str, required=True,
                      help='Path to the CSV directory')
    parser.add_argument('--hf-repo-id', type=str,
                      help='Huggingface repository ID (e.g., "username/dataset-name")')
    parser.add_argument('--private', action='store_true', default=True,
                      help='Make the repository private (default: True)')
    
    args = parser.parse_args()

    # Initialize processor and create dataset
    try:
        processor = DatasetProcessor(
            image_dir=Path(args.image_dir),
            csv_dir=Path(args.csv_dir)
        )
        
        logger.info("Preprocessing dataset...")
        processor.preprocess()
        
        logger.info("Creating dataset...")
        dataset = processor.create_dataset()
        
        # Display dataset information
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        print(f"Number of samples: {len(dataset)}")
        print(f"Features: {list(dataset.features.keys())}")
        print(f"First few Bon IDs: {dataset['bonid'][:5]}")
        print("\nSample distribution:")
        
        # Display bon_id distribution
        bon_id_counts = {}
        for bon_id in dataset['bonid']:
            bon_id_counts[bon_id] = bon_id_counts.get(bon_id, 0) + 1
        
        for bon_id, count in list(bon_id_counts.items())[:10]:  # Show first 10 bon_ids
            print(f"  Bon ID {bon_id}: {count} samples")
        
        if len(bon_id_counts) > 10:
            print(f"  ... and {len(bon_id_counts) - 10} more Bon IDs")
            
        print("="*50)
        
        if args.hf_repo_id:
            # Ask for confirmation
            confirmation = input(f"\nDo you want to upload this dataset to {args.hf_repo_id}? (yes/no): ").strip().lower()
            
            if confirmation in ('yes', 'y'):
                logger.info("Uploading dataset to Huggingface...")
                processor.upload_to_huggingface(
                    dataset=dataset,
                    repo_id=args.hf_repo_id,
                    private=args.private
                )
                logger.info(f"Successfully uploaded dataset to {args.hf_repo_id}")
            else:
                logger.info("Upload cancelled by user")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise