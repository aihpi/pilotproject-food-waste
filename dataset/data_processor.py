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
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
import shutil

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
        self.local_data_dir = Path("./local_data")
        self.local_data_dir.mkdir(exist_ok=True)

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
        
        # Initialize dictionary with specific column order: bonid, image, then others
        dataset_dict = {
            'bonid': [],
            'image': [],
        }
        
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
                    # Convert image to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                        
                    # Add base data
                    dataset_dict['bonid'].append(bon_id)
                    dataset_dict['image'].append(image)
                    
                    # We'll add other columns after these core columns
                    # Add sequence data from ingredients for each row individually
                    for col in sequence_columns:
                        if col not in dataset_dict:
                            dataset_dict[col] = []
                        dataset_dict[col].append(ingredients[col].tolist())

                    # Add single values from image_row
                    for col in single_columns:
                        if col not in dataset_dict:
                            dataset_dict[col] = []
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

        # Create dataset with PIL images and specified column order
        dataset = Dataset.from_dict(dataset_dict, features=features)
        logger.info(f"Created Hugging Face dataset with {len(dataset)} samples and columns in order: {list(dataset.features.keys())[:5]}...")
        
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
        """Upload the dataset to Huggingface, safely handling existing data."""
        api = HfApi()
        
        try:
            # Try to load existing dataset
            existing_dataset = load_dataset(repo_id)
            logger.info(f"Found existing dataset with {len(existing_dataset['train'])} samples")
            
            # Backup existing dataset locally before doing anything
            backup_dir = Path("./dataset_backup")
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / f"{repo_id.replace('/', '_')}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Creating backup of existing dataset at {backup_path}")
            existing_dataset['train'].save_to_disk(backup_path)
            logger.info(f"Backup created successfully at {backup_path}")
            
            # Get current version from existing dataset's metadata
            try:
                current_version = existing_dataset['train'].info.version
                # Parse semantic version (x.y.z format)
                if current_version and isinstance(current_version, str):
                    version_parts = current_version.split('.')
                    if len(version_parts) == 3 and all(part.isdigit() for part in version_parts):
                        major, minor, patch = map(int, version_parts)
                        new_version = f"{major}.{minor}.{int(patch) + 1}"
                    else:
                        new_version = "0.0.1"
                else:
                    new_version = "0.0.1"
                    
                logger.info(f"Current version: {current_version}, New version: {new_version}")
                
                # Store version history in dataset metadata instead of version field
                metadata = dataset.info.metadata or {}
                
                # Create or update version history in metadata
                if 'version_history' not in metadata:
                    metadata['version_history'] = []
                    
                metadata['version_history'].append({
                    "version": new_version,
                    "timestamp": datetime.now().isoformat(),
                    "samples_added": len(dataset),
                    "total_samples": len(existing_dataset['train']) + len(dataset)
                })
                
                # Try to combine datasets
                try:
                    # Combine existing and new datasets
                    combined_dataset = concatenate_datasets([existing_dataset['train'], dataset])
                    logger.info(f"Combined dataset has {len(combined_dataset)} samples")
                    
                    # Update dataset info
                    combined_dataset.info.version = new_version
                    combined_dataset.info.metadata = metadata
                    
                    # Push combined dataset to hub
                    combined_dataset.push_to_hub(repo_id, private=private)
                    
                    # Create version tag
                    try:
                        api.create_tag(
                            repo_id=repo_id,
                            tag=f"v{new_version}",
                            message=f"Version {new_version}: Added {len(dataset)} samples"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create version tag: {e}")
                    
                    logger.info(f"Successfully uploaded combined dataset to {repo_id}")
                    
                    return combined_dataset
                
                except ValueError as schema_error:
                    logger.warning(f"Schema mismatch between datasets: {schema_error}")
                    logger.warning("Cannot merge with existing dataset due to incompatible schemas.")
                    logger.warning("Adding new data as a separate parquet file...")
                    
                    # Create a dataset folder structure
                    tmp_dir = Path("./tmp_dataset")
                    tmp_dir.mkdir(exist_ok=True)
                    
                    # Export current combined dataset
                    existing_path = tmp_dir / "existing"
                    existing_path.mkdir(exist_ok=True)
                    existing_dataset['train'].save_to_disk(existing_path)
                    
                    # Export new dataset
                    new_path = tmp_dir / "new"
                    new_path.mkdir(exist_ok=True)
                    dataset.save_to_disk(new_path)
                    
                    # Push both datasets to hub with clear naming
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Generate parquet files
                    parquet_dir = tmp_dir / "parquet"
                    parquet_dir.mkdir(exist_ok=True)
                    
                    # Save existing dataset as parquet
                    existing_parquet = parquet_dir / f"existing_data.parquet"
                    existing_dataset['train'].to_parquet(existing_parquet)
                    
                    # Save new dataset as parquet with version in filename
                    new_parquet = parquet_dir / f"new_data_v{new_version}_{timestamp}.parquet"
                    dataset.to_parquet(new_parquet)
                    
                    # Upload both parquet files
                    api.upload_file(
                        path_or_fileobj=str(existing_parquet),
                        path_in_repo=f"parquet/existing_data.parquet",
                        repo_id=repo_id,
                        repo_type="dataset"
                    )
                    
                    api.upload_file(
                        path_or_fileobj=str(new_parquet),
                        path_in_repo=f"parquet/new_data_v{new_version}_{timestamp}.parquet",
                        repo_id=repo_id,
                        repo_type="dataset"
                    )
                    
                    # Upload a README with instructions
                    readme_content = f"""# Food Waste Dataset

## Dataset Information
- **Updated**: {datetime.now().strftime("%Y-%m-%d")}
- **Existing samples**: {len(existing_dataset['train'])}
- **New samples**: {len(dataset)}
- **Schema note**: The new data (v{new_version}) has a different schema than existing data.

## Accessing Data
This repository contains multiple parquet files due to schema differences:
- `parquet/existing_data.parquet`: Original dataset with {len(existing_dataset['train'])} samples
- `parquet/new_data_v{new_version}_{timestamp}.parquet`: New dataset with {len(dataset)} samples

## Loading Specific Datasets
```python
from datasets import load_dataset

# Load the original dataset
original_dataset = load_dataset("parquet", data_files="https://huggingface.co/{repo_id}/resolve/main/parquet/existing_data.parquet")

# Load the new dataset
new_dataset = load_dataset("parquet", data_files="https://huggingface.co/{repo_id}/resolve/main/parquet/new_data_v{new_version}_{timestamp}.parquet")
```

## Version History
{json.dumps(metadata['version_history'], indent=2)}
"""
                    
                    readme_path = tmp_dir / "README.md"
                    with open(readme_path, "w", encoding="utf-8") as f:
                        f.write(readme_content)
                    
                    api.upload_file(
                        path_or_fileobj=str(readme_path),
                        path_in_repo="README.md",
                        repo_id=repo_id,
                        repo_type="dataset"
                    )
                    
                    logger.info(f"Successfully uploaded both datasets as separate parquet files")
                    logger.info(f"Original data saved as parquet/existing_data.parquet")
                    logger.info(f"New data saved as parquet/new_data_v{new_version}_{timestamp}.parquet")
                    
                    return dataset
                    
            except Exception as e:
                logger.warning(f"Error handling version: {e}")
                new_version = "0.0.1"
                
                # Set version for new dataset
                dataset.info.version = new_version
                
                # Push to hub
                dataset.push_to_hub(repo_id, private=private)
                logger.info(f"Uploaded dataset with reset version {new_version}")
                return dataset
                
        except Exception as e:
            logger.warning(f"No existing dataset found or error loading it: {e}")
            logger.info("Creating new dataset...")
            
            # Create new repository if it doesn't exist
            try:
                api.create_repo(repo_id=repo_id, repo_type="dataset", private=private)
            except Exception as e:
                logger.warning(f"Repository might already exist: {e}")
            
            # Initialize version for new dataset (standard semantic version)
            new_version = "0.0.1"
            
            # Store version history in metadata
            metadata = dataset.info.metadata or {}
            metadata['version_history'] = [{
                "version": new_version,
                "timestamp": datetime.now().isoformat(),
                "samples_added": len(dataset),
                "total_samples": len(dataset)
            }]
            
            # Update dataset info
            dataset.info.version = new_version
            dataset.info.metadata = metadata
            
            # Push new dataset to hub
            dataset.push_to_hub(repo_id, private=private)
            
            # Create initial version tag
            try:
                api.create_tag(
                    repo_id=repo_id,
                    tag=f"v{new_version}",
                    message=f"Initial version with {len(dataset)} samples"
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
            logger.info(f"Dataset version: {new_version}")
            logger.info(f"Version history stored in dataset metadata")

        return dataset

    def download_dataset(self, repo_id):
        """
        Download a dataset from Hugging Face to a local directory and return the path to its parquet files.
        
        Args:
            repo_id: Hugging Face repository ID
            
        Returns:
            Dataset object and path to parquet directory or None if download fails
        """
        logger.info(f"Attempting to download dataset from {repo_id}")
        parquet_dir = self.local_data_dir / f"{repo_id.replace('/', '_')}_parquet"
        
        try:
            # Try to load existing dataset
            dataset = load_dataset(repo_id)
            logger.info(f"Successfully loaded dataset with {len(dataset['train'])} samples")
            
            # Create parquet directory
            parquet_dir.mkdir(exist_ok=True, parents=True)
            
            # Save dataset as parquet
            parquet_path = parquet_dir / "dataset.parquet"
            dataset['train'].to_parquet(parquet_path)
            logger.info(f"Saved dataset to {parquet_path}")
            
            return dataset['train'], parquet_dir
            
        except Exception as e:
            logger.warning(f"Failed to download dataset from {repo_id}: {e}")
            logger.info("No existing dataset found or error downloading it")
            return None, None

    def combine_with_new_data(self, existing_dataset, new_dataset):
        """
        Attempt to combine existing dataset with new dataset, handling schema differences.
        
        Args:
            existing_dataset: Existing Dataset object
            new_dataset: New Dataset object
            
        Returns:
            Combined Dataset object or the new dataset if combination fails
        """
        logger.info(f"Attempting to combine datasets: existing ({len(existing_dataset)} samples) + new ({len(new_dataset)} samples)")
        
        try:
            # Try simple concatenation first
            combined_dataset = concatenate_datasets([existing_dataset, new_dataset])
            logger.info(f"Successfully combined datasets with {len(combined_dataset)} total samples")
            return combined_dataset
        except ValueError as e:
            logger.warning(f"Schema mismatch when combining datasets: {e}")
            logger.warning("Attempting to align schemas and preserve image data...")
            
            try:
                # First, save images from both datasets
                existing_images = existing_dataset['image'] if 'image' in existing_dataset.features else []
                new_images = new_dataset['image'] if 'image' in new_dataset.features else []
                
                logger.info(f"Preserving {len(existing_images)} existing images and {len(new_images)} new images")
                
                # Convert to pandas for easier schema alignment
                existing_df = existing_dataset.to_pandas()
                new_df = new_dataset.to_pandas()
                
                # Remove image column temporarily from both dataframes
                if 'image' in existing_df.columns:
                    existing_df = existing_df.drop('image', axis=1)
                if 'image' in new_df.columns:
                    new_df = new_df.drop('image', axis=1)
                
                logger.info("Schema alignment: Converting all columns to string type")
                
                # Identify all columns
                all_columns = set(existing_df.columns).union(set(new_df.columns))
                
                # Add missing columns to each dataframe
                for col in all_columns:
                    if col not in existing_df.columns:
                        existing_df[col] = ""
                    if col not in new_df.columns:
                        new_df[col] = ""
                
                # Convert all columns to string to avoid type mismatches
                for col in all_columns:
                    existing_df[col] = existing_df[col].astype(str)
                    new_df[col] = new_df[col].astype(str)
                
                # Combine dataframes
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                logger.info(f"Combined dataframes with {len(combined_df)} rows")
                
                # Create a new dataset with all features except image
                from datasets import Dataset as HFDataset
                temp_dataset = HFDataset.from_pandas(combined_df)
                
                # Create dataset dictionary with specific column ordering
                bonid_column = 'bonid'
                if bonid_column not in temp_dataset.column_names and 'Bon_ID' in temp_dataset.column_names:
                    bonid_column = 'Bon_ID'
                
                # Start with bonid and image as the first two columns
                dataset_dict = {
                    bonid_column: temp_dataset[bonid_column],
                    'image': existing_images + new_images if (existing_images or new_images) else []
                }
                
                # Add all other columns except bonid (already added)
                for col in temp_dataset.column_names:
                    if col != bonid_column:
                        dataset_dict[col] = temp_dataset[col]
                
                # Set features with specific ordering
                features = {}
                if bonid_column == 'bonid':
                    features[bonid_column] = Value('int64')
                else:
                    features[bonid_column] = temp_dataset.features[bonid_column]
                
                features['image'] = HFImage()
                
                # Add other features
                for col in temp_dataset.column_names:
                    if col != bonid_column:
                        features[col] = temp_dataset.features[col]
                
                # Create final combined dataset with correct column order
                combined_dataset = Dataset.from_dict(dataset_dict, features=Features(features))
                logger.info(f"Created combined dataset with {len(combined_dataset)} samples")
                logger.info(f"Combined dataset features: {list(combined_dataset.features.keys())[:5]}...")
                
                # Verify image data
                if 'image' in combined_dataset.features and len(combined_dataset) > 0:
                    img_check = combined_dataset['image'][0]
                    logger.info(f"Image data type check: {type(img_check)}")
                
                return combined_dataset
            except Exception as e2:
                logger.error(f"Failed to align schemas: {e2}")
                logger.warning("Returning only the new dataset")
                return new_dataset


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
    parser.add_argument('--combine-mode', choices=['ignore', 'append', 'overwrite'], default='append',
                      help='How to handle existing data: ignore=use only new data, append=combine with existing, overwrite=replace existing')
    
    args = parser.parse_args()

    # Initialize processor and create dataset
    try:
        processor = DatasetProcessor(
            image_dir=Path(args.image_dir),
            csv_dir=Path(args.csv_dir)
        )
        
        # Check if we should download and load existing dataset
        existing_dataset = None
        parquet_dir = None
        
        if args.hf_repo_id and args.combine_mode != 'ignore':
            # Attempt to download and get parquet files
            existing_dataset, parquet_dir = processor.download_dataset(args.hf_repo_id)
            
            if existing_dataset is not None:
                logger.info(f"Downloaded dataset with {len(existing_dataset)} samples")
            elif args.combine_mode == 'overwrite':
                logger.error("Failed to download dataset for overwrite mode")
                exit(1)
        
        # Process new data if not in overwrite-only mode
        if args.combine_mode != 'overwrite' or existing_dataset is None:
            logger.info("Preprocessing new data...")
            processor.preprocess()
            
            logger.info("Creating dataset from new data...")
            new_dataset = processor.create_dataset()
            
            # Display new dataset information
            print("\n" + "="*50)
            print("NEW DATASET SUMMARY")
            print("="*50)
            print(f"Number of samples: {len(new_dataset)}")
            print(f"Features: {list(new_dataset.features.keys())}")
            if len(new_dataset) > 0:
                print(f"First few Bon IDs: {new_dataset['bonid'][:min(5, len(new_dataset))]}")
                
                # Display bon_id distribution
                bon_id_counts = {}
                for bon_id in new_dataset['bonid']:
                    bon_id_counts[bon_id] = bon_id_counts.get(bon_id, 0) + 1
                
                print("\nSample distribution:")
                for bon_id, count in list(bon_id_counts.items())[:10]:  # Show first 10 bon_ids
                    print(f"  Bon ID {bon_id}: {count} samples")
                
                if len(bon_id_counts) > 10:
                    print(f"  ... and {len(bon_id_counts) - 10} more Bon IDs")
            print("="*50)
        else:
            logger.info("Skipping new data processing as overwrite mode is selected")
            new_dataset = None
        
        # Combine datasets if needed
        final_dataset = None
        if existing_dataset is not None and new_dataset is not None and args.combine_mode == 'append':
            print("\n" + "="*50)
            print("COMBINING DATASETS")
            print("="*50)
            print(f"Existing dataset: {len(existing_dataset)} samples")
            print(f"New dataset: {len(new_dataset)} samples")
            print(f"Expected total: {len(existing_dataset) + len(new_dataset)} samples")
            print("="*50)
            
            final_dataset = processor.combine_with_new_data(existing_dataset, new_dataset)
            
            print("\n" + "="*50)
            print("COMBINED DATASET SUMMARY")
            print("="*50)
            print(f"Final dataset size: {len(final_dataset)} samples")
            print("="*50)
        elif existing_dataset is not None and args.combine_mode == 'overwrite':
            logger.info("Overwriting existing data with new data")
            final_dataset = new_dataset
        elif new_dataset is not None:
            logger.info("Using only new data")
            final_dataset = new_dataset
        else:
            logger.info("Using only existing data")
            final_dataset = existing_dataset
        
        if args.hf_repo_id and final_dataset is not None:
            # Check if dataset already exists on HF
            try:
                hf_dataset = load_dataset(args.hf_repo_id)
                print("\n" + "="*50)
                print("HUGGING FACE DATASET SUMMARY")
                print("="*50)
                print(f"Existing HF dataset: {len(hf_dataset['train'])} samples")
                print(f"Local dataset to upload: {len(final_dataset)} samples")
                print(f"Dataset Image to Upload: {type(final_dataset['image'][0])}")
                print("="*50)
            except Exception:
                print("\n" + "="*50)
                print("NEW REPOSITORY")
                print("="*50)
                print(f"This will create a new dataset repository '{args.hf_repo_id}'")
                print(f"With {len(final_dataset)} initial samples")
                print("="*50)
            
            # Ask for confirmation
            confirmation = input(f"\nDo you want to upload this dataset to {args.hf_repo_id}? (yes/no): ").strip().lower()
            
            if confirmation in ('yes', 'y'):
                logger.info("Uploading dataset to Huggingface...")
                processor.upload_to_huggingface(
                        dataset=final_dataset,
                    repo_id=args.hf_repo_id,
                    private=args.private
                )
                logger.info(f"Successfully uploaded dataset to {args.hf_repo_id}")
            else:
                logger.info("Upload cancelled by user")
        elif final_dataset is not None:
            # Save locally if not uploading
            output_dir = Path("./output_dataset")
            output_dir.mkdir(exist_ok=True)
            final_dataset.save_to_disk(output_dir)
            logger.info(f"Saved dataset locally to {output_dir}")
        else:
            logger.error("No dataset available to upload or save")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise