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
        self.gericht_df.index = self.gericht_df['image_filename'].str.replace('\\.(jpg|jpeg)', '', regex=True, case=False)
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
                # Ensure metadata exists
                if dataset.info.metadata is None:
                    dataset.info.metadata = {}
                metadata = dataset.info.metadata # Now safe to access
                
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
            # Ensure metadata exists
            if dataset.info.metadata is None:
                dataset.info.metadata = {}
            metadata = dataset.info.metadata # Now safe to access
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
        Combine existing dataset with new dataset by proactively aligning schemas.
        """
        print("="*80)
        print("START OF DATASET COMBINATION")
        print("="*80)
        
        print(f"Existing dataset: {len(existing_dataset)} samples")
        print(f"New dataset: {len(new_dataset)} samples")
        
        # Print sample data from both datasets
        if len(existing_dataset) > 0:
            print("\nSAMPLE FROM EXISTING DATASET:")
            sample = existing_dataset[0]
            for key, value in list(sample.items())[:10]:
                if key != 'image':
                    print(f"  {key}: {type(value).__name__} - {value}")
        
        if len(new_dataset) > 0:
            print("\nSAMPLE FROM NEW DATASET:")
            sample = new_dataset[0]
            for key, value in list(sample.items())[:10]:
                if key != 'image':
                    print(f"  {key}: {type(value).__name__} - {value}")
        
        # Compare features and identify mismatches BEFORE attempting concatenation
        print("\nCOMPARING FEATURES:")
        mismatches = []
        
        print("Existing dataset features:")
        for name, feature in existing_dataset.features.items():
            if name != 'image':
                print(f"  {name}: {str(feature)}")
        
        print("New dataset features:")
        for name, feature in new_dataset.features.items():
            if name != 'image':
                print(f"  {name}: {str(feature)}")
        
        for feature_name in set(existing_dataset.features.keys()).intersection(set(new_dataset.features.keys())):
            if feature_name == 'image':
                continue  # Skip image feature
            
            existing_feature = existing_dataset.features[feature_name]
            new_feature = new_dataset.features[feature_name]
            
            # Check if types are different
            if str(existing_feature) != str(new_feature):
                mismatches.append((feature_name, existing_feature, new_feature))
                print(f"MISMATCH: {feature_name}: existing={existing_feature}, new={new_feature}")
        
        print(f"\nFound {len(mismatches)} mismatched features")
        
        # If no mismatches, we can directly concatenate
        if not mismatches:
            print("No schema mismatches found, proceeding with direct concatenation")
            try:
                combined_dataset = concatenate_datasets([existing_dataset, new_dataset])
                print(f"Successfully combined datasets with {len(combined_dataset)} total samples")
                return combined_dataset
            except Exception as e:
                print(f"ERROR during concatenation despite no mismatches: {e}")
                # Continue with alignment logic
        
        # Process each mismatch
        modified_dataset = new_dataset
        
        # Print sample data for mismatched features
        if len(new_dataset) > 0:
            print("\nSAMPLE DATA FOR MISMATCHED FEATURES:")
            sample = new_dataset[0]
            for feature_name, _, _ in mismatches:
                if feature_name in sample:
                    value = sample[feature_name]
                    print(f"  {feature_name}: {type(value).__name__} - {value}")
                    if isinstance(value, list) and len(value) > 0:
                        print(f"    First element: {type(value[0]).__name__} - {value[0]}")
        
        # Process each mismatch
        print("\nAPPLYING TYPE CONVERSIONS:")
        for feature_name, existing_feature, new_feature in mismatches:
            # Handle sequence features
            if hasattr(existing_feature, 'feature') and hasattr(new_feature, 'feature'):
                # Both are sequence types
                existing_type = existing_feature.feature.dtype
                new_type = new_feature.feature.dtype
                
                print(f"Converting sequence feature '{feature_name}' from {new_type} to {existing_type}")
                
                # Get sample data to better understand what we're working with
                if len(new_dataset) > 0:
                    sample_value = new_dataset[0][feature_name] if feature_name in new_dataset[0] else None
                    print(f"  Sample data: {type(sample_value).__name__} - {sample_value[:3] if isinstance(sample_value, list) and len(sample_value) >= 3 else sample_value}")
                
                # Check for both flat lists and nested lists
                sample_is_nested = False
                if isinstance(sample_value, list) and len(sample_value) > 0 and isinstance(sample_value[0], list):
                    sample_is_nested = True
                    print(f"  Detected nested list structure for {feature_name}")
                
                # For numeric columns that need comma formatting in strings
                needs_comma_format = feature_name in ['Gewicht_Kelle', 'Gewicht_Teller', 'kcal_Teller', 'kj_Teller', 'Fett_Teller', 
                                                    'ges_Fettsäuren_Teller', 'Kohlenhydrate_Teller', 'Zucker_Teller', 'Eiweiß_Teller', 'Salz_Teller']
                
                if existing_type == 'string' and new_type in ('int64', 'float32'):
                    # Convert numbers to strings, using comma as decimal separator if needed
                    print(f"  Converting numerics to strings for {feature_name}")
                    try:
                        if needs_comma_format:
                            # Convert numbers to strings with comma as decimal separator
                            def format_with_comma(x):
                                if isinstance(x, (int, float)):
                                    return str(x).replace('.', ',')
                                elif isinstance(x, str) and x.replace('.', '', 1).isdigit():
                                    return str(float(x)).replace('.', ',')
                                return str(x)
                            
                            modified_dataset = modified_dataset.map(
                                lambda x: {feature_name: [format_with_comma(v) for v in x[feature_name]]},
                                desc=f"Converting {feature_name} to strings with comma decimal"
                            )
                            print(f"  Converted to strings with comma decimal format")
                        else:
                            # Standard string conversion
                            modified_dataset = modified_dataset.map(
                                lambda x: {feature_name: [str(v) for v in x[feature_name]]},
                                desc=f"Converting {feature_name} to strings"
                            )
                            print(f"  Converted to plain strings")
                    except Exception as e:
                        print(f"  ERROR during conversion to string: {e}")
                        print(f"  Sample causing error: {new_dataset[0][feature_name] if len(new_dataset) > 0 and feature_name in new_dataset[0] else 'unknown'}")
                
                elif existing_type == 'int64' and new_type == 'float32':
                    # Convert float sequences to int sequences
                    print(f"  Converting floats to ints for {feature_name}")
                    try:
                        modified_dataset = modified_dataset.map(
                            lambda x: {feature_name: [int(float(v)) for v in x[feature_name]]},
                            desc=f"Converting {feature_name} floats to ints"
                        )
                        print(f"  Float to int conversion successful")
                    except Exception as e:
                        print(f"  ERROR during float to int conversion: {e}")
                        print(f"  Sample causing error: {new_dataset[0][feature_name] if len(new_dataset) > 0 and feature_name in new_dataset[0] else 'unknown'}")
                
                elif existing_type == 'float32' and new_type == 'int64':
                    # Convert int sequences to float sequences
                    print(f"  Converting ints to floats for {feature_name}")
                    try:
                        modified_dataset = modified_dataset.map(
                            lambda x: {feature_name: [float(v) for v in x[feature_name]]},
                            desc=f"Converting {feature_name} ints to floats"
                        )
                        print(f"  Int to float conversion successful")
                    except Exception as e:
                        print(f"  ERROR during int to float conversion: {e}")
                        print(f"  Sample causing error: {new_dataset[0][feature_name] if len(new_dataset) > 0 and feature_name in new_dataset[0] else 'unknown'}")
                    
                elif new_type == 'string' and existing_type in ('int64', 'float32'):
                    # String to numeric conversion with handling comma decimal separator
                    print(f"  Converting strings to {existing_type} for {feature_name}")
                    try:
                        if existing_type == 'int64':
                            # Function to convert string (potentially with comma) to int
                            def str_to_int(s):
                                try:
                                    # Replace comma with period, convert to float first then int
                                    if isinstance(s, str):
                                        s = s.replace(',', '.')
                                    return int(float(s))
                                except (ValueError, TypeError):
                                    print(f"    Warning: Could not convert '{s}' to int, using 0")
                                    return 0
                            
                            modified_dataset = modified_dataset.map(
                                lambda x: {feature_name: [str_to_int(v) for v in x[feature_name]]},
                                desc=f"Converting {feature_name} strings to ints"
                            )
                            print(f"  String to int conversion successful")
                        else:  # float32
                            # Function to convert string (potentially with comma) to float
                            def str_to_float(s):
                                try:
                                    # Replace comma with period for decimal
                                    if isinstance(s, str):
                                        s = s.replace(',', '.')
                                    return float(s)
                                except (ValueError, TypeError):
                                    print(f"    Warning: Could not convert '{s}' to float, using 0.0")
                                    return 0.0
                            
                            modified_dataset = modified_dataset.map(
                                lambda x: {feature_name: [str_to_float(v) for v in x[feature_name]]},
                                desc=f"Converting {feature_name} strings to floats"
                            )
                            print(f"  String to float conversion successful")
                    except Exception as e:
                        print(f"  ERROR during string to numeric conversion: {e}")
                        print(f"  Sample causing error: {new_dataset[0][feature_name] if len(new_dataset) > 0 and feature_name in new_dataset[0] else 'unknown'}")
            
            # Handle non-sequence features (scalar values)
            else:
                existing_type = existing_feature.dtype
                new_type = new_feature.dtype
                
                print(f"Converting non-sequence feature '{feature_name}' from {new_type} to {existing_type}")
                
                # For numeric columns that need comma formatting in strings
                needs_comma_format = feature_name in ['Gewicht_Kelle', 'Gewicht_Teller', 'kcal_Teller', 'kj_Teller', 'Fett_Teller', 
                                                    'ges_Fettsäuren_Teller', 'Kohlenhydrate_Teller', 'Zucker_Teller', 'Eiweiß_Teller', 'Salz_Teller']
                
                if existing_type == 'string' and new_type in ('int64', 'float32'):
                    # Convert to string, using comma as decimal separator if needed
                    try:
                        if needs_comma_format:
                            # Numbers to strings with comma as decimal separator
                            modified_dataset = modified_dataset.map(
                                lambda x: {
                                    feature_name: str(float(x[feature_name])).replace('.', ',') 
                                    if isinstance(x[feature_name], (int, float)) or (
                                        isinstance(x[feature_name], str) and 
                                        x[feature_name].replace('.', '', 1).isdigit()
                                    ) else str(x[feature_name])
                                },
                                desc=f"Converting {feature_name} to string with comma decimal"
                            )
                            print(f"  Converted to string with comma decimal format")
                        else:
                            # Standard string conversion
                            modified_dataset = modified_dataset.map(
                                lambda x: {feature_name: str(x[feature_name])},
                                desc=f"Converting {feature_name} to string"
                            )
                            print(f"  Converted to plain string")
                    except Exception as e:
                        print(f"  ERROR during conversion to string: {e}")
                        print(f"  Sample causing error: {new_dataset[0][feature_name] if len(new_dataset) > 0 and feature_name in new_dataset[0] else 'unknown'}")
                
                elif existing_type == 'int64' and new_type == 'float32':
                    # Convert float to int
                    try:
                        modified_dataset = modified_dataset.map(
                            lambda x: {feature_name: int(float(x[feature_name]))},
                            desc=f"Converting {feature_name} float to int"
                        )
                        print(f"  Float to int conversion successful")
                    except Exception as e:
                        print(f"  ERROR during float to int conversion: {e}")
                        print(f"  Sample causing error: {new_dataset[0][feature_name] if len(new_dataset) > 0 and feature_name in new_dataset[0] else 'unknown'}")
                
                elif existing_type == 'float32' and new_type == 'int64':
                    # Convert int to float
                    try:
                        modified_dataset = modified_dataset.map(
                            lambda x: {feature_name: float(x[feature_name])},
                            desc=f"Converting {feature_name} int to float"
                        )
                        print(f"  Int to float conversion successful")
                    except Exception as e:
                        print(f"  ERROR during int to float conversion: {e}")
                        print(f"  Sample causing error: {new_dataset[0][feature_name] if len(new_dataset) > 0 and feature_name in new_dataset[0] else 'unknown'}")
                    
                elif new_type == 'string' and existing_type in ('int64', 'float32'):
                    # String to numeric conversion
                    try:
                        if existing_type == 'int64':
                            # Function to convert string (potentially with comma) to int
                            def scalar_str_to_int(s):
                                try:
                                    # Replace comma with period, convert to float first then int
                                    if isinstance(s, str):
                                        s = s.replace(',', '.')
                                    return int(float(s))
                                except (ValueError, TypeError):
                                    print(f"    Warning: Could not convert '{s}' to int, using 0")
                                    return 0
                        
                            modified_dataset = modified_dataset.map(
                                lambda x: {feature_name: scalar_str_to_int(x[feature_name])},
                                desc=f"Converting {feature_name} string to int"
                            )
                            print(f"  String to int conversion successful")
                        else:  # float32
                            # Function to convert string (potentially with comma) to float
                            def scalar_str_to_float(s):
                                try:
                                    # Replace comma with period for decimal
                                    if isinstance(s, str):
                                        s = s.replace(',', '.')
                                    return float(s)
                                except (ValueError, TypeError):
                                    print(f"    Warning: Could not convert '{s}' to float, using 0.0")
                                    return 0.0
                        
                            modified_dataset = modified_dataset.map(
                                lambda x: {feature_name: scalar_str_to_float(x[feature_name])},
                                desc=f"Converting {feature_name} string to float"
                            )
                            print(f"  String to float conversion successful")
                    except Exception as e:
                        print(f"  ERROR during string to numeric conversion: {e}")
                        print(f"  Sample causing error: {new_dataset[0][feature_name] if len(new_dataset) > 0 and feature_name in new_dataset[0] else 'unknown'}")
        
        # Verify that feature schemas match after conversion
        print("\nVERIFYING FEATURE SCHEMAS AFTER CONVERSION:")
        remaining_mismatches = []
        for feature_name in set(existing_dataset.features.keys()).intersection(set(modified_dataset.features.keys())):
            if feature_name == 'image':
                continue
            
            existing_feature = existing_dataset.features[feature_name]
            modified_feature = modified_dataset.features[feature_name]
            
            if str(existing_feature) != str(modified_feature):
                remaining_mismatches.append((feature_name, existing_feature, modified_feature))
                print(f"STILL MISMATCHED: {feature_name}: existing={existing_feature}, modified={modified_feature}")
        
        if remaining_mismatches:
            print(f"\nWARNING: {len(remaining_mismatches)} features still have mismatched schemas")
        else:
            print("\nAll feature schemas now match!")
        
        # Now concatenate with aligned schemas
        print("\nATTEMPTING CONCATENATION WITH ALIGNED SCHEMAS")
        try:
            combined_dataset = concatenate_datasets([existing_dataset, modified_dataset])
            print(f"Successfully combined datasets after schema alignment: {len(combined_dataset)} total samples")
            
            # Print a sample from the combined dataset
            if len(combined_dataset) > 0:
                print("\nCOMBINED DATASET SAMPLE:")
                sample = combined_dataset[0]
                for key, value in list(sample.items())[:10]:
                    if key != 'image':
                        print(f"  {key}: {type(value).__name__} - {value}")
            
            print("="*80)
            print("END OF DATASET COMBINATION - SUCCESS")
            print("="*80)
            return combined_dataset
        except ValueError as e:
            print(f"FAILED to combine datasets after type conversion: {e}")
            print("\nDETAILED FEATURE COMPARISON AFTER ALIGNMENT:")
            for feature_name in set(existing_dataset.features.keys()).intersection(set(modified_dataset.features.keys())):
                if feature_name != 'image':
                    print(f"  {feature_name}: existing={existing_dataset.features[feature_name]}, modified={modified_dataset.features[feature_name]}")
            
            print("="*80)
            print("END OF DATASET COMBINATION - FAILURE")
            print("="*80)
            raise


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
        else:
            logger.info("Skipping new data processing as overwrite mode is selected")
            new_dataset = None
        
        # Combine datasets if needed
        final_dataset = None
        if existing_dataset is not None and new_dataset is not None and args.combine_mode == 'append':
            final_dataset = processor.combine_with_new_data(existing_dataset, new_dataset)
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
            # Ask for confirmation
            confirmation = input(f"\nUpload {len(final_dataset)} samples to {args.hf_repo_id}? (yes/no): ").strip().lower()

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