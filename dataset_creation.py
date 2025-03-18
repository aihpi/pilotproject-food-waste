import os
import pandas as pd
from PIL import Image
import glob
from pathlib import Path
import io
import base64
import re

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
            raise ValueError("data_dir not specified during initialization")

        # Create a dictionary to group images by Bon_ID
        image_groups = {}
        # Use regex pattern to match jpg/jpeg files (case insensitive)
        pattern = re.compile(r'\.(jpg|jpeg)$', re.IGNORECASE)
        for img in self.image_dir.iterdir():
            if pattern.search(img.name):
                bon_id = int(img.name.split('.')[0])
                if bon_id not in image_groups:
                    image_groups[bon_id] = []
                image_groups[bon_id].append(img.name)
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
                    new_row['image_base64'] = self.image_to_base64(img_path)
                    expanded_rows.append(new_row)

        # Create final DataFrame
        self.gericht_df = pd.DataFrame(expanded_rows)
        print(self.gericht_df)
        # Set image filename as index and rename to image_id
        self.gericht_df.index = self.gericht_df['image_filename'].str.replace('.jpg', '')
        self.gericht_df.index.name = 'image_id'
        self.gericht_df = self.gericht_df.drop('image_filename', axis=1)
        return self

    def preprocess(self):
        """Run the full preprocessing pipeline."""
        self.load_csv_data()
        self.process_data()
        self.add_images()
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
        from datasets import Dataset, Features, Value, Image as HFImage, Sequence
        import base64
        from PIL import Image
        import io

        if self.merged_data is None or self.gericht_df is None:
            raise ValueError("Datasets have not been created yet. Run preprocess() first.")

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
            
            for idx, image_row in bon_group.iterrows():
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
        return dataset

    def upload_to_huggingface(self, dataset, repo_id, private=True):
        """Upload the dataset to Huggingface."""
        from huggingface_hub import HfApi

        # Create repository if it doesn't exist
        api = HfApi()
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=private)
        except Exception as e:
            print(f"Repository might already exist: {e}")

        # Push dataset to hub
        dataset.push_to_hub(repo_id, private=private)

        return dataset


if __name__ == "__main__":
    import argparse
    import logging
    from pathlib import Path

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

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
        print(dataset)
        if args.hf_repo_id:
            logger.info("Uploading dataset to Huggingface...")
            processor.upload_to_huggingface(
                dataset=dataset,
                repo_id=args.hf_repo_id,
                private=args.private
            )
            logger.info(f"Successfully uploaded dataset to {args.hf_repo_id}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise