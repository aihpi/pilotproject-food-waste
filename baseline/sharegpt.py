import json
from typing import List, Dict
from datasets import load_dataset
import pandas as pd
import ast
import os
from PIL import Image

def download_dataset():
    """
    Downloads the food waste dataset from Hugging Face and saves images locally.
    
    Returns:
        DataFrame containing the food waste data
    """
    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)
    
    # Load dataset from Hugging Face
    dataset = load_dataset("AI-ServicesBB/food-waste-dataset")
    
    # Get the actual image paths from the dataset
    def get_image_path(example, index):
        # Save the image locally and return its path
        image = example['image']
        if image is not None:
            # Create a filename based on the index
            filename = f"{os.getcwd()}/images/meal_{index}.jpg"
            # Save the image
            image.save(filename)
            return filename
        return None
    
    # Convert to pandas DataFrame and add image paths
    df = pd.DataFrame(dataset['train'])
    
    # Save images and get their paths
    df['image_path'] = [get_image_path(row, idx) for idx, row in enumerate(dataset['train'])]
    
    return df

def process_dataset_row(row) -> Dict:
    """
    Processes a single row from the dataset into the required format.
    
    Args:
        row: Row from the DataFrame
        
    Returns:
        Dictionary containing processed meal data
    """
    # Convert string representations of lists to actual lists if needed
    try:
        artikel_list = ast.literal_eval(row['Artikel']) if isinstance(row['Artikel'], str) else row['Artikel']
        ruecklaufer_list = ast.literal_eval(row['Menge_Rückläufer']) if isinstance(row['Menge_Rückläufer'], str) else row['Menge_Rückläufer']
        gewicht_teller_list = ast.literal_eval(row['Gewicht_Teller']) if isinstance(row['Gewicht_Teller'], str) else row['Gewicht_Teller']
        
    except:
        print(f"Error processing row {row['bonid']}")
        print(f"Artikel: {row['Artikel']}")
        print(f"Menge_Rückläufer: {row['Menge_Rückläufer']}")
        print(f"Gewicht_Teller: {row['Gewicht_Teller']}")
        return None

    # Create ingredients list with their weights and return quantities
    ingredients = []
    for artikel, gewicht, ruecklauf in zip(artikel_list, gewicht_teller_list, ruecklaufer_list):
        if pd.notna(artikel) and pd.notna(gewicht):
            ingredients.append({
                "Artikel": artikel,
                "Gewicht_Teller": gewicht,
                "Menge_Rückläufer": ruecklauf
            })
    
    # Get the actual image path from the dataset
    image_path = row.get('image_path', '')
    
    return {
        "bonid": str(row['bonid']),
        "image": image_path,
        "ingredients": ingredients,
        "Gewicht_vorher": row['Gewicht_vorher']
    }

def create_food_qa(meal_data: Dict) -> Dict:
    """
    Creates a single question-answer pair about a meal's contents and weights.
    
    Args:
        meal_data: Dictionary containing meal information
        
    Returns:
        Dictionary with messages and images in the specified format
    """
    if meal_data is None:
        return None

    # Create initial ingredients list with weights
    initial_ingredients = []
    for item in meal_data.get("ingredients", []):
        name = item.get("Artikel", "")
        weight = item.get("Gewicht_Teller", 0)
        if name and weight is not None:
            initial_ingredients.append({
                "name": name,
                "weight": weight
            })
    
    # Create the answer ingredients with returns
    returned_ingredients = []
    for item in meal_data.get("ingredients", []):
        name = item.get("Artikel", "")
        weight = item.get("Gewicht_Teller", 0)
        ruecklauf = item.get("Menge_Rückläufer", 0)
        if name and weight is not None:
            returned_ingredients.append({
                "name": name,
                "initial_weight": weight,
                "returned_weight": ruecklauf
            })
    
    # Create initial data JSON
    initial_data = {
        "ingredients": initial_ingredients,
        "total_weight": meal_data.get("Gewicht_vorher", 0)
    }
    
    # Create return data JSON
    return_data = {
        "ingredients": returned_ingredients,
        "total_initial_weight": meal_data.get("Gewicht_vorher", 0)
    }
    
    # Get image path - using the correct key "image" instead of "image_path"
    image_path = meal_data.get("image", "")
    
    qa_entry = {
        "messages": [
            {
                "content": f"<image>Given this initial meal data in JSON format:\n\n```json\n{json.dumps(initial_data, indent=2)}\n```\n\nPlease provide the amount of food returned for each ingredient in JSON format.",
                "role": "user"
            },
            {
                "content": f"Here are the ingredients with their initial weights and returned amounts:\n\n```json\n{json.dumps(return_data, indent=2)}\n```",
                "role": "assistant"
            }
        ],
        "images": [image_path] if image_path else []
    }
    
    return qa_entry

def create_sharegpt_dataset(meals_data: List[Dict]) -> List[Dict]:
    """
    Creates a ShareGPT dataset from a list of meal data.
    
    Args:
        meals_data: List of dictionaries containing meal information
        
    Returns:
        List of Q&A pairs with images in the specified format
    """
    dataset = [create_food_qa(meal) for meal in meals_data]
    return [entry for entry in dataset if entry is not None]

def save_sharegpt_dataset(dataset: List[Dict], output_file: str):
    """
    Saves the ShareGPT dataset to a JSON file.
    
    Args:
        dataset: List of Q&A pairs with images
        output_file: Path to output JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

# Example usage:
if __name__ == "__main__":
    # Download and process the dataset
    df = download_dataset()
    print("Dataset columns:", df.columns)
    print("\nSample row Artikel:", df['Artikel'].iloc[0])
    print("Sample row Menge_Rückläufer:", df['Menge_Rückläufer'].iloc[0])
    print("Sample row image_path:", df['image_path'].iloc[0])
    
    # Process each row in the dataset
    meals_data = [process_dataset_row(row) for _, row in df.iterrows()]
    
    # Create dataset
    dataset = create_sharegpt_dataset(meals_data)
    
    # Print some statistics
    print(f"\nProcessed {len(dataset)} valid entries")
    
    # Save to file
    save_sharegpt_dataset(dataset, "food_conversations.json")
