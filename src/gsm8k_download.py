import requests
import pandas as pd
import json
import os
from pathlib import Path

def download_gsm8k():
    """
    Download GSM8K dataset from Hugging Face API and convert to JSON
    """
    # API endpoint
    url = "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/train"
    
    print("Fetching parquet file URLs from Hugging Face API...")
    
    try:
        # Get the parquet file information
        response = requests.get(url)
        response.raise_for_status()
        
        parquet_files = response.json()
        
        if not parquet_files:
            print("No parquet files found.")
            return
        
        # Download the first parquet file
        parquet_url = parquet_files[0] if isinstance(parquet_files, list) else parquet_files
        
        print(f"Downloading parquet file from: {parquet_url}")
        
        # Download parquet file
        parquet_response = requests.get(parquet_url)
        parquet_response.raise_for_status()
        
        # Save temporary parquet file
        temp_parquet = "temp_gsm8k.parquet"
        with open(temp_parquet, 'wb') as f:
            f.write(parquet_response.content)
        
        print("Converting parquet to JSON...")
        
        # Read parquet file
        df = pd.read_parquet(temp_parquet)
        
        # Create data directory if it doesn't exist
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        output_path = data_dir / "gsm8k.json"
        df.to_json(output_path, orient='records', indent=2)
        
        # Clean up temporary file
        os.remove(temp_parquet)
        
        print(f"✓ Successfully saved to {output_path}")
        print(f"Total records: {len(df)}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    download_gsm8k()
