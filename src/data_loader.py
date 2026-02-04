"""
Data loader module for downloading and loading sentiment analysis dataset.
Handles dataset acquisition and saves to data/raw/ directory.
"""
import os
import pandas as pd
import requests
from pathlib import Path


class DataLoader:
    """Handles data loading and downloading for sentiment analysis."""
    
    def __init__(self, data_dir="data"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, data_path=None, data_type="train"):
        """
        Load data from local path or download from source.
        
        Args:
            data_path: Optional local path to CSV file
            data_type: Type of data ('train' or 'inference')
            
        Returns:
            pandas.DataFrame: Loaded dataset
        """
        if data_path and os.path.exists(data_path):
            print(f"Loading data from: {data_path}")
            df = pd.read_csv(data_path)
        else:
            # For this project, we'll use the IMDB dataset
            # Since the problem mentions specific paths, we'll handle both cases
            print(f"Attempting to load {data_type} data...")
            
            # Try to load from raw directory
            raw_file = self.raw_dir / f"{data_type}.csv"
            if raw_file.exists():
                print(f"Loading from {raw_file}")
                df = pd.read_csv(raw_file)
            else:
                # Download from alternative source or use demo data
                print(f"Data file not found. Please place {data_type}.csv in {self.raw_dir}")
                print("For this demo, creating sample dataset...")
                df = self._create_sample_data(data_type)
                
        # Save to raw directory
        output_path = self.raw_dir / f"{data_type}.csv"
        df.to_csv(output_path, index=False)
        print(f"Data saved to: {output_path}")
        print(f"Dataset shape: {df.shape}")
        
        return df
    
    def _create_sample_data(self, data_type="train"):
        """
        Create sample data for demonstration.
        In production, this would download from actual source.
        
        Args:
            data_type: Type of dataset to create
            
        Returns:
            pandas.DataFrame: Sample dataset
        """
        # This is a placeholder - in real scenario, data should be provided
        # or downloaded from authorized source
        if data_type == "train":
            sample_data = {
                'text': [
                    'This movie was fantastic! I really enjoyed it.',
                    'Terrible film, waste of time and money.',
                    'Amazing performance by the lead actor.',
                    'Boring and predictable plot.',
                    'One of the best movies I have seen this year!',
                ],
                'sentiment': ['pos', 'neg', 'pos', 'neg', 'pos']
            }
        else:
            sample_data = {
                'text': [
                    'Great cinematography and storyline.',
                    'Not worth watching, very disappointing.',
                ]
            }
        
        return pd.DataFrame(sample_data)
    
    def download_imdb_dataset(self):
        """
        Download IMDB dataset from Kaggle or alternative source.
        Note: Requires API credentials or manual download.
        """
        print("For IMDB dataset, please download manually from:")
        print("https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        print(f"And place files in: {self.raw_dir}")


def main():
    """Main function to demonstrate data loading."""
    loader = DataLoader()
    
    # Load training data
    print("=" * 50)
    print("Loading Training Data")
    print("=" * 50)
    train_df = loader.load_data(data_type="train")
    print(f"\nTraining data preview:")
    print(train_df.head())
    
    # Load inference data
    print("\n" + "=" * 50)
    print("Loading Inference Data")
    print("=" * 50)
    inference_df = loader.load_data(data_type="inference")
    print(f"\nInference data preview:")
    print(inference_df.head())


if __name__ == "__main__":
    main()
