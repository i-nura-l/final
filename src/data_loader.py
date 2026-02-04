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
            # Create a larger sample dataset for demonstration
            positive_reviews = [
                'This movie was fantastic! I really enjoyed it.',
                'Amazing performance by the lead actor.',
                'One of the best movies I have seen this year!',
                'Brilliant cinematography and excellent storytelling.',
                'Outstanding film with great acting and direction.',
                'Absolutely loved every minute of this masterpiece.',
                'The plot was engaging and the characters were well developed.',
                'A must-watch film with stunning visual effects.',
                'Incredible movie that exceeded all my expectations.',
                'Perfect blend of action, drama, and emotion.',
                'This film is a work of art. Simply beautiful.',
                'Superb acting and an amazing script.',
                'One of the most entertaining movies I have ever seen.',
                'The director did an excellent job with this film.',
                'A truly remarkable and unforgettable cinematic experience.',
                'Great story with wonderful performances all around.',
                'This movie kept me engaged from start to finish.',
                'Phenomenal movie with outstanding production value.',
                'Highly recommend this film to everyone.',
                'An absolute gem of a movie. Loved it!',
            ]
            
            negative_reviews = [
                'Terrible film, waste of time and money.',
                'Boring and predictable plot.',
                'Disappointing movie with poor acting.',
                'One of the worst films I have ever watched.',
                'The plot made no sense and the acting was terrible.',
                'Complete waste of time. Do not watch this.',
                'Poorly written script and bad direction.',
                'This movie was a total disaster.',
                'Terrible pacing and boring storyline.',
                'The acting was wooden and unconvincing.',
                'Failed to meet even the lowest expectations.',
                'Awful movie with no redeeming qualities.',
                'Painfully bad film that I could not finish.',
                'The worst movie of the year without a doubt.',
                'Terrible dialogue and poorly developed characters.',
                'A complete mess of a film.',
                'Boring, predictable, and poorly executed.',
                'Not worth watching at all. Very disappointing.',
                'This film was unbearable to sit through.',
                'Absolutely terrible in every possible way.',
            ]
            
            texts = positive_reviews + negative_reviews
            sentiments = ['pos'] * len(positive_reviews) + ['neg'] * len(negative_reviews)
            
            sample_data = {
                'text': texts,
                'sentiment': sentiments
            }
        else:
            sample_data = {
                'text': [
                    'Great cinematography and storyline.',
                    'Not worth watching, very disappointing.',
                    'Excellent acting and direction.',
                    'Boring movie with a weak plot.',
                    'One of my favorite films of all time.',
                    'Terrible movie, would not recommend.',
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
