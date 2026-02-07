"""
CSV file creation utilities for EmoDb project.
"""
import pandas as pd
from pathlib import Path


def create_csv(df, file_path, file_name):
    """
    Create a CSV file from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        file_path (str): Directory path where the CSV will be saved
        file_name (str): Name of the CSV file (with or without .csv extension)
        
    Returns:
        str: Full path to the created CSV file
        
    Example:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> create_csv(df, 'data/csv', 'my_data.csv')
        'data/csv/my_data.csv'
    """
    # Ensure file_path is a Path object
    path = Path(file_path)
    
    # Create directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)
    
    # Add .csv extension if not present
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    
    # Create full file path
    full_path = path / file_name
    
    # Save DataFrame to CSV
    df.to_csv(full_path, index=False)
    
    print(f"CSV file created at: {full_path}")
    
    return str(full_path)
