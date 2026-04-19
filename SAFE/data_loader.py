"""
Data loader for SAFE (Sound Analysis for Fall Events) dataset.
Loads audio files and prepares them for feature extraction.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd


class SAFEDataLoader:
    """Loader for SAFE dataset audio files."""
    
    def __init__(self, data_dir: str = "../data", target_sr: int = 22050):
        """
        Initialize the SAFE data loader.
        
        Args:
            data_dir: Directory containing SAFE dataset audio files
            target_sr: Target sampling rate for audio (default: 22050 Hz)
        """
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr
        self.audio_files = []
        self.labels = []
        self.fold_ids: Optional[np.ndarray] = None
        
    def load_dataset(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load all audio files from the dataset.
        
        Returns:
            Tuple of (audio_arrays, labels) where:
            - audio_arrays: List of audio arrays (one per file)
            - labels: Binary labels (1 for fall, 0 for non-fall)
        """
        print("Loading SAFE dataset...")
        
        # Get all .wav files
        wav_files = sorted(list(self.data_dir.glob("*.wav")))
        
        if len(wav_files) == 0:
            raise ValueError(f"No .wav files found in {self.data_dir}")
        
        print(f"Found {len(wav_files)} audio files")
        
        audio_arrays = []
        labels = []
        fold_ids_list: List[int] = []
        
        for wav_file in wav_files:
            try:
                # Load audio file
                audio, sr = librosa.load(str(wav_file), sr=self.target_sr, mono=True)
                
                audio_arrays.append(audio)
                
                # Determine label from filename
                # SAFE dataset filename format: AA-BBB-CC-DDD-FF
                # Where: AA - Fold number (01 to 10)
                #        BBB - Numeric code for sample shuffling
                #        CC - ID of the environment
                #        DDD - Sequence number of original video clip
                #        FF - Class (01 for fall, 02 for no fall)
                filename_parts = wav_file.stem.split('-')
                
                if len(filename_parts) >= 5:
                    # Fold index AA (01–10) for paper-style 10-fold grouping (Table 5 CV)
                    try:
                        fold_id = int(filename_parts[0])
                    except ValueError:
                        fold_id = -1
                    class_code = filename_parts[-1]  # Get last part (FF)
                    if class_code == '01':
                        label = 1  # Fall event
                    elif class_code == '02':
                        label = 0  # Non-fall event
                    else:
                        # Fallback if class code is unexpected
                        print(f"Warning: Unexpected class code '{class_code}' in {wav_file.name}. Assuming non-fall.")
                        label = 0
                else:
                    # Fallback if filename doesn't match expected format
                    print(f"Warning: Filename '{wav_file.name}' doesn't match expected format AA-BBB-CC-DDD-FF. Skipping.")
                    audio_arrays.pop()  # Remove the audio we just added
                    continue
                
                labels.append(label)
                fold_ids_list.append(fold_id)
                
            except Exception as e:
                print(f"Error loading {wav_file}: {e}")
                continue
        
        self.audio_files = audio_arrays
        self.labels = np.array(labels)
        self.fold_ids = np.array(fold_ids_list, dtype=np.int64)
        
        invalid_folds = np.sum(self.fold_ids < 1)
        if invalid_folds:
            print(f"Warning: {invalid_folds} samples have invalid fold id; GroupKFold may need fallback.")
        
        print(f"Loaded {len(audio_arrays)} files")
        print(f"Fall events: {np.sum(self.labels == 1)}, Non-fall events: {np.sum(self.labels == 0)}")
        
        return audio_arrays, self.labels
    
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split dataset into train and test sets.
        
        Args:
            test_size: Proportion of dataset to include in test split
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        if len(self.audio_files) == 0:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.audio_files,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )
        
        return X_train, X_test, y_train, y_test
