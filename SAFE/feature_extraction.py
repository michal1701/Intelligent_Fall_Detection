"""
Spectrogram feature extraction for SAFE dataset.
Implements various spectrogram-based feature extraction methods as described in SAFE paper Section 4.2.
"""

import numpy as np
import librosa
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler


class SpectrogramFeatureExtractor:
    """
    Extract spectrogram-based features from audio signals.
    Implements methods from SAFE paper Section 4.2.
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 13,
        sr: int = 22050
    ):
        """
        Initialize feature extractor.
        
        Args:
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of Mel filterbanks
            n_mfcc: Number of MFCC coefficients
            sr: Sampling rate
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.scaler = StandardScaler()
        
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Mel spectrogram features.
        
        Args:
            audio: Audio signal array
            
        Returns:
            Mel spectrogram (n_mels, time_frames)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Chroma features.
        
        Args:
            audio: Audio signal array
            
        Returns:
            Chroma features (12, time_frames)
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return chroma
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features.
        
        Args:
            audio: Audio signal array
            
        Returns:
            MFCC features (n_mfcc, time_frames)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc
    
    def extract_spectral_contrast(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Spectral Contrast features.
        
        Args:
            audio: Audio signal array
            
        Returns:
            Spectral contrast features
        """
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return spectral_contrast
    
    def extract_tonnetz(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Tonnetz features.
        
        Args:
            audio: Audio signal array
            
        Returns:
            Tonnetz features
        """
        tonnetz = librosa.feature.tonnetz(
            y=audio,
            sr=self.sr,
            chroma=librosa.feature.chroma_stft(
                y=audio,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
        )
        return tonnetz
    
    def extract_stft_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract STFT (Short-Time Fourier Transform) Spectrogram.
        Table 4 - SAFE paper Section 5.2.
        
        Args:
            audio: Audio signal array
            
        Returns:
            STFT spectrogram (frequency_bins, time_frames) in dB
        """
        stft = librosa.stft(
            y=audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        magnitude = np.abs(stft)
        stft_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        return stft_db
    
    def extract_cqt_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract CQT (Constant Q Transform) Spectrogram.
        Table 4 - SAFE paper Section 5.2.
        
        Args:
            audio: Audio signal array
            
        Returns:
            CQT spectrogram in dB
        """
        cqt = librosa.cqt(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length
        )
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        return cqt_db
    
    def extract_cwt_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract CWT (Continuous Wavelet Transform) Spectrogram.
        Table 4 - SAFE paper Section 5.2. Uses PyWavelets with Morlet wavelet.
        
        Args:
            audio: Audio signal array
            
        Returns:
            CWT spectrogram in dB
        """
        import pywt
        # Scale range for time-frequency resolution (higher = lower freq)
        scales = np.arange(1, 128)
        # Complex Morlet wavelet (cmor); alternative: 'morl' for real Morlet
        coefficients, _ = pywt.cwt(
            audio,
            scales,
            "cmor1.5-1.0",
            sampling_period=1.0 / self.sr,
        )
        cwt_magnitude = np.abs(coefficients)
        cwt_db = librosa.amplitude_to_db(cwt_magnitude, ref=np.max)
        return cwt_db
    
    def extract_all_features(self, audio: np.ndarray, flatten: bool = True) -> np.ndarray:
        """
        Extract all spectrogram features and optionally flatten them.
        
        Args:
            audio: Audio signal array
            flatten: Whether to flatten features into a 1D vector
            
        Returns:
            Feature vector (flattened) or dictionary of features
        """
        features = {
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'chroma': self.extract_chroma(audio),
            'mfcc': self.extract_mfcc(audio),
            'spectral_contrast': self.extract_spectral_contrast(audio),
            'tonnetz': self.extract_tonnetz(audio)
        }
        
        if flatten:
            # Flatten each feature and concatenate
            feature_vector = np.concatenate([
                features['mel_spectrogram'].flatten(),
                features['chroma'].flatten(),
                features['mfcc'].flatten(),
                features['spectral_contrast'].flatten(),
                features['tonnetz'].flatten()
            ])
            return feature_vector
        else:
            return features
    
    def extract_statistical_features(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from a spectrogram (mean, std, min, max per bin).
        Useful for reducing dimensionality while preserving key information.
        
        Args:
            spectrogram: Spectrogram array (frequency_bins, time_frames)
            
        Returns:
            Statistical features vector
        """
        stats = np.concatenate([
            np.mean(spectrogram, axis=1),  # Mean per frequency bin
            np.std(spectrogram, axis=1),   # Std per frequency bin
            np.min(spectrogram, axis=1),   # Min per frequency bin
            np.max(spectrogram, axis=1)    # Max per frequency bin
        ])
        return stats
    
    def extract_features_batch(
        self,
        audio_list: List[np.ndarray],
        feature_type: str = 'mel_spectrogram',
        use_stats: bool = True
    ) -> np.ndarray:
        """
        Extract features for a batch of audio files.
        
        Args:
            audio_list: List of audio arrays
            feature_type: Type of feature to extract (Table 4: 'mel_spectrogram', 'stft_spectrogram', 'mfcc', 'cqt_spectrogram', 'cwt_spectrogram', 'chroma'; or 'spectral_contrast', 'tonnetz', 'all')
            use_stats: If True, extract statistical features instead of full spectrograms
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        features = []
        
        for audio in audio_list:
            if feature_type == 'mel_spectrogram':
                spec = self.extract_mel_spectrogram(audio)
            elif feature_type == 'stft_spectrogram':
                spec = self.extract_stft_spectrogram(audio)
            elif feature_type == 'mfcc':
                spec = self.extract_mfcc(audio)
            elif feature_type == 'cqt_spectrogram':
                spec = self.extract_cqt_spectrogram(audio)
            elif feature_type == 'cwt_spectrogram':
                spec = self.extract_cwt_spectrogram(audio)
            elif feature_type == 'chroma':
                spec = self.extract_chroma(audio)
            elif feature_type == 'spectral_contrast':
                spec = self.extract_spectral_contrast(audio)
            elif feature_type == 'tonnetz':
                spec = self.extract_tonnetz(audio)
            elif feature_type == 'all':
                feature_vec = self.extract_all_features(audio, flatten=True)
                features.append(feature_vec)
                continue
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")
            
            if use_stats:
                # Use statistical features for dimensionality reduction
                feature_vec = self.extract_statistical_features(spec)
            else:
                # Flatten the full spectrogram
                feature_vec = spec.flatten()
            
            features.append(feature_vec)
        
        return np.array(features)
    
    def fit_scaler(self, X: np.ndarray):
        """Fit the standard scaler on training data."""
        self.scaler.fit(X)
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features using fitted scaler."""
        return self.scaler.transform(X)
