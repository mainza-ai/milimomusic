"""
HeartMuLa Training Data Preparation Pipeline

Converts audio files + metadata into training-ready token sequences.
"""

import torch
import torchaudio
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Callable
import sys

logger = logging.getLogger(__name__)

@dataclass
class TrainingSample:
    """A single training sample with audio tokens and conditioning."""
    audio_tokens: torch.Tensor  # [seq_len, num_codebooks]
    caption: str
    tags: List[str]
    duration_ms: int
    
    def save(self, path: Path):
        """Save sample to disk."""
        torch.save({
            'audio_tokens': self.audio_tokens,
            'caption': self.caption,
            'tags': self.tags,
            'duration_ms': self.duration_ms
        }, path)
    
    @classmethod
    def load(cls, path: Path) -> 'TrainingSample':
        """Load sample from disk."""
        data = torch.load(path, weights_only=False)
        return cls(**data)


class DataPreparator:
    """Prepares audio files for HeartMuLa training."""
    
    TARGET_SAMPLE_RATE = 48000
    
    def __init__(self, heartcodec_path: str, device: str = "cpu"):
        """
        Initialize data preparator with HeartCodec for tokenization.
        
        Args:
            heartcodec_path: Path to HeartCodec checkpoint
            device: Device to run encoding on
        """
        self.device = device
        self.codec = None
        self.heartcodec_path = heartcodec_path
        
    def _load_codec(self):
        """Lazy load HeartCodec to avoid startup overhead."""
        if self.codec is not None:
            return
            
        try:
            from heartlib.heartcodec.modeling_heartcodec import HeartCodec
            logger.info(f"Loading HeartCodec from {self.heartcodec_path}")
            self.codec = HeartCodec.from_pretrained(
                self.heartcodec_path + "/HeartCodec-oss",
                torch_dtype=torch.float32
            ).to(self.device)
            self.codec.eval()
            logger.info("HeartCodec loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load HeartCodec: {e}")
            raise
    
    def load_audio(self, path: Path) -> Tuple[torch.Tensor, int]:
        """
        Load and resample audio file to target sample rate.
        
        Returns:
            audio: Tensor of shape [1, num_samples]
            duration_ms: Duration in milliseconds
        """
        audio, sr = torchaudio.load(str(path))
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.TARGET_SAMPLE_RATE)
            audio = resampler(audio)
        
        duration_ms = int(audio.shape[1] / self.TARGET_SAMPLE_RATE * 1000)
        
        return audio, duration_ms
    
    def tokenize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio waveform to HeartCodec tokens.
        
        Args:
            audio: Tensor of shape [1, num_samples] or [num_samples]
            
        Returns:
            tokens: Tensor of shape [seq_len, num_codebooks]
        """
        self._load_codec()
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Add batch dimension: [B, 1, T]
        audio = audio.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            tokens = self.codec.encode(audio)
        
        # tokens shape: [B, num_codebooks, seq_len] -> [seq_len, num_codebooks]
        tokens = tokens.squeeze(0).permute(1, 0).cpu()
        
        return tokens
    
    def prepare_sample(
        self,
        audio_path: Path,
        caption: str,
        tags: Optional[List[str]] = None
    ) -> TrainingSample:
        """
        Prepare a single audio file for training.
        
        Args:
            audio_path: Path to audio file
            caption: Text caption/lyrics for the audio
            tags: Optional list of style tags
            
        Returns:
            TrainingSample ready for training
        """
        logger.info(f"Preparing sample: {audio_path.name}")
        
        # Load audio
        audio, duration_ms = self.load_audio(audio_path)
        logger.debug(f"  Loaded audio: {duration_ms}ms, shape {audio.shape}")
        
        # Tokenize
        tokens = self.tokenize_audio(audio)
        logger.debug(f"  Tokenized: shape {tokens.shape}")
        
        # Format caption with HeartMuLa-expected tag structure
        # Per HEARTLIB_BIBLE.md: "<tag> Tag1, Tag2 </tag> [lyrics]"
        formatted_caption = caption
        if tags and len(tags) > 0:
            tag_string = ", ".join(tags)
            formatted_caption = f"<tag> {tag_string} </tag> {caption}"
            logger.debug(f"  Formatted caption with tags: {formatted_caption[:80]}...")
        
        return TrainingSample(
            audio_tokens=tokens,
            caption=formatted_caption,
            tags=tags or [],
            duration_ms=duration_ms
        )
    
    def prepare_dataset(
        self,
        audio_dir: Path,
        manifest_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Path]:
        """
        Prepare an entire dataset for training.
        
        Args:
            audio_dir: Directory containing audio files
            manifest_path: Path to dataset manifest JSON
            output_dir: Directory to save prepared samples
            
        Returns:
            List of paths to prepared .pt files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        prepared_paths = []
        
        audio_files = manifest.get('audio_files', [])
        total_files = len(audio_files)
        
        for i, audio_file in enumerate(audio_files):
            filename = audio_file['filename']
            caption = audio_file.get('caption', filename)
            
            audio_path = audio_dir / filename
            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
                continue
            
            try:
                sample = self.prepare_sample(
                    audio_path=audio_path,
                    caption=caption,
                    tags=manifest.get('styles', [])
                )
                
                if progress_callback:
                    progress_callback(i + 1, total_files, filename)
                
                # Save prepared sample
                output_path = output_dir / f"{audio_path.stem}.pt"
                sample.save(output_path)
                prepared_paths.append(output_path)
                
                logger.info(f"  Saved: {output_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to prepare {filename}: {e}")
                continue
        
        logger.info(f"Prepared {len(prepared_paths)} samples")
        return prepared_paths


class TrainingDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for HeartMuLa training samples."""
    
    def __init__(self, sample_paths: List[Path], max_seq_len: int = 2048):
        self.sample_paths = sample_paths
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.sample_paths)
    
    def __getitem__(self, idx):
        sample = TrainingSample.load(self.sample_paths[idx])
        
        tokens = sample.audio_tokens
        
        # Truncate if too long
        if tokens.shape[0] > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        
        return {
            'audio_tokens': tokens,
            'caption': sample.caption,
            'tags': sample.tags
        }
    
    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader."""
        # Pad sequences to same length
        max_len = max(b['audio_tokens'].shape[0] for b in batch)
        num_codebooks = batch[0]['audio_tokens'].shape[1]
        
        padded_tokens = torch.zeros(len(batch), max_len, num_codebooks, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        
        for i, b in enumerate(batch):
            seq_len = b['audio_tokens'].shape[0]
            padded_tokens[i, :seq_len] = b['audio_tokens']
            attention_mask[i, :seq_len] = True
        
        return {
            'audio_tokens': padded_tokens,
            'attention_mask': attention_mask,
            'captions': [b['caption'] for b in batch],
            'tags': [b['tags'] for b in batch]
        }
