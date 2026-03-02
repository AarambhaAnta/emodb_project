"""Extract ECAPA-TDNN embeddings for PLDA training."""
import os
import logging

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import hyperpyyaml

import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: 'soundfile'

logger = logging.getLogger(__name__)

_HPARAMS = str(Path(__file__).parent.parent.parent / 'config' / 'ecapa_hparams.yaml')


def _load_model(model_dir, hparams_file=None, device='cpu'):
    """Load embedding model + normalizer from model.pt."""
    with open(hparams_file or _HPARAMS) as f:
        hparams = hyperpyyaml.load_hyperpyyaml(f)
    emb_model  = hparams['embedding_model']
    normalizer = hparams['mean_var_norm']
    payload = torch.load(Path(model_dir) / 'model.pt', map_location='cpu')
    emb_model.load_state_dict(payload['embedding_model'])
    normalizer.load_state_dict(payload['normalizer'], strict=False)
    emb_model.eval()
    return emb_model.to(device), normalizer.to(device)


def _extract_one(mfcc_path, emb_model, normalizer, device):
    """Extract embedding from a single MFCC .npy file."""
    feat = np.squeeze(np.load(mfcc_path))
    if feat.ndim == 2 and feat.shape[0] <= 256 and feat.shape[1] > feat.shape[0]:
        feat = feat.T
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.clamp(
        torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device),
        -50.0, 50.0,
    )
    lens = torch.tensor([1.0], device=device)
    t = normalizer(t, lens)
    with torch.no_grad():
        emb = emb_model(t, lens)
    return emb.squeeze().cpu().numpy()


def _extract_split(csv_path, emb_model, normalizer, out_dir, out_csv, device):
    """Extract embeddings for all rows in csv_path; write embeddings CSV."""
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=Path(csv_path).stem, leave=False):
        sample_id = str(row.get('ID', row.get('id', idx)))
        mfcc_path = str(row['mfcc'])
        if not os.path.exists(mfcc_path):
            logger.warning(f"Missing MFCC: {mfcc_path}")
            continue
        try:
            emb = _extract_one(mfcc_path, emb_model, normalizer, device)
            emb_path = os.path.join(out_dir, f"{sample_id}.npy")
            np.save(emb_path, emb)
            rows.append({**row.to_dict(), 'embedding': emb_path})
        except Exception as e:
            logger.warning(f"Failed {sample_id}: {e}")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info(f"Saved {len(rows)}/{len(df)} embeddings → {out_csv}")


def extract_embeddings_for_speaker(speaker_id, loso_dir, model_dir, output_dir,
                                    hparams_file=None, device='cpu'):
    """Extract train/val/test embeddings for one speaker; save per-split CSVs."""
    spk_loso  = os.path.join(loso_dir,   f"speaker_{speaker_id}")
    spk_model = os.path.join(model_dir,  f"speaker_{speaker_id}")
    spk_out   = os.path.join(output_dir, f"speaker_{speaker_id}")

    emb_model, normalizer = _load_model(spk_model, hparams_file, device)

    for split in ('train', 'val', 'test'):
        csv_path = os.path.join(spk_loso, f"{split}.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"Missing {split}.csv for speaker {speaker_id}")
            continue
        _extract_split(
            csv_path, emb_model, normalizer,
            out_dir=os.path.join(spk_out, split),
            out_csv=os.path.join(spk_out, f"{split}_embeddings.csv"),
            device=device,
        )

    logger.info(f"Speaker {speaker_id}: embeddings done → {spk_out}")
