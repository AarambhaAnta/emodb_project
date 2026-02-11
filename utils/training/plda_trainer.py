"""
SpeechBrain PLDA training utilities for embeddings.

This module trains a true SpeechBrain PLDA model from embeddings using
StatObject_SB. The PLDA model is saved per speaker.

Typical workflow:
1. Extract embeddings from trained ECAPA-TDNN models
2. Pack embeddings into StatObject_SB
3. Train SpeechBrain PLDA
4. Save PLDA model per speaker
"""
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from speechbrain.processing.PLDA_LDA import PLDA, StatObject_SB


def _length_normalize(embeddings, eps=1e-12):
    """Apply L2 length normalization to embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return embeddings / norms


def _whiten_embeddings(embeddings, eps=1e-6):
    """Whiten embeddings with diagonal regularization."""
    mean = np.mean(embeddings, axis=0, keepdims=True)
    centered = embeddings - mean
    cov = np.cov(centered, rowvar=False)
    cov = cov + np.eye(cov.shape[0]) * eps
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, eps)
    whiten_mat = evecs @ np.diag(1.0 / np.sqrt(evals))
    whitened = centered @ whiten_mat
    return whitened, mean.squeeze(), whiten_mat


class PLDAModel:
    """Deprecated PLDA-style classifier wrapper (kept for backward compatibility)."""
    
    def __init__(self, embedding_dim=192, use_lda=True, lda_dim=None, 
                 classifier_type='logistic'):
        """
        Initialize PLDA-style model.
        
        Args:
            embedding_dim: Dimension of input embeddings (e.g., 192 for ECAPA-TDNN)
            use_lda: Whether to apply LDA before classification
            lda_dim: LDA output dimension (None = n_classes - 1)
            classifier_type: 'logistic' or 'svm' for probabilistic classification
        """
        self.embedding_dim = embedding_dim
        self.use_lda = use_lda
        self.lda_dim = lda_dim
        self.classifier_type = classifier_type
        
        # Initialize models
        self.scaler = StandardScaler()
        self.lda = None
        self.classifier = None
        
        if use_lda:
            self.lda = LinearDiscriminantAnalysis(n_components=lda_dim)
        
        # Initialize classifier
        if classifier_type == 'logistic':
            self.classifier = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        elif classifier_type == 'svm':
            self.classifier = SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")
        
        self.is_fitted = False
        self.emotion_labels = {
            0: 'Happiness',
            1: 'Neutral',
            2: 'Anger',
            3: 'Fear',
            4: 'Boredom',
            5: 'Disgust',
            6: 'Sadness'
        }
        self.classes_ = None
    
    def fit(self, X, y):
        """
        Fit PLDA-style model to training embeddings.
        
        Args:
            X: Training embeddings (n_samples, embedding_dim)
            y: Training labels (n_samples,)
        
        Returns:
            self
        """
        print(f"Training PLDA-style model on {X.shape[0]} samples with {X.shape[1]}-dim embeddings...")
        
        # Standardize embeddings
        X_scaled = self.scaler.fit_transform(X)
        
        # Store unique classes
        self.classes_ = np.unique(y)
        
        # Apply LDA for dimensionality reduction if enabled
        if self.use_lda and self.lda is not None:
            print(f"  Applying LDA dimensionality reduction...")
            X_reduced = self.lda.fit_transform(X_scaled, y)
            print(f"  LDA: {X_scaled.shape[1]} → {X_reduced.shape[1]} dimensions")
        else:
            X_reduced = X_scaled
        
        # Train probabilistic classifier
        print(f"  Training {self.classifier_type} classifier...")
        self.classifier.fit(X_reduced, y)
        print(f"  ✓ Classifier trained")
        
        self.is_fitted = True
        print(f"✓ PLDA-style model fitted successfully")
        
        return self
    
    def transform(self, X):
        """
        Transform embeddings through standardization and optional LDA.
        
        Args:
            X: Embeddings (n_samples, embedding_dim)
        
        Returns:
            X_transformed: Transformed embeddings
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.use_lda and self.lda is not None:
            return self.lda.transform(X_scaled)
        return X_scaled
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Embeddings (n_samples, embedding_dim)
        
        Returns:
            predictions: Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Transform embeddings
        X_transformed = self.transform(X)
        
        # Predict using classifier
        predictions = self.classifier.predict(X_transformed)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Embeddings (n_samples, embedding_dim)
        
        Returns:
            probabilities: Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Transform embeddings
        X_transformed = self.transform(X)
        
        # Get probabilities
        return self.classifier.predict_proba(X_transformed)
    
    def score(self, X, y):
        """
        Compute accuracy score.
        
        Args:
            X: Embeddings (n_samples, embedding_dim)
            y: True labels (n_samples,)
        
        Returns:
            accuracy: Classification accuracy
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def evaluate(self, X, y, verbose=True):
        """
        Comprehensive evaluation with metrics.
        
        Args:
            X: Embeddings (n_samples, embedding_dim)
            y: True labels (n_samples,)
            verbose: Print detailed report
        
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'error_rate': 1 - accuracy,
            'n_samples': len(y),
            'confusion_matrix': confusion_matrix(y, predictions).tolist()
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print("PLDA Evaluation Results")
            print(f"{'='*70}")
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Error Rate: {metrics['error_rate']:.4f}")
            print(f"Samples: {metrics['n_samples']}")
            print(f"\nClassification Report:")
            print(classification_report(
                y, predictions,
                target_names=[self.emotion_labels[i] for i in sorted(self.emotion_labels.keys())],
                zero_division=0
            ))
            print(f"\nConfusion Matrix:")
            print(confusion_matrix(y, predictions))
            print(f"{'='*70}")
        
        return metrics
    
    def save(self, filepath):
        """
        Save PLDA model to disk.
        
        Args:
            filepath: Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'scaler': self.scaler,
            'lda': self.lda,
            'classifier': self.classifier,
            'classifier_type': self.classifier_type,
            'embedding_dim': self.embedding_dim,
            'use_lda': self.use_lda,
            'lda_dim': self.lda_dim,
            'is_fitted': self.is_fitted,
            'emotion_labels': self.emotion_labels,
            'classes_': self.classes_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ PLDA model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load PLDA model from disk.
        
        Args:
            filepath: Path to saved model
        
        Returns:
            model: Loaded PLDAModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Get classifier_type with fallback
        classifier_type = model_data.get('classifier_type', 'logistic')
        
        model = cls(
            embedding_dim=model_data['embedding_dim'],
            use_lda=model_data['use_lda'],
            lda_dim=model_data['lda_dim'],
            classifier_type=classifier_type
        )
        
        model.scaler = model_data['scaler']
        model.lda = model_data['lda']
        model.classifier = model_data['classifier']
        model.is_fitted = model_data['is_fitted']
        model.emotion_labels = model_data['emotion_labels']
        model.classes_ = model_data.get('classes_')
        
        print(f"✓ PLDA model loaded from {filepath}")
        return model


def load_embeddings_from_npy(embedding_dir, csv_path=None):
    """
    Load pre-extracted embeddings from .npy files.
    
    Args:
        embedding_dir: Directory containing embedding .npy files
        csv_path: Optional CSV with metadata
    
    Returns:
        embeddings: Loaded embeddings (n_samples, embedding_dim)
        labels: Corresponding labels (n_samples,)
        ids: Sample IDs
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        embeddings = []
        labels = []
        ids = []
        
        print(f"Loading embeddings from {embedding_dir}...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading embeddings"):
            # Construct embedding file path
            sample_id = row.get('id', idx)
            emb_path = os.path.join(embedding_dir, f"{sample_id}.npy")
            
            if not os.path.exists(emb_path):
                print(f"Warning: Missing embedding file {emb_path}, skipping...")
                continue
            
            emb = np.load(emb_path)
            emb = np.asarray(emb, dtype=np.float64)
            if np.isnan(emb).any() or np.isinf(emb).any():
                emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            embeddings.append(emb)
            labels.append(row['label'])
            ids.append(sample_id)
        
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        print(f"✓ Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        
        return embeddings, labels, ids
    else:
        # Load all .npy files from directory
        print(f"Loading all embeddings from {embedding_dir}...")
        
        embedding_files = sorted([f for f in os.listdir(embedding_dir) if f.endswith('.npy')])
        
        embeddings = []
        ids = []
        
        for emb_file in tqdm(embedding_files, desc="Loading embeddings"):
            emb_path = os.path.join(embedding_dir, emb_file)
            emb = np.load(emb_path)
            emb = np.asarray(emb, dtype=np.float64)
            if np.isnan(emb).any() or np.isinf(emb).any():
                emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            embeddings.append(emb)
            ids.append(emb_file.replace('.npy', ''))
        
        embeddings = np.array(embeddings)
        
        print(f"✓ Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        
        return embeddings, None, ids


def _build_stat_object(embeddings, labels, ids):
    """Build SpeechBrain StatObject_SB from embeddings and labels."""
    modelset = np.array([str(label) for label in labels], dtype="|O")
    segset = np.array([str(seg_id) for seg_id in ids], dtype="|O")
    start = np.array([None] * len(ids))
    stop = np.array([None] * len(ids))
    stat0 = np.ones((len(ids), 1), dtype=np.float64)
    stat1 = np.asarray(embeddings, dtype=np.float64)

    return StatObject_SB(
        modelset=modelset,
        segset=segset,
        start=start,
        stop=stop,
        stat0=stat0,
        stat1=stat1,
    )


def train_speaker_plda(
    speaker_id,
    loso_dir=None,
    output_dir=None,
    embeddings_dir=None,
    embeddings_split="dev",
    plda_dim=None,
    plda_iters=10
):
    """
    Train PLDA model for a single speaker using LOSO setup.
    
    Args:
        speaker_id: Speaker ID (e.g., '03')
        loso_dir: Base LOSO directory
        output_dir: Output directory for models
        embeddings_dir: Directory with pre-extracted embeddings
        embeddings_split: Split name under each speaker (e.g., dev, other)
        use_lda: Whether to use LDA dimensionality reduction
        lda_dim: LDA output dimension
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    if loso_dir is None:
        loso_dir = '/Users/adityakumar/Developer/Projects/emodb_project/data/processed/loso'
    
    if output_dir is None:
        output_dir = '/Users/adityakumar/Developer/Projects/emodb_project/output/models/plda'
    
    speaker_dir = os.path.join(loso_dir, f"speaker_{speaker_id}")
    train_csv = os.path.join(speaker_dir, "dev.csv")
    test_csv = os.path.join(speaker_dir, "test.csv")
    
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    
    print(f"\n{'='*70}")
    print(f"Training PLDA for Speaker {speaker_id}")
    print(f"{'='*70}")
    
    # Load embeddings
    if embeddings_dir and os.path.exists(embeddings_dir):
        print("Loading pre-extracted embeddings...")
        speaker_emb_dir = os.path.join(embeddings_dir, f"speaker_{speaker_id}")
        train_emb_dir = os.path.join(speaker_emb_dir, embeddings_split)
        train_emb_csv = os.path.join(speaker_emb_dir, f"{embeddings_split}_embeddings.csv")
        
        if not os.path.exists(train_emb_csv):
            raise FileNotFoundError(
                f"Training embeddings CSV not found for split '{embeddings_split}': {train_emb_csv}"
            )
        
        X_train, y_train, train_ids = load_embeddings_from_npy(
            train_emb_dir,
            train_emb_csv
        )
        
        # Optional evaluation on test split if available
        X_val = y_val = val_ids = None
        test_emb_dir = os.path.join(speaker_emb_dir, "test")
        test_emb_csv = os.path.join(speaker_emb_dir, "test_embeddings.csv")
        if os.path.exists(test_emb_csv):
            X_val, y_val, val_ids = load_embeddings_from_npy(
                test_emb_dir,
                test_emb_csv
            )
    else:
        raise ValueError("Embeddings not available. Please extract embeddings from ECAPA-TDNN models first.")
    
    # Normalize and whiten embeddings for numerical stability
    X_train = _length_normalize(X_train)
    X_train, whiten_mean, whiten_mat = _whiten_embeddings(X_train)

    # Build StatObject_SB for training
    train_stat = _build_stat_object(X_train, y_train, train_ids)

    # Initialize and train PLDA
    embedding_dim = X_train.shape[1]
    rank_f = plda_dim if plda_dim is not None else min(embedding_dim, 100)
    plda = PLDA(rank_f=rank_f, nb_iter=plda_iters)
    plda.plda(train_stat)

    # Save model
    speaker_output_dir = os.path.join(output_dir, f"speaker_{speaker_id}")
    os.makedirs(speaker_output_dir, exist_ok=True)
    model_path = os.path.join(speaker_output_dir, "plda_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "plda": plda,
                "whiten_mean": whiten_mean,
                "whiten_mat": whiten_mat,
                "length_norm": True,
            },
            f,
            pickle.HIGHEST_PROTOCOL,
        )

    # Prepare results
    results = {
        'speaker': speaker_id,
        'model_path': model_path,
        'n_train_samples': len(train_ids),
        'embedding_dim': embedding_dim,
        'plda_dim': rank_f,
        'plda_iters': plda_iters
    }
    
    # Save results JSON
    results_path = os.path.join(speaker_output_dir, "plda_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    
    return results


def train_all_speakers_plda(
    loso_dir=None,
    output_dir=None,
    embeddings_dir=None,
    embeddings_split="dev",
    plda_dim=None,
    plda_iters=10
):
    """
    Train PLDA models for all speakers using LOSO.
    
    Args:
        loso_dir: Base LOSO directory
        output_dir: Output directory for models
        embeddings_dir: Directory with pre-extracted embeddings
        embeddings_split: Split name under each speaker (e.g., dev, other)
        use_lda: Whether to use LDA dimensionality reduction
        lda_dim: LDA output dimension
    
    Returns:
        all_results: Dictionary with results for all speakers
    """
    if loso_dir is None:
        loso_dir = '/Users/adityakumar/Developer/Projects/emodb_project/data/processed/loso'
    
    if output_dir is None:
        output_dir = '/Users/adityakumar/Developer/Projects/emodb_project/output/models/plda'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all speaker directories
    speaker_dirs = sorted([d for d in os.listdir(loso_dir) if d.startswith('speaker_')])
    speakers = [d.replace('speaker_', '') for d in speaker_dirs]
    
    all_results = {}
    
    for speaker_id in speakers:
        try:
            results = train_speaker_plda(
                speaker_id=speaker_id,
                loso_dir=loso_dir,
                output_dir=output_dir,
                embeddings_dir=embeddings_dir,
                embeddings_split=embeddings_split,
                plda_dim=plda_dim,
                plda_iters=plda_iters
            )
            
            all_results[speaker_id] = results
            
        except Exception as e:
            print(f"\n✗ Failed to train speaker {speaker_id}: {e}")
            all_results[speaker_id] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Save combined results
    combined_results_path = os.path.join(output_dir, "plda_training_results.json")
    with open(combined_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("PLDA Training Summary")
    print(f"{'='*70}")
    
    successful = [s for s, r in all_results.items() if 'model_path' in r]
    failed = [s for s, r in all_results.items() if 'status' in r and r['status'] == 'failed']
    
    print(f"Successful: {len(successful)}/{len(speakers)}")
    print(f"Failed: {len(failed)}/{len(speakers)}")
    
    print(f"\n✓ Combined results saved to {combined_results_path}")
    print(f"{'='*70}")
    
    return all_results
