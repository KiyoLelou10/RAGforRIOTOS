#!/usr/bin/env python3
"""
RIOT OS Documentation Triplet Loss Autoencoder Training

This script loads embeddings from the ChromaDB database, trains an autoencoder
with triplet loss supervision to compress 768-dimensional embeddings to 256 dimensions
while ensuring semantic similarity clustering, and saves the compressed embeddings
along with chunks to a JSON file.

Dependencies:
    pip install chromadb torch numpy tqdm scikit-learn

Usage:
    python riot_triplet_train.py --db-path ./riot_vector_db --output triplet_compressed_chunks.json

Author: Assistant
License: MIT
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import chromadb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TripletEmbeddingAutoencoder(nn.Module):
    """Autoencoder with triplet loss supervision for semantic embedding compression."""

    def __init__(self, input_dim: int = 768, compressed_dim: int = 256):
        super(TripletEmbeddingAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.compressed_dim = compressed_dim

        # Encoder layers with progressive dimensionality reduction
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, compressed_dim),
            nn.Tanh()  # Normalize to [-1, 1] range
        )

        # Decoder layers - mirror of encoder
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, input_dim),
            nn.Tanh()  # Match encoder output range
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x):
        """Encode input to compressed representation."""
        return self.encoder(x)

    def decode(self, x):
        """Decode compressed representation back to original dimension."""
        return self.decoder(x)

    def forward(self, x):
        """Full autoencoder forward pass."""
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded


class TripletDataset(Dataset):
    """Dataset for generating triplets for triplet loss training."""

    def __init__(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]],
                 similarity_threshold: float = 0.7):
        self.embeddings = embeddings
        self.chunks = chunks
        self.similarity_threshold = similarity_threshold

        # Pre-compute similarity matrix for efficient triplet mining
        logger.info("Computing similarity matrix for triplet mining...")
        self.similarity_matrix = cosine_similarity(embeddings)

        # Generate triplets
        self.triplets = self._generate_triplets()
        logger.info(f"Generated {len(self.triplets)} triplets")

    def _generate_triplets(self) -> List[Tuple[int, int, int]]:
        """Generate anchor-positive-negative triplets based on semantic similarity."""
        triplets = []
        n_samples = len(self.embeddings)

        for anchor_idx in tqdm(range(n_samples), desc="Generating triplets"):
            # Find similar chunks (potential positives)
            similarities = self.similarity_matrix[anchor_idx]

            # Get indices sorted by similarity (highest first)
            sorted_indices = np.argsort(similarities)[::-1]

            # Find positives (similar but not identical)
            positives = []
            for idx in sorted_indices[1:]:  # Skip self (index 0)
                if similarities[idx] >= self.similarity_threshold:
                    positives.append(idx)
                else:
                    break

            # Find negatives (dissimilar)
            negatives = []
            for idx in sorted_indices[::-1]:  # Start from least similar
                if similarities[idx] < self.similarity_threshold:
                    negatives.append(idx)
                else:
                    break

            # Create triplets
            if len(positives) > 0 and len(negatives) > 0:
                # Sample multiple triplets per anchor
                n_triplets = min(3, len(positives), len(negatives))
                for _ in range(n_triplets):
                    pos_idx = random.choice(positives)
                    neg_idx = random.choice(negatives)
                    triplets.append((anchor_idx, pos_idx, neg_idx))

        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_idx, pos_idx, neg_idx = self.triplets[idx]

        anchor = torch.FloatTensor(self.embeddings[anchor_idx])
        positive = torch.FloatTensor(self.embeddings[pos_idx])
        negative = torch.FloatTensor(self.embeddings[neg_idx])

        return anchor, positive, negative


def triplet_loss(anchor, positive, negative, margin: float = 1.0):
    """
    Triplet loss function.

    Args:
        anchor: Anchor embeddings
        positive: Positive embeddings (similar to anchor)
        negative: Negative embeddings (dissimilar to anchor)
        margin: Margin for triplet loss
    """
    # Calculate distances
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)

    # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


class RIOTTripletTrainer:
    """Trainer for the RIOT embedding autoencoder with triplet loss."""

    def __init__(self, db_path: str = "./riot_vector_db",
                 collection_name: str = "riot_docs"):
        self.db_path = Path(db_path)
        self.collection_name = collection_name

        # Initialize ChromaDB
        self.chroma_client = self._initialize_chromadb()
        self.collection = self._load_collection()

        # Model and training components
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Triplet Trainer initialized with device: {self.device}")

    def _initialize_chromadb(self) -> chromadb.Client:
        """Initialize ChromaDB client."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database path not found: {self.db_path}")

        return chromadb.PersistentClient(path=str(self.db_path))

    def _load_collection(self) -> chromadb.Collection:
        """Load the collection from ChromaDB."""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()
            logger.info(f"Loaded collection '{self.collection_name}' with {count} documents")
            return collection
        except Exception as e:
            raise RuntimeError(f"Failed to load collection '{self.collection_name}': {e}")

    def load_all_embeddings_and_chunks(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load all embeddings and chunks from the database."""
        logger.info("Loading all embeddings and chunks from database...")

        # Get all data from collection
        total_count = self.collection.count()

        # Retrieve all data in batches to avoid memory issues
        batch_size = 1000
        all_embeddings = []
        all_chunks = []

        for i in tqdm(range(0, total_count, batch_size), desc="Loading data"):
            # Get batch results
            results = self.collection.get(
                limit=batch_size,
                offset=i,
                include=["embeddings", "documents", "metadatas"]
            )

            # Check if embeddings list is not empty and has valid data
            if results['embeddings'] is not None and len(results['embeddings']) > 0:
                all_embeddings.extend(results['embeddings'])

            # Create chunk objects
            documents = results.get('documents', [])
            metadatas = results.get('metadatas', [])

            for j in range(len(documents)):
                chunk = {
                    'content': documents[j] if documents else '',
                    'metadata': metadatas[j] if metadatas and j < len(metadatas) else {}
                }
                all_chunks.append(chunk)

        if not all_embeddings:
            raise ValueError("No embeddings found in the database!")

        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        logger.info(f"Loaded {len(all_embeddings)} embeddings with shape {embeddings_array.shape}")
        logger.info(f"Loaded {len(all_chunks)} chunks")

        return embeddings_array, all_chunks

    def prepare_training_data(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> Tuple[
        TripletDataset, TripletDataset]:
        """Prepare embeddings and chunks for triplet training."""
        logger.info("Preparing training data for triplet loss...")

        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            logger.warning("Found NaN or infinite values in embeddings, cleaning...")
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize embeddings
        normalized_embeddings = self.scaler.fit_transform(embeddings)

        # Split into train/validation
        train_embeddings, val_embeddings, train_chunks, val_chunks = train_test_split(
            normalized_embeddings, chunks,
            test_size=0.1,
            random_state=42
        )
        self.similarity_threshold = 0.7
        # Create triplet datasets
        train_dataset = TripletDataset(train_embeddings, train_chunks, similarity_threshold=self.similarity_threshold)
        val_dataset = TripletDataset(val_embeddings, val_chunks, similarity_threshold=self.similarity_threshold)

        logger.info(f"Training set: {len(train_embeddings)} embeddings, {len(train_dataset)} triplets")
        logger.info(f"Validation set: {len(val_embeddings)} embeddings, {len(val_dataset)} triplets")

        return train_dataset, val_dataset

    def create_model(self, input_dim: int, compressed_dim: int = 256) -> TripletEmbeddingAutoencoder:
        """Create and initialize the autoencoder model."""
        model = TripletEmbeddingAutoencoder(input_dim=input_dim, compressed_dim=compressed_dim)
        model.to(self.device)

        logger.info(f"Created triplet autoencoder: {input_dim} -> {compressed_dim} -> {input_dim}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    def train_triplet_autoencoder(self, train_dataset: TripletDataset, val_dataset: TripletDataset,
                                  epochs: int = 100, batch_size: int = 64,
                                  learning_rate: float = 0.001,
                                  lambda_triplet: float = 1.0,
                                  margin: float = 1.0) -> Dict[str, List[float]]:
        """Train the autoencoder with triplet loss."""
        logger.info(f"Starting triplet autoencoder training with λ={lambda_triplet}, margin={margin}...")

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # Initialize model
        input_dim = train_dataset.embeddings.shape[1]
        self.model = self.create_model(input_dim)

        # Loss function and optimizer
        mse_criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_reconstruction': [],
            'train_triplet': [],
            'val_reconstruction': [],
            'val_triplet': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_total_loss = 0.0
            train_recon_loss = 0.0
            train_triplet_loss = 0.0

            for anchor, positive, negative in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                optimizer.zero_grad()

                # Forward pass for all triplet components
                anchor_recon, anchor_encoded = self.model(anchor)
                pos_recon, pos_encoded = self.model(positive)
                neg_recon, neg_encoded = self.model(negative)

                # Reconstruction losses
                anchor_recon_loss = mse_criterion(anchor_recon, anchor)
                pos_recon_loss = mse_criterion(pos_recon, positive)
                neg_recon_loss = mse_criterion(neg_recon, negative)

                total_recon_loss = (anchor_recon_loss + pos_recon_loss + neg_recon_loss) / 3

                # Triplet loss on encoded representations
                triplet_loss_value = triplet_loss(anchor_encoded, pos_encoded, neg_encoded, margin)

                # Combined loss
                total_loss = total_recon_loss + lambda_triplet * triplet_loss_value

                # Add L2 regularization on encoded representations
                l2_encoded = (torch.mean(torch.sum(anchor_encoded ** 2, dim=1)) +
                              torch.mean(torch.sum(pos_encoded ** 2, dim=1)) +
                              torch.mean(torch.sum(neg_encoded ** 2, dim=1))) / 3
                total_loss += 0.001 * l2_encoded

                # Backward pass
                total_loss.backward()
                optimizer.step()

                train_total_loss += total_loss.item()
                train_recon_loss += total_recon_loss.item()
                train_triplet_loss += triplet_loss_value.item()

            # Validation phase
            self.model.eval()
            val_total_loss = 0.0
            val_recon_loss = 0.0
            val_triplet_loss = 0.0

            with torch.no_grad():
                for anchor, positive, negative in val_loader:
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negative = negative.to(self.device)

                    # Forward pass
                    anchor_recon, anchor_encoded = self.model(anchor)
                    pos_recon, pos_encoded = self.model(positive)
                    neg_recon, neg_encoded = self.model(negative)

                    # Losses
                    anchor_recon_loss = mse_criterion(anchor_recon, anchor)
                    pos_recon_loss = mse_criterion(pos_recon, positive)
                    neg_recon_loss = mse_criterion(neg_recon, negative)

                    total_recon_loss = (anchor_recon_loss + pos_recon_loss + neg_recon_loss) / 3
                    triplet_loss_value = triplet_loss(anchor_encoded, pos_encoded, neg_encoded, margin)

                    total_loss = total_recon_loss + lambda_triplet * triplet_loss_value

                    val_total_loss += total_loss.item()
                    val_recon_loss += total_recon_loss.item()
                    val_triplet_loss += triplet_loss_value.item()

            # Calculate average losses
            avg_train_loss = train_total_loss / len(train_loader)
            avg_val_loss = val_total_loss / len(val_loader)
            avg_train_recon = train_recon_loss / len(train_loader)
            avg_train_triplet = train_triplet_loss / len(train_loader)
            avg_val_recon = val_recon_loss / len(val_loader)
            avg_val_triplet = val_triplet_loss / len(val_loader)

            # Update learning rate
            scheduler.step(avg_val_loss)

            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_reconstruction'].append(avg_train_recon)
            history['train_triplet'].append(avg_train_triplet)
            history['val_reconstruction'].append(avg_val_recon)
            history['val_triplet'].append(avg_val_triplet)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Log progress
            if (epoch + 1) % 5 == 0 or epoch < 3:
                logger.info(f"Epoch {epoch + 1}/{epochs}: "
                            f"Train Loss: {avg_train_loss:.6f} "
                            f"(Recon: {avg_train_recon:.6f}, Triplet: {avg_train_triplet:.6f}), "
                            f"Val Loss: {avg_val_loss:.6f} "
                            f"(Recon: {avg_val_recon:.6f}, Triplet: {avg_val_triplet:.6f})")

            # Early stopping
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        logger.info("Triplet autoencoder training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")

        return history

    def compress_all_embeddings(self, original_embeddings: np.ndarray) -> np.ndarray:
        """Compress all embeddings using the trained triplet autoencoder."""
        logger.info("Compressing all embeddings using triplet autoencoder...")

        if self.model is None:
            raise ValueError("Model not trained yet!")

        self.model.eval()

        # Normalize embeddings using the same scaler
        normalized_embeddings = self.scaler.transform(original_embeddings)

        # Convert to tensor
        embeddings_tensor = torch.FloatTensor(normalized_embeddings).to(self.device)

        # Compress in batches
        compressed_embeddings = []
        batch_size = 512

        with torch.no_grad():
            for i in tqdm(range(0, len(embeddings_tensor), batch_size),
                          desc="Compressing embeddings"):
                batch = embeddings_tensor[i:i + batch_size]
                compressed_batch = self.model.encode(batch)
                compressed_embeddings.append(compressed_batch.cpu().numpy())

        # Concatenate all batches
        compressed_array = np.vstack(compressed_embeddings)

        logger.info(f"Compressed embeddings shape: {compressed_array.shape}")

        return compressed_array

    def save_compressed_chunks(self, chunks: List[Dict[str, Any]],
                               compressed_embeddings: np.ndarray,
                               output_file: str):
        """Save chunks with compressed embeddings to JSON file."""
        logger.info(f"Saving triplet compressed chunks to {output_file}...")

        # Prepare output data
        output_chunks = []

        for i, chunk in enumerate(chunks):
            # Get metadata safely
            metadata = chunk.get('metadata', {})
            file_path = metadata.get('file_path', '')

            # Format chunk similar to the search result format
            compressed_chunk = {
                'rank': i + 1,
                'content': chunk.get('content', ''),
                'metadata': metadata,
                'relevance_score': 1.0,
                'file_path': file_path,
                'module': metadata.get('module', ''),
                'category': metadata.get('category', ''),
                'file_name': Path(file_path).name if file_path else '',
                'section': metadata.get('section', ''),
                'tags': metadata.get('tags', ''),
                'triplet_compressed_embedding': compressed_embeddings[i].tolist()
            }
            output_chunks.append(compressed_chunk)

        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(output_chunks)} triplet compressed chunks to {output_file}")

    def save_model_and_scaler(self, model_path: str, scaler_path: str):
        """Save the trained triplet model and scaler."""
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'compressed_dim': self.model.compressed_dim,
            'model_class': TripletEmbeddingAutoencoder,
            'model_type': 'TripletAutoencoder'
        }, model_path)

        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        logger.info(f"Triplet model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")

    def train_and_save(self, output_chunks_file: str,
                       model_file: str = "triplet_autoencoder_model.pth",
                       scaler_file: str = "triplet_embedding_scaler.pkl",
                       epochs: int = 100,
                       batch_size: int = 64,
                       learning_rate: float = 0.001,
                       lambda_triplet: float = 1.0,
                       margin: float = 1.0):
        """Complete triplet training pipeline."""
        try:
            # Load data
            embeddings, chunks = self.load_all_embeddings_and_chunks()

            # Prepare training data
            train_dataset, val_dataset = self.prepare_training_data(embeddings, chunks)

            # Train triplet autoencoder
            history = self.train_triplet_autoencoder(
                train_dataset, val_dataset,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                lambda_triplet=lambda_triplet,
                margin=margin
            )

            # Compress all embeddings
            compressed_embeddings = self.compress_all_embeddings(embeddings)

            # Save compressed chunks
            self.save_compressed_chunks(chunks, compressed_embeddings, output_chunks_file)

            # Save model and scaler
            self.save_model_and_scaler(model_file, scaler_file)

            logger.info("Triplet training and compression pipeline completed successfully!")

            return history

        except Exception as e:
            logger.error(f"Error in triplet training pipeline: {e}")
            raise


def main():
    """Main function for the triplet autoencoder training."""
    parser = argparse.ArgumentParser(
        description='RIOT OS Documentation Triplet Loss Autoencoder Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python riot_triplet_train.py --output triplet_compressed_chunks.json
  python riot_triplet_train.py --db-path ./riot_vector_db --epochs 150 --lambda-triplet 0.5
        """
    )

    parser.add_argument('--db-path', default="./riot_vector_db",
                        help='Path to ChromaDB database')
    parser.add_argument('--collection', default="riot_docs",
                        help='Collection name in ChromaDB')
    parser.add_argument('--output', default="triplet_compressed_chunks.json",
                        help='Output JSON file for triplet compressed chunks')
    parser.add_argument('--model-file', default="triplet_autoencoder_model.pth",
                        help='Output file for trained triplet model')
    parser.add_argument('--scaler-file', default="triplet_embedding_scaler.pkl",
                        help='Output file for embedding scaler')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--lambda-triplet', type=float, default=1.0,
                        help='Weight for triplet loss term')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for triplet loss')
    parser.add_argument('--similarity-threshold', type=float, default=0.7,
                        help='Cosine‐similarity cutoff for positives vs negatives')
    parser.add_argument('--compressed-dim', type=int, default=256,
                        help='Compressed embedding dimension')

    args = parser.parse_args()
    similarity_threshold = args.similarity_threshold

    try:
        # Initialize trainer
        trainer = RIOTTripletTrainer(
            db_path=args.db_path,
            collection_name=args.collection
        )

        # Run training pipeline
        history = trainer.train_and_save(
            output_chunks_file=args.output,
            model_file=args.model_file,
            scaler_file=args.scaler_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lambda_triplet=args.lambda_triplet,
            margin=args.margin
        )

        # Print training summary
        print(f"\nTriplet Training Summary:")
        print(f"Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        print(f"Final reconstruction loss: {history['train_reconstruction'][-1]:.6f}")
        print(f"Final triplet loss: {history['train_triplet'][-1]:.6f}")
        print(f"Triplet compressed chunks saved to: {args.output}")
        print(f"Triplet model saved to: {args.model_file}")
        print(f"Scaler saved to: {args.scaler_file}")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())