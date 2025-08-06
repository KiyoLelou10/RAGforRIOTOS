#!/usr/bin/env python3
"""
RIOT OS Documentation Embedding Autoencoder Training

This script loads embeddings from the ChromaDB database, trains an autoencoder
to compress 768-dimensional embeddings to 256 dimensions, and saves the
compressed embeddings along with chunks to a JSON file.

Dependencies:
    pip install chromadb torch numpy tqdm scikit-learn

Usage:
    python riot_autoencoder_train.py --db-path ./riot_vector_db --output compressed_chunks.json

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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import chromadb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingAutoencoder(nn.Module):
    """Autoencoder for compressing embeddings from 768 to 256 dimensions."""

    def __init__(self, input_dim: int = 768, compressed_dim: int = 256):
        super(EmbeddingAutoencoder, self).__init__()

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


class RIOTAutoencoderTrainer:
    """Trainer for the RIOT embedding autoencoder."""

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

        logger.info(f"Trainer initialized with device: {self.device}")

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

            # Fixed: Check if embeddings list is not empty and has valid data
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

    def prepare_training_data(self, embeddings: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare embeddings for training."""
        logger.info("Preparing training data...")

        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            logger.warning("Found NaN or infinite values in embeddings, cleaning...")
            # Replace NaN and inf with 0
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize embeddings
        normalized_embeddings = self.scaler.fit_transform(embeddings)

        # Split into train/validation
        train_embeddings, val_embeddings = train_test_split(
            normalized_embeddings,
            test_size=0.1,
            random_state=42
        )

        # Convert to PyTorch tensors
        train_tensor = torch.FloatTensor(train_embeddings)
        val_tensor = torch.FloatTensor(val_embeddings)

        logger.info(f"Training set shape: {train_tensor.shape}")
        logger.info(f"Validation set shape: {val_tensor.shape}")

        return train_tensor, val_tensor

    def create_model(self, input_dim: int, compressed_dim: int = 256) -> EmbeddingAutoencoder:
        """Create and initialize the autoencoder model."""
        model = EmbeddingAutoencoder(input_dim=input_dim, compressed_dim=compressed_dim)
        model.to(self.device)

        logger.info(f"Created autoencoder: {input_dim} -> {compressed_dim} -> {input_dim}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    def train_autoencoder(self, train_data: torch.Tensor, val_data: torch.Tensor,
                          epochs: int = 100, batch_size: int = 128,
                          learning_rate: float = 0.001) -> Dict[str, List[float]]:
        """Train the autoencoder model."""
        logger.info("Starting autoencoder training...")

        # Create data loaders
        train_dataset = TensorDataset(train_data, train_data)  # Input = Target for autoencoder
        val_dataset = TensorDataset(val_data, val_data)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        input_dim = train_data.shape[1]
        self.model = self.create_model(input_dim)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'reconstruction_error': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_data, batch_target in train_loader:
                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                reconstructed, encoded = self.model(batch_data)

                # Calculate loss
                loss = criterion(reconstructed, batch_target)

                # Add L2 regularization on encoded representations
                l2_encoded = torch.mean(torch.sum(encoded ** 2, dim=1))
                loss += 0.001 * l2_encoded

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            reconstruction_errors = []

            with torch.no_grad():
                for batch_data, batch_target in val_loader:
                    batch_data = batch_data.to(self.device)
                    batch_target = batch_target.to(self.device)

                    reconstructed, encoded = self.model(batch_data)
                    loss = criterion(reconstructed, batch_target)
                    val_loss += loss.item()

                    # Calculate reconstruction error
                    batch_error = torch.mean(torch.sum((reconstructed - batch_target) ** 2, dim=1))
                    reconstruction_errors.append(batch_error.item())

            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_reconstruction_error = np.mean(reconstruction_errors)

            # Update learning rate
            scheduler.step(avg_val_loss)

            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['reconstruction_error'].append(avg_reconstruction_error)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Log progress
            if (epoch + 1) % 10 == 0 or epoch < 5:
                logger.info(f"Epoch {epoch + 1}/{epochs}: "
                            f"Train Loss: {avg_train_loss:.6f}, "
                            f"Val Loss: {avg_val_loss:.6f}, "
                            f"Reconstruction Error: {avg_reconstruction_error:.6f}")

            # Early stopping
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")

        return history

    def compress_all_embeddings(self, original_embeddings: np.ndarray) -> np.ndarray:
        """Compress all embeddings using the trained autoencoder."""
        logger.info("Compressing all embeddings...")

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
        logger.info(f"Saving compressed chunks to {output_file}...")

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
                'relevance_score': 1.0,  # Placeholder, will be calculated during search
                'file_path': file_path,
                'module': metadata.get('module', ''),
                'category': metadata.get('category', ''),
                'file_name': Path(file_path).name if file_path else '',
                'section': metadata.get('section', ''),
                'tags': metadata.get('tags', ''),
                'compressed_embedding': compressed_embeddings[i].tolist()
            }
            output_chunks.append(compressed_chunk)

        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(output_chunks)} compressed chunks to {output_file}")

    def save_model_and_scaler(self, model_path: str, scaler_path: str):
        """Save the trained model and scaler."""
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'compressed_dim': self.model.compressed_dim,
            'model_class': EmbeddingAutoencoder
        }, model_path)

        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")

    def train_and_save(self, output_chunks_file: str,
                       model_file: str = "autoencoder_model.pth",
                       scaler_file: str = "embedding_scaler.pkl",
                       epochs: int = 100,
                       batch_size: int = 128,
                       learning_rate: float = 0.001):
        """Complete training pipeline."""
        try:
            # Load data
            embeddings, chunks = self.load_all_embeddings_and_chunks()

            # Prepare training data
            train_data, val_data = self.prepare_training_data(embeddings)

            # Train autoencoder
            history = self.train_autoencoder(
                train_data, val_data,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )

            # Compress all embeddings
            compressed_embeddings = self.compress_all_embeddings(embeddings)

            # Save compressed chunks
            self.save_compressed_chunks(chunks, compressed_embeddings, output_chunks_file)

            # Save model and scaler
            self.save_model_and_scaler(model_file, scaler_file)

            logger.info("Training and compression pipeline completed successfully!")

            return history

        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise


def main():
    """Main function for the autoencoder training."""
    parser = argparse.ArgumentParser(
        description='RIOT OS Documentation Embedding Autoencoder Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python riot_autoencoder_train.py --output compressed_chunks.json
  python riot_autoencoder_train.py --db-path ./riot_vector_db --epochs 150 --batch-size 256
        """
    )

    parser.add_argument('--db-path', default="./riot_vector_db",
                        help='Path to ChromaDB database')
    parser.add_argument('--collection', default="riot_docs",
                        help='Collection name in ChromaDB')
    parser.add_argument('--output', default="compressed_chunks.json",
                        help='Output JSON file for compressed chunks')
    parser.add_argument('--model-file', default="autoencoder_model.pth",
                        help='Output file for trained model')
    parser.add_argument('--scaler-file', default="embedding_scaler.pkl",
                        help='Output file for embedding scaler')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--compressed-dim', type=int, default=256,
                        help='Compressed embedding dimension')

    args = parser.parse_args()

    try:
        # Initialize trainer
        trainer = RIOTAutoencoderTrainer(
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
            learning_rate=args.learning_rate
        )

        # Print training summary
        print(f"\nTraining Summary:")
        print(f"Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        print(f"Final reconstruction error: {history['reconstruction_error'][-1]:.6f}")
        print(f"Compressed chunks saved to: {args.output}")
        print(f"Model saved to: {args.model_file}")
        print(f"Scaler saved to: {args.scaler_file}")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())