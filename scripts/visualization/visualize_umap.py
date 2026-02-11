"""
UMAP Visualization for Fashion Embeddings
==========================================

Visualize image and JSON embeddings in 2D space using UMAP.
UMAP is faster and preserves global structure better than t-SNE.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm

# UMAP import
try:
    import umap
except ImportError:
    print("Installing UMAP...")
    import subprocess
    subprocess.check_call(["pip", "install", "umap-learn"])
    import umap

from src.data.fashion_dataset import FashionDataModule
from models.contrastive_learner import ContrastiveLearner
from utils.config import TrainingConfig


class UMAPVisualizer:
    """UMAP visualization for fashion embeddings."""

    def __init__(self,
                 model_path: str,
                 dataset_path: str,
                 output_dir: str = "results/umap_visualizations",
                 device: str = "cuda"):
        """
        Initialize UMAP visualizer.

        Args:
            model_path: Path to trained model checkpoint
            dataset_path: Path to dataset
            output_dir: Output directory for visualizations
            device: Device to use (cuda/cpu)
        """
        self.model_path = Path(model_path)
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        print(f"Loading model from {model_path}...")
        self.model, self.vocab_sizes = self._load_model()
        self.model.eval()

    def _load_model(self) -> Tuple[ContrastiveLearner, Dict[str, int]]:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Get vocab sizes from checkpoint
        vocab_sizes = checkpoint.get('vocab_sizes', {
            'category': 23,
            'style': 4,
            'silhouette': 8,
            'material': 25,
            'detail': 41
        })

        # Create config
        config = TrainingConfig()

        # Initialize model
        model = ContrastiveLearner(
            image_encoder_name='patrickjohncyh/fashion-clip',
            vocab_sizes=vocab_sizes,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            temperature=config.temperature,
            dropout_rate=config.dropout_rate,
            finetune_clip=False
        ).to(self.device)

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])

        return model, vocab_sizes

    def extract_embeddings(self,
                          data_module: FashionDataModule,
                          max_samples: int = 10000) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from validation set.

        Args:
            data_module: Data module with validation data
            max_samples: Maximum number of samples to extract

        Returns:
            Dictionary containing embeddings and labels
        """
        print(f"\nExtracting embeddings (max {max_samples} samples)...")

        val_loader = data_module.val_dataloader()

        image_embeddings = []
        json_embeddings = []
        categories = []

        samples_collected = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Processing batches"):
                if samples_collected >= max_samples:
                    break

                # Move to device
                images = batch.images.to(self.device)
                json_data = {
                    'category': batch.category_ids.to(self.device),
                    'style': batch.style_ids.to(self.device),
                    'silhouette': batch.silhouette_ids.to(self.device),
                    'material': batch.material_ids.to(self.device),
                    'detail': batch.detail_ids.to(self.device),
                    'style_mask': batch.style_mask.to(self.device),
                    'material_mask': batch.material_mask.to(self.device),
                    'detail_mask': batch.detail_mask.to(self.device)
                }

                # Get embeddings
                img_emb = self.model.image_encoder(images)
                json_emb = self.model.json_encoder(json_data)

                # Collect
                image_embeddings.append(img_emb.cpu().numpy())
                json_embeddings.append(json_emb.cpu().numpy())
                categories.append(batch.category_ids.cpu().numpy())

                samples_collected += len(images)

        # Concatenate
        image_embeddings = np.vstack(image_embeddings)[:max_samples]
        json_embeddings = np.vstack(json_embeddings)[:max_samples]
        categories = np.concatenate(categories)[:max_samples]

        print(f"✓ Extracted {len(image_embeddings)} samples")

        return {
            'image_embeddings': image_embeddings,
            'json_embeddings': json_embeddings,
            'categories': categories
        }

    def compute_umap(self,
                     embeddings: np.ndarray,
                     n_neighbors: int = 15,
                     min_dist: float = 0.1,
                     metric: str = 'cosine') -> np.ndarray:
        """
        Compute UMAP projection.

        Args:
            embeddings: Input embeddings [N, D]
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance between points
            metric: Distance metric

        Returns:
            2D UMAP coordinates [N, 2]
        """
        print(f"\nComputing UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric=metric,
            random_state=42,
            verbose=True
        )

        umap_coords = reducer.fit_transform(embeddings)

        print("✓ UMAP completed")
        return umap_coords

    def plot_umap(self,
                  umap_coords: np.ndarray,
                  categories: np.ndarray,
                  category_names: List[str],
                  title: str,
                  save_path: Path):
        """
        Plot UMAP visualization.

        Args:
            umap_coords: UMAP coordinates [N, 2]
            categories: Category labels [N]
            category_names: List of category names
            title: Plot title
            save_path: Path to save figure
        """
        plt.figure(figsize=(16, 12))

        # Use a colormap
        unique_categories = np.unique(categories)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))

        # Plot each category
        for idx, cat_id in enumerate(unique_categories):
            mask = categories == cat_id
            cat_name = category_names[cat_id] if cat_id < len(category_names) else f"Category {cat_id}"

            plt.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=[colors[idx]],
                label=cat_name,
                alpha=0.6,
                s=30,
                edgecolors='none'
            )

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def visualize(self, max_samples: int = 10000):
        """
        Run full UMAP visualization pipeline.

        Args:
            max_samples: Maximum number of samples to visualize
        """
        print("="*80)
        print("UMAP VISUALIZATION")
        print("="*80)

        # Load data
        print("\nLoading dataset...")
        data_module = FashionDataModule(
            dataset_path=self.dataset_path,
            batch_size=256,
            num_workers=4,
            train_split=0.9
        )
        data_module.setup()

        # Get category names
        category_names = list(data_module.dataset_loader.processor.vocabularies.get('category', {}).keys())

        # Extract embeddings
        embeddings_data = self.extract_embeddings(data_module, max_samples)

        # Compute UMAP for image embeddings
        image_umap = self.compute_umap(embeddings_data['image_embeddings'])
        self.plot_umap(
            image_umap,
            embeddings_data['categories'],
            category_names,
            "UMAP: Image Embeddings (Fashion-CLIP)",
            self.output_dir / "umap_image_embeddings.png"
        )

        # Compute UMAP for JSON embeddings
        json_umap = self.compute_umap(embeddings_data['json_embeddings'])
        self.plot_umap(
            json_umap,
            embeddings_data['categories'],
            category_names,
            "UMAP: JSON Embeddings (Metadata)",
            self.output_dir / "umap_json_embeddings.png"
        )

        # Compute UMAP for combined (concatenated) embeddings
        combined_embeddings = np.hstack([
            embeddings_data['image_embeddings'],
            embeddings_data['json_embeddings']
        ])
        combined_umap = self.compute_umap(combined_embeddings)
        self.plot_umap(
            combined_umap,
            embeddings_data['categories'],
            category_names,
            "UMAP: Combined Embeddings (Image + JSON)",
            self.output_dir / "umap_combined_embeddings.png"
        )

        # Save summary
        summary = {
            'model_path': str(self.model_path),
            'num_samples': len(embeddings_data['categories']),
            'num_categories': len(category_names),
            'category_names': category_names,
            'embedding_dim': embeddings_data['image_embeddings'].shape[1]
        }

        summary_path = self.output_dir / "umap_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n✅ UMAP visualization completed!")
        print(f"📁 Results saved to: {self.output_dir}")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="UMAP visualization for fashion embeddings")
    parser.add_argument("--model", type=str, default="checkpoints/emergency_final/best_model.pt",
                       help="Path to trained model checkpoint")
    parser.add_argument("--data", type=str, default=r"C:\Work\hwangseonghun\K-fashion\Training",
                       help="Path to dataset")
    parser.add_argument("--output", type=str, default="results/umap_visualizations",
                       help="Output directory")
    parser.add_argument("--samples", type=int, default=10000,
                       help="Maximum number of samples")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    visualizer = UMAPVisualizer(
        model_path=args.model,
        dataset_path=args.data,
        output_dir=args.output,
        device=args.device
    )

    visualizer.visualize(max_samples=args.samples)


if __name__ == "__main__":
    main()
