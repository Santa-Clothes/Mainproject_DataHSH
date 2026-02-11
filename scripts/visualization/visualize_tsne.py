import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from pathlib import Path
from training.trainer import FashionTrainer
from data.fashion_dataset import FashionDataModule, collate_fashion_batch
from utils.config import TrainingConfig

def extract_embeddings(trainer, data_loader, device, max_samples=2000):
    """Extract embeddings from model"""
    trainer.contrastive_learner.eval()
    
    all_image_embeddings = []
    all_json_embeddings = []
    all_categories = []
    
    sample_count = 0
    
    print(f"\n📊 Extracting embeddings (max {max_samples} samples)...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batches"):
            if sample_count >= max_samples:
                break
            
            # Move batch to device
            images = batch.images.to(device)
            json_batch = {
                'category': batch.category_ids.to(device),
                'style': batch.style_ids.to(device),
                'style_mask': batch.style_mask.to(device),
                'silhouette': batch.silhouette_ids.to(device),
                'material': batch.material_ids.to(device),
                'material_mask': batch.material_mask.to(device),
                'detail': batch.detail_ids.to(device),
                'detail_mask': batch.detail_mask.to(device),
            }
            
            # Get embeddings
            embeddings = trainer.contrastive_learner.get_embeddings(images, json_batch)
            
            batch_size = images.shape[0]
            remaining = max_samples - sample_count
            take = min(batch_size, remaining)
            
            all_image_embeddings.append(embeddings['image_embeddings'][:take].cpu())
            all_json_embeddings.append(embeddings['json_embeddings'][:take].cpu())
            all_categories.append(batch.category_ids[:take].cpu())
            
            sample_count += take
    
    # Concatenate
    image_embeddings = torch.cat(all_image_embeddings, dim=0).numpy()
    json_embeddings = torch.cat(all_json_embeddings, dim=0).numpy()
    categories = torch.cat(all_categories, dim=0).numpy()
    
    print(f"✓ Extracted {len(image_embeddings)} samples")
    
    return image_embeddings, json_embeddings, categories

def plot_tsne(embeddings, categories, category_names, title, save_path):
    """Plot t-SNE visualization"""
    print(f"\n🔍 Computing t-SNE for {title}...")
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create figure
    plt.figure(figsize=(16, 12))
    
    # Get unique categories
    unique_categories = np.unique(categories)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))
    
    # Plot each category
    for idx, cat_id in enumerate(unique_categories):
        mask = categories == cat_id
        cat_name = category_names.get(cat_id, f"Cat_{cat_id}")
        
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[idx]],
            label=cat_name,
            alpha=0.6,
            s=50
        )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Config
    config = TrainingConfig(
        batch_size=192,
        learning_rate=3e-4,
        max_epochs=3,
        temperature=0.1,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=512,
        dropout_rate=0.1,
        weight_decay=1e-4
    )
    
    ALL_CATEGORIES = [
        '레트로', '로맨틱', '리조트', '매니시', '모던',
        '밀리터리', '섹시', '소피스트케이티드', '스트리트', '스포티',
        '아방가르드', '오리엔탈', '웨스턴', '젠더리스', '컨트리',
        '클래식', '키치', '톰보이', '펑크', '페미닌',
        '프레피', '히피', '힙합'
    ]
    
    # Load validation data
    print("\n📂 Loading validation data...")
    val_data_module = FashionDataModule(
        dataset_path=r"C:\Work\hwangseonghun\K-fashion\Training",
        target_categories=ALL_CATEGORIES,
        batch_size=config.batch_size,
        num_workers=0,
        train_split=0.9,
        image_size=224,
        augment_prob=0.0
    )
    
    val_data_module.setup()
    
    # Use subset for visualization
    EVAL_SUBSET = 5000
    original_size = len(val_data_module.val_dataset.file_paths)
    val_data_module.val_dataset.file_paths = val_data_module.val_dataset.file_paths[:EVAL_SUBSET]
    print(f"⚡ Using validation subset: {len(val_data_module.val_dataset.file_paths)}/{original_size} samples")
    
    val_loader = torch.utils.data.DataLoader(
        val_data_module.val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fashion_batch
    )
    
    # Load trained model
    print("\n🤖 Loading trained model...")
    trainer = FashionTrainer(
        vocab_sizes=val_data_module.get_vocab_sizes(),
        config=config,
        device=device
    )
    
    # Load epoch 3 checkpoint
    checkpoint_path = "checkpoints/checkpoint_epoch_3.pt"
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'json_encoder_state_dict' in checkpoint:
            trainer.contrastive_learner.json_encoder.load_state_dict(checkpoint['json_encoder_state_dict'])
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
            train_loss = checkpoint.get('train_loss', None)
            if train_loss is not None:
                print(f"  Train Loss: {train_loss:.4f}")
        else:
            trainer.contrastive_learner.load_state_dict(checkpoint)
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Extract embeddings
    image_embeddings, json_embeddings, categories = extract_embeddings(
        trainer, val_loader, device, max_samples=2000
    )
    
    # Create category name mapping
    category_names = {i: name for i, name in enumerate(ALL_CATEGORIES)}
    
    # Create output directory
    output_dir = Path("results/tsne_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING T-SNE VISUALIZATIONS")
    print("="*80)
    
    # Plot image embeddings
    plot_tsne(
        image_embeddings,
        categories,
        category_names,
        "t-SNE: Image Embeddings (Epoch 3)",
        output_dir / "tsne_image_embeddings_epoch3.png"
    )
    
    # Plot JSON embeddings
    plot_tsne(
        json_embeddings,
        categories,
        category_names,
        "t-SNE: JSON Embeddings (Epoch 3)",
        output_dir / "tsne_json_embeddings_epoch3.png"
    )
    
    # Plot combined embeddings
    combined_embeddings = np.concatenate([image_embeddings, json_embeddings], axis=0)
    combined_categories = np.concatenate([categories, categories], axis=0)
    
    plot_tsne(
        combined_embeddings,
        combined_categories,
        category_names,
        "t-SNE: Combined Image + JSON Embeddings (Epoch 3)",
        output_dir / "tsne_combined_embeddings_epoch3.png"
    )
    
    print("\n" + "="*80)
    print("✅ T-SNE VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - tsne_image_embeddings_epoch3.png")
    print(f"  - tsne_json_embeddings_epoch3.png")
    print(f"  - tsne_combined_embeddings_epoch3.png")

if __name__ == "__main__":
    main()
