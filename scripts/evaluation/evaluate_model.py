import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from training.trainer import FashionTrainer
from src.data.fashion_dataset import FashionDataModule, collate_fashion_batch
from utils.config import TrainingConfig

def calculate_metrics(trainer, val_loader, device):
    """Calculate Top-1, Top-5, Top-10, MRR metrics"""
    trainer.contrastive_learner.eval()
    
    all_image_embeddings = []
    all_json_embeddings = []
    
    print("\n📊 Extracting embeddings from validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Processing batches"):
            # Move batch to device (ProcessedBatch structure)
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
            
            all_image_embeddings.append(embeddings['image_embeddings'].cpu())
            all_json_embeddings.append(embeddings['json_embeddings'].cpu())
    
    # Concatenate all embeddings
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_json_embeddings = torch.cat(all_json_embeddings, dim=0)
    
    print(f"✓ Total samples: {all_image_embeddings.shape[0]}")
    
    # Calculate similarity matrix
    print("\n🔍 Calculating similarity matrix...")
    similarity_matrix = torch.matmul(all_image_embeddings, all_json_embeddings.T)
    
    # Calculate metrics
    print("\n📈 Computing metrics...")
    batch_size = similarity_matrix.shape[0]
    
    # Ground truth: diagonal elements (i-th image matches i-th JSON)
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    mrr_sum = 0.0
    
    for i in tqdm(range(batch_size), desc="Calculating metrics"):
        # Get similarity scores for i-th image
        scores = similarity_matrix[i]
        
        # Get top-k indices
        topk_indices = torch.argsort(scores, descending=True)
        
        # Find rank of ground truth (i-th JSON)
        rank = (topk_indices == i).nonzero(as_tuple=True)[0].item() + 1
        
        # Update metrics
        if rank == 1:
            top1_correct += 1
        if rank <= 5:
            top5_correct += 1
        if rank <= 10:
            top10_correct += 1
        
        mrr_sum += 1.0 / rank
    
    # Calculate final metrics
    top1_acc = (top1_correct / batch_size) * 100
    top5_acc = (top5_correct / batch_size) * 100
    top10_acc = (top10_correct / batch_size) * 100
    mrr = mrr_sum / batch_size
    
    return {
        'top1': top1_acc,
        'top5': top5_acc,
        'top10': top10_acc,
        'mrr': mrr,
        'total_samples': batch_size
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Config (same as training)
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
    
    # Load validation data (10K subset for fast evaluation)
    print("\n📂 Loading validation data...")
    val_data_module = FashionDataModule(
        dataset_path=r"C:\Work\hwangseonghun\K-fashion\Training",
        target_categories=ALL_CATEGORIES,
        batch_size=config.batch_size,
        num_workers=0,
        train_split=0.9,  # Use 10% for validation
        image_size=224,
        augment_prob=0.0  # No augmentation for validation
    )
    
    val_data_module.setup()
    
    # Use subset for fast evaluation
    EVAL_SUBSET = 10000
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
    
    print(f"✓ Validation batches: {len(val_loader)}")
    
    # Load trained model
    print("\n🤖 Loading trained model...")
    trainer = FashionTrainer(
        vocab_sizes=val_data_module.get_vocab_sizes(),
        config=config,
        device=device
    )
    
    checkpoint_path = "checkpoints/best_model.pt"
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Load only the json_encoder state (CLIP is frozen)
        if 'json_encoder_state_dict' in checkpoint:
            trainer.contrastive_learner.json_encoder.load_state_dict(checkpoint['json_encoder_state_dict'])
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        else:
            trainer.contrastive_learner.load_state_dict(checkpoint)
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Calculate metrics
    print("\n" + "="*80)
    print("EVALUATING MODEL")
    print("="*80)
    
    metrics = calculate_metrics(trainer, val_loader, device)
    
    # Print results
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    print(f"Total samples evaluated: {metrics['total_samples']}")
    print(f"\nTop-1 Accuracy:  {metrics['top1']:.2f}%")
    print(f"Top-5 Accuracy:  {metrics['top5']:.2f}%")
    print(f"Top-10 Accuracy: {metrics['top10']:.2f}%")
    print(f"MRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")
    print("="*80)
    
    # Save results
    import json
    results_path = "results/fast_stage2_evaluation.json"
    Path("results").mkdir(exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {results_path}")

if __name__ == "__main__":
    main()
