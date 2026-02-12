"""
FashionCLIP Embedding Generator

Loads trained FashionCLIP model and generates embeddings from images.
"""

from pathlib import Path
from typing import Union, List, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class FashionCLIPEmbeddingGenerator:
    """
    FashionCLIP 임베딩 생성기

    학습된 FashionCLIP 모델을 로드하고 이미지 → 임베딩 변환을 수행합니다.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        use_safetensors: bool = True,
    ):
        """
        Args:
            checkpoint_path: 체크포인트 경로 (None이면 pretrained FashionCLIP)
            device: 'cuda', 'cpu', 또는 None (자동 감지)
            use_safetensors: SafeTensors 사용 여부
        """
        self.checkpoint_path = checkpoint_path
        self.use_safetensors = use_safetensors

        # Device 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[Embedding Generator] Using device: {self.device}")

        # 모델 로드
        self.model = None
        self.image_transforms = None
        self._load_model()
        self._setup_transforms()

    def _load_model(self):
        """FashionCLIP 모델 로드"""
        from transformers import CLIPVisionModel

        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            # 학습된 체크포인트 로드
            print(f"[Embedding Generator] Loading checkpoint: {self.checkpoint_path}")

            # FashionCLIP 기본 모델 먼저 로드
            try:
                self.model = CLIPVisionModel.from_pretrained(
                    "patrickjohncyh/fashion-clip",
                    use_safetensors=self.use_safetensors
                )
                print(f"[OK] FashionCLIP base model loaded")
            except Exception as e:
                print(f"[WARNING] FashionCLIP load failed ({e}), using standard CLIP")
                self.model = CLIPVisionModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    use_safetensors=self.use_safetensors
                )
                print(f"[OK] Standard CLIP base model loaded")

            # 체크포인트에서 state_dict만 로드 (pickle 의존성 회피)
            try:
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                    weights_only=False  # config 객체 때문에 필요
                )

                if 'clip_encoder_state_dict' in checkpoint:
                    # Fine-tuned weights 로드
                    missing_keys, unexpected_keys = self.model.load_state_dict(
                        checkpoint['clip_encoder_state_dict'],
                        strict=False
                    )
                    print(f"[OK] Loaded fine-tuned weights from checkpoint")
                    if missing_keys:
                        print(f"  [INFO] Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        print(f"  [INFO] Unexpected keys: {len(unexpected_keys)}")
                else:
                    print(f"[WARNING] No 'clip_encoder_state_dict' in checkpoint")
                    print(f"[OK] Using pretrained weights only")

            except Exception as e:
                print(f"[WARNING] Could not load checkpoint: {e}")
                print(f"[OK] Using pretrained weights only")
        else:
            # Pretrained FashionCLIP만 사용
            print(f"[Embedding Generator] Loading pretrained FashionCLIP")
            try:
                self.model = CLIPVisionModel.from_pretrained(
                    "patrickjohncyh/fashion-clip",
                    use_safetensors=self.use_safetensors
                )
                print(f"[OK] FashionCLIP loaded")
            except Exception as e:
                print(f"[WARNING] FashionCLIP load failed ({e}), using standard CLIP")
                self.model = CLIPVisionModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    use_safetensors=self.use_safetensors
                )
                print(f"[OK] Standard CLIP loaded")

        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # 파라미터 동결
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"[OK] Model ready for inference")

    def _setup_transforms(self):
        """이미지 전처리 transforms 설정"""
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        """
        이미지 로드 (파일 경로, URL, 또는 PIL Image)

        Args:
            image_source: 이미지 경로, URL, 또는 PIL Image

        Returns:
            PIL Image (RGB)
        """
        if isinstance(image_source, Image.Image):
            return image_source.convert('RGB')

        # URL인 경우
        if isinstance(image_source, str) and image_source.startswith(('http://', 'https://')):
            try:
                from io import BytesIO
                import requests
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert('RGB')
            except Exception as e:
                raise ValueError(f"Failed to load image from URL: {e}")

        # 로컬 파일인 경우
        if isinstance(image_source, str):
            try:
                return Image.open(image_source).convert('RGB')
            except Exception as e:
                raise ValueError(f"Failed to load image from file: {e}")

        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    def generate_embedding(
        self,
        image_source: Union[str, Image.Image],
        normalize: bool = True
    ) -> np.ndarray:
        """
        단일 이미지에서 임베딩 생성

        Args:
            image_source: 이미지 경로, URL, 또는 PIL Image
            normalize: L2 정규화 여부

        Returns:
            임베딩 벡터 (numpy array, shape: [embedding_dim])
        """
        # 이미지 로드 및 전처리
        image = self._load_image(image_source)
        image_tensor = self.image_transforms(image).unsqueeze(0)  # [1, 3, 224, 224]
        image_tensor = image_tensor.to(self.device)

        # 임베딩 생성
        with torch.no_grad():
            outputs = self.model(pixel_values=image_tensor)
            embedding = outputs.pooler_output  # [1, embedding_dim]

            # L2 정규화
            if normalize:
                embedding = nn.functional.normalize(embedding, p=2, dim=-1)

            # numpy로 변환
            embedding = embedding.cpu().numpy()[0]  # [embedding_dim]

        return embedding

    def generate_embeddings_batch(
        self,
        image_sources: List[Union[str, Image.Image]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        여러 이미지에서 임베딩 생성 (배치 처리)

        Args:
            image_sources: 이미지 경로/URL/PIL Image 리스트
            batch_size: 배치 크기
            normalize: L2 정규화 여부
            show_progress: 진행률 표시 여부

        Returns:
            임베딩 행렬 (numpy array, shape: [num_images, embedding_dim])
        """
        embeddings = []
        num_images = len(image_sources)

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, num_images, batch_size), desc="Generating embeddings")
            except ImportError:
                iterator = range(0, num_images, batch_size)
                print(f"Generating embeddings for {num_images} images...")
        else:
            iterator = range(0, num_images, batch_size)

        for i in iterator:
            batch_sources = image_sources[i:i + batch_size]

            # 배치 이미지 로드 및 전처리
            batch_tensors = []
            for img_source in batch_sources:
                try:
                    image = self._load_image(img_source)
                    image_tensor = self.image_transforms(image)
                    batch_tensors.append(image_tensor)
                except Exception as e:
                    warnings.warn(f"Failed to process image {img_source}: {e}")
                    # 실패한 경우 더미 텐서 (검은 이미지)
                    batch_tensors.append(torch.zeros(3, 224, 224))

            # 배치로 묶기
            if len(batch_tensors) == 0:
                continue

            batch_tensor = torch.stack(batch_tensors).to(self.device)  # [batch, 3, 224, 224]

            # 임베딩 생성
            with torch.no_grad():
                outputs = self.model(pixel_values=batch_tensor)
                batch_embeddings = outputs.pooler_output  # [batch, embedding_dim]

                # L2 정규화
                if normalize:
                    batch_embeddings = nn.functional.normalize(batch_embeddings, p=2, dim=-1)

                # numpy로 변환
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.append(batch_embeddings)

        # 모든 배치 합치기
        if len(embeddings) == 0:
            raise ValueError("No valid images to process")

        all_embeddings = np.vstack(embeddings)  # [num_images, embedding_dim]

        return all_embeddings

    @property
    def embedding_dim(self) -> int:
        """임베딩 차원 반환"""
        # CLIP vision model의 hidden size
        return self.model.config.hidden_size


# Convenience function
def create_embedding_generator(
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None
) -> FashionCLIPEmbeddingGenerator:
    """
    임베딩 생성기 생성 (convenience function)

    Args:
        checkpoint_path: 체크포인트 경로
        device: 디바이스

    Returns:
        FashionCLIPEmbeddingGenerator
    """
    return FashionCLIPEmbeddingGenerator(
        checkpoint_path=checkpoint_path,
        device=device
    )


if __name__ == "__main__":
    # 테스트
    print("\n" + "="*80)
    print("FashionCLIP Embedding Generator Test")
    print("="*80)

    # 임베딩 생성기 생성
    generator = create_embedding_generator(
        checkpoint_path="checkpoints/multi_domain/best_model.pt"
    )

    print(f"\nEmbedding dimension: {generator.embedding_dim}")

    # 더미 이미지로 테스트
    print("\nGenerating test embedding from dummy image...")
    dummy_image = Image.new('RGB', (224, 224), color='red')
    embedding = generator.generate_embedding(dummy_image)

    print(f"[OK] Embedding shape: {embedding.shape}")
    print(f"[OK] Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"[OK] Embedding sample: {embedding[:5]}")

    print("\n[OK] Embedding generator is ready!")
