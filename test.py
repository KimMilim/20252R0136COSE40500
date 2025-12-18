# vit.py 파일이 있는 폴더에서 실행
from vit import VisionTransformer
import torch

def test_vit():
    # 모델 인스턴스 생성 (CIFAR-10용 설정)
    model = VisionTransformer(
        img_size=32, 
        patch_size=4, 
        embed_dim=192, 
        depth=6, 
        num_heads=3, 
        num_classes=10
    )
    
    # 더미 데이터 생성 (Batch=2, Channels=3, H=32, W=32)
    dummy_input = torch.randn(2, 3, 32, 32)
    
    # Forward Pass
    output = model(dummy_input)
    
    print(f"Model Input Shape: {dummy_input.shape}")
    print(f"Model Output Shape: {output.shape}") # 예상: (2, 10)
    
    # 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

if __name__ == "__main__":
    test_vit()