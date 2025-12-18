"""

download dataset

"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=128):
    """
    CIFAR-10 데이터셋을 다운로드하고 DataLoader를 반환합니다.
    과제 요구사항:
    - 32x32 RGB 이미지 
    - Train(50,000) / Test(10,000) Split [cite: 18, 49]
    - Data Augmentation 설정 필요
    """
    
    # 1. 전처리 파이프라인 (Transforms)
    # 기본적인 정규화(Normalization)만 우선 적용합니다.
    # CIFAR-10의 평균과 표준편차를 사용하면 학습 수렴이 빨라집니다.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # Data Augmentation (간단한 예시)
        transforms.RandomHorizontalFlip(),    # Data Augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 2. 데이터셋 다운로드 및 로드
    # root='./data'에 데이터가 저장됩니다.
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    # 3. DataLoader 생성
    # 맥북에서는 num_workers를 2~4 정도로 설정하면 로딩 속도가 빠릅니다.
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader, train_set.classes

if __name__ == "__main__":
    # 장치 설정 확인 (MPS 사용 가능 여부)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Success: Apple Metal (MPS) 가속이 활성화되었습니다.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("⚠️ Warning: CPU 모드입니다. 학습이 느릴 수 있습니다.")

    print(f"Current Device: {device}")

    # 데이터 로더 테스트
    print("\n--- 데이터셋 로딩 테스트 ---")
    train_loader, test_loader, classes = get_cifar10_loaders(batch_size=64)
    
    # 첫 번째 배치를 가져와서 형상(Shape) 확인
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    
    print(f"Batch Image Shape: {images.shape}") 
    # 예상 출력: torch.Size([64, 3, 32, 32]) -> (Batch, Channel, Height, Width)
    print(f"Batch Label Shape: {labels.shape}")
    print(f"Classes example: {classes[:5]}")
    
    print("\n✅ 데이터셋 준비 완료. 다음 단계(ViT 구현)로 넘어갈 준비가 되었습니다.")