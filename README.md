# 2025-PyTorch
2025년 여름 PyTorch 특강

- https://cs231n.github.io/convolutional-networks/

## sample code
```
x = torch.tensor([10, 20, 30])
print(x.shape)         # torch.Size([3])

y = x.unsqueeze(1)
print(y.shape)         # torch.Size([3, 1])
print(y)
```

```
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset

# 1. 가짜 데이터셋 (라벨 0이 90개, 라벨 1이 10개 → 불균형 데이터)
data = torch.arange(100).float().unsqueeze(1)   # 0~99까지 숫자
labels = torch.cat([torch.zeros(90, dtype=torch.long), torch.ones(10, dtype=torch.long)])

dataset = TensorDataset(data, labels)

# 2. 클래스별 샘플 개수 계산
class_counts = torch.bincount(labels)
print("클래스별 개수:", class_counts.tolist())

# 3. 클래스별 가중치 설정 (개수가 적을수록 큰 가중치)
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[labels]  # 각 샘플별 가중치

# 4. WeightedRandomSampler 생성
sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)

# 5. DataLoader에 적용
loader = DataLoader(dataset, batch_size=10, sampler=sampler)

# 6. 한 번 뽑아보기
for batch_data, batch_labels in loader:
    print("라벨 분포:", batch_labels.tolist())
    break  # 첫 배치만 확인
```
