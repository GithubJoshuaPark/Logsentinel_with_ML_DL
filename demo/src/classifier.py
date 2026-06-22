import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """PyTorch 기반의 LSTM 다중 분류(Multi-class Classification) 모델 클래스입니다.
    
    1차로 전처리 및 수치화된 로그 데이터 시퀀스를 입력받아,
    로그의 문맥 흐름(Sequence)을 분석하여 최종 에러 등급(ALERT, FATAL, WARNING 등)으로 분류합니다.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2):
        """LSTMClassifier 클래스 생성자.
        
        TF-IDF 등 별도로 임베딩이 불필요한 고차원 수치 벡터를 입력받아 LSTM 레이어로 전달하도록 설계되었습니다.
        
        Args:
            input_dim (int): 입력 피처의 차원 수 (예: TF-IDF 단어 사전 크기인 max_features).
            hidden_dim (int): LSTM 은닉 상태(Hidden State)의 차원 수.
            num_classes (int): 분류할 에러 등급(클래스)의 총 개수.
            num_layers (int): 쌓을 LSTM 레이어의 층수. 기본값은 2입니다.
        """
        super(LSTMClassifier, self).__init__()
        # Embedding 레이어를 제거하고 TF-IDF 수치 벡터(input_dim)를 직접 입력받음
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """모델의 순전파(Forward Propagation) 연산을 정의합니다.
        
        로그 시퀀스 배치 데이터를 LSTM 입력으로 흘려보낸 후,
        마지막 타임스텝의 출력값을 선형 레이어(Fully Connected Layer)에 연결하여 클래스별 예측 스코어를 계산합니다.
        
        Args:
            x (torch.Tensor): (batch_size, sequence_length, input_dim) 모양의 3차원 입력 텐서.
            
        Returns:
            torch.Tensor: (batch_size, num_classes) 모양의 원시 예측 스코어(Logits) 텐서.
        """
        out, (hn, cn) = self.lstm(x)
        # 마지막 타임스텝의 아웃풋 사용
        out = self.fc(out[:, -1, :])
        return out

def train_model(model: nn.Module, train_loader, criterion, optimizer, epochs: int, device: torch.device):
    """지정된 에폭 수 동안 PyTorch LSTM 분류 모델의 훈련을 수행합니다.
    
    에폭마다 데이터 로더에서 로그 시퀀스 텐서와 라벨 텐서를 로드하고,
    오차(Loss) 역전파를 수행하여 가중치를 최적화하고 에폭별 손실 및 정확도를 출력합니다.
    
    Args:
        model (nn.Module): 훈련할 LSTMClassifier 모델 인스턴스.
        train_loader (DataLoader): 훈련 데이터 배치를 공급하는 PyTorch DataLoader.
        criterion: 손실 함수 (예: CrossEntropyLoss).
        optimizer: 옵티마이저 (예: Adam).
        epochs (int): 학습을 반복할 총 에폭 수.
        device (torch.device): 연산을 처리할 하드웨어 장치 객체 (예: cpu, cuda, mps).
        
    Returns:
        dict: 에폭별 손실("loss") 및 정확도("acc") 리스트를 담은 딕셔너리.
    """
    model.to(device)
    history = {"loss": [], "acc": []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = corrects.item() / total
        
        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
    return history

