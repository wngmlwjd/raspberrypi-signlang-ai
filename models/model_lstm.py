import torch
import torch.nn as nn

class SignLanguageModel(nn.Module):
    def __init__(self, input_size=42, hidden_size=128, num_layers=2, num_classes=30, dropout=0.2):
        """
        input_size : 손 관절 좌표 수 (x,y,z) * 21 = 63, 2D만 쓰면 42
        hidden_size: LSTM hidden dim
        num_layers : LSTM 층 수
        num_classes: 분류할 수어 클래스 수
        """
        super(SignLanguageModel, self).__init__()

        # LSTM 기반 시퀀스 모델
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully connected for classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, num_classes)
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_size]
        """
        # LSTM 출력
        out, (hn, cn) = self.lstm(x)  # out: [batch, seq_len, hidden]
        # 마지막 time-step만 사용
        out = out[:, -1, :]            # [batch, hidden]
        out = self.fc(out)             # [batch, num_classes]
        return out

# Example
if __name__ == "__main__":
    model = SignLanguageModel(input_size=42, num_classes=30)
    dummy_input = torch.randn(8, 50, 42)  # batch=8, seq_len=50, feature=42
    out = model(dummy_input)
    print(out.shape)  # torch.Size([8, 30])
