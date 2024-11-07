import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64*3*3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 64*3*3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = torch.load('model1.pt', map_location=torch.device('cpu'))

print(type(model))  # dictなのか、state_dictなのか確認
print(model.keys() if isinstance(model, dict) else "Not a dictionary")

# モデルの入力形状を確認
def print_model_input_shape():
    if isinstance(model, dict):
        net = Net()
        net.load_state_dict(model)
    else:
        net = model
    
    # モデルの全体構造を表示
    print("モデル構造：")
    print(net)
    
    # モデルの全パラメータを取得
    print("\nモデルパラメータ：")
    for name, param in net.named_parameters():
        print(f"{name}: {param.shape}")

# print_model_input_shape()

def evaluate_with_random_input():
    # 評価モードに設定
    model.eval()
    
    # ランダム入力の作成
    batch_size = 1
    seq_length = 50 
    feature_dim = 10  
    
    # ランダム入力の生成
    random_input = torch.randn(batch_size, seq_length, feature_dim)
    
    print(f"\nランダム入力の形状: {random_input.shape}")
    
    # no_gradを使用して推論
    with torch.no_grad():
        try:
            output = model(random_input)
            print(f"出力の形状: {output.shape}")
            print(f"出力サンプル:\n{output}")
        except Exception as e:
            print(f"実行時エラー: {str(e)}")

# 运行评估
evaluate_with_random_input()


