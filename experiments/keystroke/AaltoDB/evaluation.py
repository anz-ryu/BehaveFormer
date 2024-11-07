import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

import onnx
import onnxruntime

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
    
    print(f"\nランダム入力の形: {random_input.shape}")
    
    # no_gradを使用して推論
    with torch.no_grad():
        try:
            output = model(random_input)
            print(f"出力の形: {output.shape}")
            print(f"出力サンプル:\n{output}")
        except Exception as e:
            print(f"実行時エラー: {str(e)}")

# evaluate_with_random_input()

# ランダム入力の形: torch.Size([1, 50, 10])
# 出力の形: torch.Size([1, 64])

def convert_to_onnx():
    # モデルを評価モードに設定
    model.eval()
    
    # KeystrokeTransformerモデル用の入力形状
    batch_size = 1
    seq_length = 50 
    feature_dim = 10
    dummy_input = torch.randn(batch_size, seq_length, feature_dim)
    
    # ONNXモデルへの変換
    try:
        torch.onnx.export(model,                    # モデル
                         dummy_input,               # モデル入力例
                         "model.onnx",             # 保存先ファイル名
                         export_params=True,        # 学習済みパラメータを保存
                         opset_version=13,         # ONNXのバージョン
                         do_constant_folding=True,  # 定数畳み込みの最適化
                         input_names=['input'],     # 入力名
                         output_names=['output'],   # 出力名
                         dynamic_axes={'input': {0: 'batch_size', 1: 'sequence'},    # 可変サイズの次元
                                     'output': {0: 'batch_size'}})
        print("モデルをONNX形式に変換し、'model.onnx'として保存しました")
    except Exception as e:
        print(f"変換中にエラーが発生しました: {str(e)}")

def verify_onnx_model():
    
    try:
        # ONNXモデルの基本的な検証
        onnx_model = onnx.load("model.onnx")
        onnx.checker.check_model(onnx_model)
        print("ONNXモデルの構造は有効です")
        
        # ONNXランタイムセッションの作成
        ort_session = onnxruntime.InferenceSession("model.onnx")
        
        # テスト入力の作成（KeystrokeTransformer用）
        dummy_input = torch.randn(1, 50, 10)  # batch_size, seq_length, feature_dim
        input_name = ort_session.get_inputs()[0].name
        
        # ONNXモデルで推論実行
        ort_inputs = {input_name: dummy_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)
        
        # PyTorchモデルで同じ入力での推論実行
        model.eval()
        with torch.no_grad():
            torch_output = model(dummy_input)
        
        # 出力の比較
        print("\nPyTorch出力とONNX出力の比較:")
        print(f"PyTorch出力形状: {torch_output.shape}")
        print(f"ONNX出力形状: {ort_output[0].shape}")
        
        # 出力値の差異を確認
        import numpy as np
        np.testing.assert_allclose(torch_output.numpy(), 
                                 ort_output[0], 
                                 rtol=1e-03, 
                                 atol=1e-05)
        print("PyTorchとONNXの出力が一致しています！")
        
    except Exception as e:
        print(f"検証中にエラーが発生しました: {str(e)}")

# まずONNXに変換
print("ONNXへの変換を開始します...")
convert_to_onnx()

# 変換されたモデルを検証
print("\nONNXモデルの検証を開始します...")
verify_onnx_model()

