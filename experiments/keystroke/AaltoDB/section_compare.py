import torch
import numpy as np
import pandas as pd
from pathlib import Path

def load_section_data(section_id: int):
    """特定のセクションIDのデータを読み込んで処理"""
    # CSVファイルを読み込む
    df = pd.read_csv('keystrokes.csv', 
                     names=['KEYSTROKE_ID', 'PRESS_TIME', 'RELEASE_TIME', 'LETTER', 
                           'TEST_SECTION_ID', 'KEYCODE', 'IKI'],
                     encoding='latin-1')
    
    # 特定のセクションのデータのみをフィルタリング
    section_data = df[df['TEST_SECTION_ID'] == section_id].copy()
    
    # 時間データを正規化（最初のPRESS_TIMEを0とする）
    base_time = section_data['PRESS_TIME'].min()
    section_data['PRESS_TIME'] = section_data['PRESS_TIME'] - base_time
    section_data['RELEASE_TIME'] = section_data['RELEASE_TIME'] - base_time
    
    # KEYCODEを正規化（0-1の範囲に）
    section_data['KEYCODE'] = section_data['KEYCODE'] / 255.0
    
    # 時間を秒単位に変換（ミリ秒から秒へ）
    section_data['PRESS_TIME'] = section_data['PRESS_TIME'] / 1000.0
    section_data['RELEASE_TIME'] = section_data['RELEASE_TIME'] / 1000.0
    
    # モデルの入力形式に合わせてデータを整形
    formatted_data = []
    for _, row in section_data.iterrows():
        formatted_data.append([
            row['PRESS_TIME'],
            row['RELEASE_TIME'],
            row['RELEASE_TIME'] - row['PRESS_TIME'],  # キー押下時間
            row['KEYCODE']
        ])
    
    return formatted_data

def compare_sections(section_ids=[42, 43]):
    """2つのセクションのデータを比較"""
    # モデルを読み込む
    model = torch.load('model1.pt', map_location=torch.device('cpu'))
    model.eval()
    
    results = {}
    for section_id in section_ids:
        # セクションデータを読み込む
        data = load_section_data(section_id)
        
        # データをテンソルに変換（バッチサイズ1, シーケンス長50, 特徴量10）
        input_data = torch.tensor([data[:50]], dtype=torch.float32)  # 最初の50タイムステップを使用
        
        # 推論実行
        with torch.no_grad():
            try:
                output = model(input_data)
                results[section_id] = output
                print(f"\nセクション{section_id}の出力形状: {output.shape}")
                print(f"出力サンプル:\n{output[:5]}")  # 最初の5要素を表示
            except Exception as e:
                print(f"セクション{section_id}の処理中にエラーが発生: {str(e)}")
    
    # 距離を計算
    if len(results) == 2:
        distance = torch.sqrt(torch.sum(
            (results[section_ids[0]] - results[section_ids[1]]) ** 2
        )).item()
        print(f"\n2つのセクション間のユークリッド距離: {distance}")

if __name__ == "__main__":
    compare_sections([42, 43]) 