import pandas as pd
from pathlib import Path

def add_headers_to_sections():
    """セクションファイルにヘッダーを追加"""
    headers = ['KEYSTROKE_ID', 'PRESS_TIME', 'RELEASE_TIME', 'LETTER', 
              'TEST_SECTION_ID', 'KEYCODE', 'IKI']
    
    # 各セクションファイルを処理
    for section_id in [42, 43]:
        input_file = f'keystrokes_section_{section_id}.csv'
        output_file = f'keystrokes_section_{section_id}_with_headers.csv'
        
        # データを読み込む
        df = pd.read_csv(input_file, names=headers)
        
        # ヘッダー付きでCSVに保存
        df.to_csv(output_file, index=False)
        print(f"セクション{section_id}のデータを {output_file} に保存しました")

def compare_sections():
    """ヘッダー付きのセクションデータを比較"""
    sections_data = {}
    
    # 各セクションのデータを読み込む
    for section_id in [42, 43]:
        file_path = f'keystrokes_section_{section_id}_with_headers.csv'
        df = pd.read_csv(file_path)
        
        # 時間データを正規化
        base_time = df['PRESS_TIME'].min()
        df['PRESS_TIME'] = df['PRESS_TIME'] - base_time
        df['RELEASE_TIME'] = df['RELEASE_TIME'] - base_time
        
        # KEYCODEを正規化（0-1の範囲に）
        df['KEYCODE'] = df['KEYCODE'] / 255.0
        
        # 時間をミリ秒から秒に変換
        df['PRESS_TIME'] = df['PRESS_TIME'] / 1000.0
        df['RELEASE_TIME'] = df['RELEASE_TIME'] / 1000.0
        
        # IKIを計算（次のキーのPRESS_TIMEと現在のRELEASE_TIMEの差）
        df['IKI'] = df['PRESS_TIME'].shift(-1) - df['RELEASE_TIME']
        
        sections_data[section_id] = df
        
        print(f"\nセクション{section_id}の統計情報:")
        print(df.describe())
        
    return sections_data

if __name__ == "__main__":
    # ヘッダーを追加
    add_headers_to_sections()
    
    # データを比較
    sections_data = compare_sections()
    
    # 基本的な比較情報を表示
    print("\n各セクションのキーストローク数:")
    for section_id, df in sections_data.items():
        print(f"セクション{section_id}: {len(df)}キーストローク")
    
    print("\n各セクションの平均キー押下時間:")
    for section_id, df in sections_data.items():
        mean_press_time = (df['RELEASE_TIME'] - df['PRESS_TIME']).mean()
        print(f"セクション{section_id}: {mean_press_time:.3f}秒") 