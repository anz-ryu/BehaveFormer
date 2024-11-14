import onnxruntime
import pandas as pd
import numpy as np
import glob

def compare_sections_with_onnx(section_id1, section_id2):
    """ONNXモデルを使用して2つのセクションを比較
    
    Args:
        section_id1 (int): 比較する1つ目のセクションID
        section_id2 (int): 比較する2つ目のセクションID
    """
    # ONNXランタイムセッションの作成
    ort_session = onnxruntime.InferenceSession("model.onnx")
    input_name = ort_session.get_inputs()[0].name
    
    results = {}
    for section_id in [section_id1, section_id2]:  # 42, 43を変数に置き換え
        # セクションデータを読み込む
        file_path = f'sections/keystrokes_section_{section_id}.csv'
        df = pd.read_csv(file_path, 
                        names=['KEYSTROKE_ID', 'PRESS_TIME', 'RELEASE_TIME', 'LETTER', 
                                'TEST_SECTION_ID', 'KEYCODE', 'IKI'])
        
        # データの前処理
        base_time = df['PRESS_TIME'].min()
        df['PRESS_TIME'] = (df['PRESS_TIME'] - base_time) / 1000.0
        df['RELEASE_TIME'] = (df['RELEASE_TIME'] - base_time) / 1000.0
        df['KEYCODE'] = df['KEYCODE'] / 255.0
        
        # モデル入力用のデータを作成 (10次元の特徴量)
        input_data = []
        for i in range(min(50, len(df))):  # 最大50タイムステップ
            row = df.iloc[i]
            feature = [
                row['PRESS_TIME'],          # 1. プレス時間
                row['RELEASE_TIME'],        # 2. リリース時間
                row['RELEASE_TIME'] - row['PRESS_TIME'],  # 3. ホールド時間
                row['KEYCODE'],             # 4. キーコード
                df.iloc[i+1]['PRESS_TIME'] - row['RELEASE_TIME'] if i < len(df)-1 else 0.0,  # 5. フライト時間
                df.iloc[i+1]['PRESS_TIME'] - row['PRESS_TIME'] if i < len(df)-1 else 0.0,    # 6. プレス間隔
                df.iloc[i+1]['RELEASE_TIME'] - row['RELEASE_TIME'] if i < len(df)-1 else 0.0, # 7. リリース間隔
                df.iloc[i+1]['RELEASE_TIME'] - row['PRESS_TIME'] if i < len(df)-1 else 0.0,   # 8. プレス-リリース間隔
                df.iloc[i+1]['KEYCODE'] if i < len(df)-1 else 0.0,  # 9. 次のキーコード
                float(i) / 50.0  # 10. 正規化された位置情報
            ]
            input_data.append(feature)
        
        # 50タイムステップに満たない場合は0パディング
        while len(input_data) < 50:
            input_data.append([0.0] * 10)
        
        # 入力テンソルの形状を (1, 50, 10) に設定
        input_tensor = np.array([input_data], dtype=np.float32)
        
        # ONNXモデルで推論実行
        ort_inputs = {input_name: input_tensor}
        ort_output = ort_session.run(None, ort_inputs)
        results[section_id] = ort_output[0]
        
        # print(f"\nセクション{section_id}の入力形状: {input_tensor.shape}")
        # print(f"セクション{section_id}の出力形状: {ort_output[0].shape}")
        # print(f"出力サンプル:\n{ort_output[0][:5]}")
    
    # ユークリッド距離を計算
    if len(results) == 2:
        distance = np.sqrt(np.sum(
            (results[section_id1] - results[section_id2]) ** 2  # 42, 43を変数に置き換え
        ))
        # print(f"\n2つのセクション（{section_id1}と{section_id2}）間のユークリッド距離: {distance}")
        return distance

def create_distance_matrix():
    """全セクション間の距離行列を作成してExcelファイルに保存"""
    # sectionsフォルダから全CSVファイルを取得
    section_files = glob.glob('sections/keystrokes_section_*.csv')
    section_ids = [int(f.split('_')[-1].replace('.csv', '')) for f in section_files]
    section_ids.sort()
    
    # 結果を格納する行列を作成
    n = len(section_ids)
    distance_matrix = np.zeros((n, n))
    
    # 全ての組み合わせで比較
    for i, id1 in enumerate(section_ids):
        for j, id2 in enumerate(section_ids):
            if i < j:  # 上三角行列のみ計算
                try:
                    distance = compare_sections_with_onnx(id1, id2)
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance  # 対称行列
                    print(f"セクション{id1}とセクション{id2}の距離: {distance}")
                except Exception as e:
                    print(f"エラーが発生しました（セクション{id1}と{id2}）: {str(e)}")
    
    # DataFrameを作成してExcelに出力
    df = pd.DataFrame(distance_matrix, 
                     index=[f'S{id}' for id in section_ids],
                     columns=[f'S{id}' for id in section_ids])
    df.to_excel('section_distances.xlsx')
    print("\n距離行列をsection_distances.xlsxに保存しました。")

if __name__ == "__main__":
    create_distance_matrix()