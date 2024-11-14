def filter_keystrokes(file_path: str, target_section_id: int):
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            fields = line.strip().split(',')
            if len(fields) >= 5 and fields[4] == str(target_section_id):
                yield fields

def main(target_section_id: int):
    import os
    
    file_path = 'keystrokes.csv'
    output_file = f'sections/keystrokes_section_{target_section_id}.csv'
    
    # sectionsディレクトリが存在しない場合は作成
    os.makedirs('sections', exist_ok=True)
    
    # 出力ファイルが既に存在する場合は削除
    if os.path.exists(output_file):
        os.remove(output_file)
    
    with open(output_file, 'w', newline='') as f:
        for fields in filter_keystrokes(file_path, target_section_id):
            f.write(','.join(fields) + '\n')

if __name__ == '__main__':
    # 32853
    # for i in [120502, 120503, 120507, 120511, 120516, 120519, 120525, 120527, 120535, 120536, 120541, 120544, 120546, 120550, 120554, ]:
    #     main(i)

    # 32872
    for i in [120645, 120650, 120655, 120659, 120662, 120665, 120667, 120671, 120680, 120683, 120686, 120688, 120693, 120696, 120701, ]:
        main(i)