def filter_keystrokes(file_path: str, target_section_id: int):
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            fields = line.strip().split(',')
            if len(fields) >= 5 and fields[4] == str(target_section_id):
                yield fields

def main():
    file_path = 'keystrokes.csv'
    target_section_id = 30
    
    output_file = f'keystrokes_section_{target_section_id}.csv'
    
    with open(output_file, 'w', newline='') as f:
        for fields in filter_keystrokes(file_path, target_section_id):
            f.write(','.join(fields) + '\n')

if __name__ == '__main__':
    main()
