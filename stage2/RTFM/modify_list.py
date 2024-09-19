# 원본과 변경된 경로
old_path = "/data/hyoukjun/newGTAI3D10"
new_path = "/home/doo304702/Drive/Data/newGTAI3D10"

# 원본 파일과 출력 파일 경로 지정
input_file = "list/newGTA.list"  # 기존 리스트 파일 이름
output_file = "list/newGTA_1.list"  # 업데이트된 리스트를 저장할 파일 이름

# 파일 읽기와 쓰기
with open(input_file, 'r', encoding='utf-8') as infile:
    # 파일의 모든 줄을 읽어온 후 경로를 치환합니다.
    updated_lines = [line.replace(old_path, new_path) for line in infile]

# 변경된 경로를 새로운 파일에 저장합니다.
with open(output_file, 'w', encoding='utf-8') as outfile:
    outfile.writelines(updated_lines)

print(f"경로가 변경된 파일이 '{output_file}'로 저장되었습니다.")
