
input_file = "list/RTFM/newGTA.list"  
output_file = "list/RTFM/newGTA2.list" 
# 원본 파일을 읽고 문자열 수정 후 저장
with open(input_file, 'r') as file:
    lines = file.readlines()  # 모든 라인을 읽음

# 수정된 라인을 저장할 리스트 초기화
modified_lines = []

# 각 줄에 대해 문자열 수정 수행
for line in lines:
    # 원본 경로와 파일 이름을 새로운 경로로 바꾸고, _i3d를 삭제
    modified_line = line.replace('D:/Data',
                                 '/data/hyoukjun')
    # modified_line = modified_line.replace('_i3d', '')
    modified_lines.append(modified_line)  # 수정된 줄을 리스트에 추가

# 수정된 내용을 새 파일에 작성
with open(output_file, 'w') as file:
    file.writelines(modified_lines)