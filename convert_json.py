from solidity_parser import parser
import json

def convert_sol_to_json(sol_file_path, json_file_path):
    # Đọc và phân tích file Solidity
    with open(sol_file_path, 'r') as file:
        source_code = file.read()
    parsed_data = parser.parse(source_code)

    # Chuyển đổi sang định dạng JSON
    json_data = json.dumps(parsed_data, indent=4)

    # Lưu kết quả ra file JSON
    with open(json_file_path, 'w') as file:
        file.write(json_data)

#Đường dẫn file Solidity và file JSON đầu ra
sol_file_path = 'BID.sol'  # Thay thế với đường dẫn file .sol của bạn
json_file_path = 'a.json'  # Thay thế với đường dẫn bạn muốn lưu file .json

#Chạy hàm chuyển đổi
convert_sol_to_json(sol_file_path, json_file_path)