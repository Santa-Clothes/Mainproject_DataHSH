import json

# JSON 파일 구조 확인
with open('D:/K-Fashion/Training/라벨링데이터/레트로/100307.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("JSON 구조:")
print(f"Type: {type(data)}")
print(f"Keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
print("\n첫 2000자:")
print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])
