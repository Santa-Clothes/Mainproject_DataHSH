"""
라벨링 데이터를 테스트 데이터셋으로 변환

prelabel JSON에서 is_relevant=true인 항목만 추출하여
테스트 데이터셋 JSON으로 변환합니다.
"""

import json
import argparse
from pathlib import Path


def convert_labeling_to_test(labeling_file: str, output_file: str):
    """
    라벨링 데이터를 테스트 데이터로 변환

    Args:
        labeling_file: 라벨링 JSON 파일
        output_file: 출력 테스트 데이터 JSON 파일
    """
    # 라벨링 데이터 로드
    with open(labeling_file, 'r', encoding='utf-8') as f:
        labeling_data = json.load(f)

    # 테스트 데이터 변환
    test_data = {'queries': []}

    for query_data in labeling_data['queries']:
        # relevant 제품 ID 추출
        relevant_ids = []

        for result in query_data.get('search_results', []):
            if result.get('is_relevant') == True:  # 명시적으로 True인 것만
                relevant_ids.append(result['product_id'])

        # relevant가 하나라도 있으면 테스트 쿼리로 추가
        if relevant_ids:
            test_data['queries'].append({
                'query_id': query_data['query_id'],
                'product_id': query_data['product_id'],
                'relevant_product_ids': relevant_ids
            })

    # 저장
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    # 통계 출력
    print("\n" + "="*80)
    print("Conversion Complete")
    print("="*80)
    print(f"Input file:  {labeling_file}")
    print(f"Output file: {output_file}")
    print(f"\nStatistics:")
    print(f"  Total queries: {len(test_data['queries'])}")

    total_relevant = sum(len(q['relevant_product_ids']) for q in test_data['queries'])
    avg_relevant = total_relevant / len(test_data['queries']) if test_data['queries'] else 0

    print(f"  Total relevant products: {total_relevant}")
    print(f"  Avg relevant per query: {avg_relevant:.2f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert labeling data to test dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="tests/test_data/search_labeling_data_prelabel.json",
        help="Input labeling JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tests/test_data/search_test_queries.json",
        help="Output test dataset JSON file"
    )

    args = parser.parse_args()

    convert_labeling_to_test(args.input, args.output)
    print("✅ Ready for evaluation!")
    print(f"Run: python scripts/evaluation/evaluate_search_recall.py")
