/**
 * Fashion Search API Client
 * =========================
 *
 * Next.js에서 바로 사용할 수 있는 Fashion Search API 클라이언트
 *
 * [사용 방법]
 * 1. 이 파일을 Next.js 프로젝트의 lib/fashionSearch.ts 로 복사
 * 2. .env.local 에 아래 환경변수 추가:
 *      NEXT_PUBLIC_FASHION_SEARCH_API=http://localhost:8001
 * 3. 컴포넌트에서 import 해서 사용
 *
 * [아키텍처]
 * Next.js 컴포넌트 → searchByImage() → FastAPI (port 8001) → Supabase
 */


// ============================================================
// 타입 정의
// ============================================================

/**
 * 검색 결과 상품 1개의 타입
 *
 * 예시:
 * {
 *   rank: 1,
 *   product_id: "NV_12345",
 *   title: "오버핏 후드 티셔츠",
 *   price: 29900,
 *   image_url: "https://shopping-phinf.pstatic.net/...",
 *   category_id: "50000803",
 *   kfashion_category: "스트리트",
 *   score: 0.89          // 유사도 점수 (0~1, 높을수록 유사)
 * }
 */
export interface FashionSearchResult {
  rank: number;              // 검색 결과 순위 (1위부터 시작)
  product_id: string;        // 상품 고유 ID (예: "NV_12345")
  title: string;             // 상품명
  price: number;             // 가격 (원 단위)
  image_url: string;         // 상품 이미지 URL
  category_id: string;       // 네이버 카테고리 ID
  kfashion_category: string; // K-Fashion 스타일 카테고리 (예: "스트리트", "캐주얼")
  score: number;             // 이미지 유사도 점수 (0.0 ~ 1.0, 높을수록 유사)
}

/**
 * API 전체 응답 타입
 *
 * searchByImage() 함수가 반환하는 최상위 객체.
 * 실제 상품 목록은 response.results 에 있음.
 */
export interface FashionSearchResponse {
  /** 요청 정보 (어떤 이미지로 검색했는지) */
  query: {
    query_id: string;   // 요청 고유 ID (UUID)
    timestamp: string;  // 요청 시각 (ISO 8601)
    image_info: {
      filename: string;   // 업로드한 파일명
      size: number;       // 파일 크기 (bytes)
      dimensions: string; // 이미지 해상도 (예: "640x480")
      format: string;     // 이미지 포맷 (예: "JPEG")
    };
  };

  /** 검색 결과 상품 목록 (topK 개수만큼 반환) */
  results: FashionSearchResult[];

  /** 검색 성능 지표 */
  metrics: {
    total_results: number;          // 실제 반환된 결과 수
    search_time_ms: number;         // 검색 소요 시간 (밀리초)
    total_time_ms: number;          // 전체 처리 시간 (이미지 처리 포함, 밀리초)
    category_filter: string | null; // 적용된 카테고리 필터 (없으면 null)
    faiss_enabled: boolean;         // FAISS 벡터 인덱스 사용 여부
  };

  /** 유사도 점수 통계 */
  stats: {
    avg_score: number; // 결과 평균 유사도
    max_score: number; // 최고 유사도 (1위 상품)
    min_score: number; // 최저 유사도 (마지막 상품)
    score_distribution: Record<string, number>; // 구간별 분포 (예: {"0.8-1.0": 3, "0.6-0.8": 5, ...})
  };
}


// ============================================================
// API URL 설정
// ============================================================

/**
 * Fashion Search API 서버 주소
 *
 * 우선순위:
 *   1. .env.local의 NEXT_PUBLIC_FASHION_SEARCH_API 값
 *   2. 없으면 기본값 http://localhost:8001 사용
 *
 * 배포 환경에서는 반드시 .env.local (또는 Vercel 환경변수)에
 * 실제 서버 주소를 설정해야 함.
 * 예) NEXT_PUBLIC_FASHION_SEARCH_API=https://api.yourdomain.com
 */
const FASHION_SEARCH_API =
  process.env.NEXT_PUBLIC_FASHION_SEARCH_API ||
  'http://localhost:8001';


// ============================================================
// API 함수
// ============================================================

/**
 * 이미지 파일로 유사 패션 상품 검색
 *
 * 이미지를 FastAPI 서버에 업로드하면 FashionCLIP 모델이
 * 이미지의 임베딩(벡터)을 생성하고, Naver 상품 DB에서
 * 가장 유사한 상품을 찾아 반환합니다.
 *
 * @param imageFile  - <input type="file">에서 가져온 File 객체
 * @param topK       - 반환받을 최대 결과 수 (기본값: 10, 최대: 100)
 * @param category   - K-Fashion 카테고리 필터 (선택사항)
 *                     예: "스트리트", "캐주얼", "모던", "로맨틱" 등
 *                     없으면 전체 카테고리에서 검색
 * @returns          FashionSearchResponse (상품 목록은 .results 에 있음)
 *
 * @throws 서버 연결 실패, 이미지 처리 실패 시 Error
 *
 * @example 기본 사용법
 * ```tsx
 * const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
 *   const file = e.target.files?.[0];
 *   if (!file) return;
 *
 *   const data = await searchByImage(file, 10);
 *
 *   // 상품 목록 사용
 *   data.results.forEach(item => {
 *     console.log(item.title, item.price, item.score);
 *   });
 * };
 * ```
 *
 * @example 카테고리 필터 적용
 * ```tsx
 * const data = await searchByImage(file, 10, '스트리트');
 * ```
 */
export async function searchByImage(
  imageFile: File,
  topK: number = 10,
  category?: string
): Promise<FashionSearchResponse> {
  // multipart/form-data 로 이미지 파일을 담아 전송
  // Content-Type은 fetch가 자동으로 'multipart/form-data; boundary=...' 로 설정함
  const formData = new FormData();
  formData.append('file', imageFile); // API가 'file' 필드명으로 받음

  // URL 쿼리 파라미터 구성
  const params = new URLSearchParams();
  params.append('top_k', topK.toString()); // 결과 개수
  if (category) {
    params.append('category_filter', category); // 카테고리 필터 (있을 때만 추가)
  }

  // POST /search/upload 호출
  const response = await fetch(
    `${FASHION_SEARCH_API}/search/upload?${params}`,
    {
      method: 'POST',
      body: formData,
      // Content-Type 헤더는 직접 설정하지 않음
      // → FormData 사용 시 fetch가 boundary 포함해서 자동 설정
    }
  );

  // HTTP 에러 처리 (4xx, 5xx)
  if (!response.ok) {
    const error = await response.json();
    // API가 반환한 에러 메시지 우선 사용, 없으면 HTTP 상태 텍스트 사용
    throw new Error(error.detail || `Fashion search failed: ${response.statusText}`);
  }

  return response.json();
}

/**
 * API 서버 상태 확인 (헬스 체크)
 *
 * 서버가 정상 작동 중인지, 모델이 로드됐는지 확인합니다.
 * 앱 초기 렌더링 시 또는 검색 전에 호출해서 서버 상태를 체크할 때 사용.
 *
 * @returns 서버 상태 정보
 *
 * @example
 * ```tsx
 * useEffect(() => {
 *   checkHealth()
 *     .then(h => console.log('서버 상태:', h.status)) // "healthy"
 *     .catch(() => console.error('서버 연결 실패'));
 * }, []);
 * ```
 *
 * 정상 응답 예시:
 * ```json
 * {
 *   "status": "healthy",
 *   "model_loaded": true,
 *   "nineoz_count": 4621,
 *   "naver_count": 7538
 * }
 * ```
 */
/**
 * 검색 결과에서 스타일 분포 계산
 *
 * FashionCLIP 임베딩 자체에 스타일 정보가 반영되어 있으므로,
 * 검색 결과의 kfashion_category와 score를 집계하면
 * "이 이미지가 어떤 스타일인지"를 퍼센트로 표현할 수 있습니다.
 *
 * @param results  - searchByImage()의 results 배열
 * @returns        스타일별 퍼센트 배열 (높은 순 정렬)
 *
 * @example
 * ```tsx
 * const data = await searchByImage(file, 20); // top_k 많을수록 분포 정확
 * const styles = getStyleDistribution(data.results);
 *
 * // styles = [
 * //   { style: "스트리트", percent: 55 },
 * //   { style: "캐주얼",   percent: 30 },
 * //   { style: "모던",     percent: 15 },
 * // ]
 *
 * styles.forEach(({ style, percent }) => {
 *   console.log(`${style}: ${percent}%`);
 * });
 * ```
 */
export function getStyleDistribution(
  results: FashionSearchResult[]
): { style: string; percent: number }[] {
  if (results.length === 0) return [];

  // 카테고리별 score 합산
  const scoreByStyle: Record<string, number> = {};
  for (const item of results) {
    scoreByStyle[item.kfashion_category] =
      (scoreByStyle[item.kfashion_category] || 0) + item.score;
  }

  // 전체 score 합 (퍼센트 계산 기준)
  const totalScore = Object.values(scoreByStyle).reduce((sum, s) => sum + s, 0);

  // 퍼센트 변환 후 높은 순 정렬
  return Object.entries(scoreByStyle)
    .map(([style, score]) => ({
      style,
      percent: Math.round((score / totalScore) * 100),
    }))
    .sort((a, b) => b.percent - a.percent);
}

/**
 * API 서버 상태 확인 (헬스 체크)
 *
 * 서버가 정상 작동 중인지, 모델이 로드됐는지 확인합니다.
 * 앱 초기 렌더링 시 또는 검색 전에 호출해서 서버 상태를 체크할 때 사용.
 *
 * @returns 서버 상태 정보
 *
 * @example
 * ```tsx
 * useEffect(() => {
 *   checkHealth()
 *     .then(h => console.log('서버 상태:', h.status)) // "healthy"
 *     .catch(() => console.error('서버 연결 실패'));
 * }, []);
 * ```
 *
 * 정상 응답 예시:
 * ```json
 * {
 *   "status": "healthy",
 *   "model_loaded": true,
 *   "nineoz_count": 4621,
 *   "naver_count": 7538
 * }
 * ```
 */
export async function checkHealth(): Promise<{
  status: string;        // "healthy" | "limited" (모델 로드 실패 시 limited)
  model_loaded: boolean; // FashionCLIP 모델 로드 여부
  nineoz_count: number;  // 로드된 Nine Oz 내부 상품 수 (현재 4,621개)
  naver_count: number;   // 로드된 Naver 상품 수 (현재 7,538개)
}> {
  const response = await fetch(`${FASHION_SEARCH_API}/health`);

  if (!response.ok) {
    throw new Error('Health check failed');
  }

  return response.json();
}
