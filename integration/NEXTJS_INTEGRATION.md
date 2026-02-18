# Fashion Search - Next.js 통합 가이드

## 🎯 아키텍처

```
[Next.js Frontend]
        ↓
        ├─ Direct API Call (개발 중)
        ├─ Or via Spring Boot (프로덕션)
        ↓
[Fashion Search API]
```

**옵션 1:** Next.js → Fashion Search (직접 호출)
**옵션 2:** Next.js → Spring Boot → Fashion Search (프록시)

---

## 📡 API 호출 방법

### 1. Fetch API 사용

```typescript
// lib/fashionSearch.ts

export interface FashionSearchResult {
  rank: number;
  product_id: string;
  title: string;
  price: number;
  image_url: string;
  category_id: string;
  kfashion_category: string;
  score: number;
}

export interface FashionSearchResponse {
  query: {
    query_id: string;
    timestamp: string;
    image_info: {
      filename: string;
      size: number;
      dimensions: string;
      format: string;
    };
  };
  results: FashionSearchResult[];
  metrics: {
    total_results: number;
    search_time_ms: number;
    total_time_ms: number;
    category_filter: string | null;
    faiss_enabled: boolean;
  };
  stats: {
    avg_score: number;
    max_score: number;
    min_score: number;
    score_distribution: Record<string, number>;
  };
}

const FASHION_SEARCH_API = process.env.NEXT_PUBLIC_FASHION_SEARCH_API ||
  'http://localhost:8001';

export async function searchByImage(
  imageFile: File,
  topK: number = 10,
  category?: string
): Promise<FashionSearchResponse> {
  const formData = new FormData();
  formData.append('file', imageFile);

  const params = new URLSearchParams();
  params.append('top_k', topK.toString());
  if (category) {
    params.append('category', category);
  }

  const response = await fetch(
    `${FASHION_SEARCH_API}/search/upload?${params}`,
    {
      method: 'POST',
      body: formData,
    }
  );

  if (!response.ok) {
    throw new Error(`Fashion search failed: ${response.statusText}`);
  }

  return response.json();
}
```

---

### 2. React Component 예시

```tsx
// components/FashionSearch.tsx
'use client';

import { useState } from 'react';
import { searchByImage, FashionSearchResponse } from '@/lib/fashionSearch';
import Image from 'next/image';

export default function FashionSearch() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<FashionSearchResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setError(null);
  };

  const handleSearch = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    try {
      const results = await searchByImage(selectedFile, 10);
      setSearchResults(results);
    } catch (err) {
      setError(err instanceof Error ? err.message : '검색 중 오류가 발생했습니다');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-8">Fashion Image Search</h1>

      {/* Upload Section */}
      <div className="mb-8">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className="cursor-pointer text-blue-600 hover:text-blue-700"
          >
            <div className="text-6xl mb-4">📤</div>
            <div className="text-lg">이미지를 선택하거나 드래그하세요</div>
          </label>
        </div>

        {previewUrl && (
          <div className="mt-4">
            <img
              src={previewUrl}
              alt="Preview"
              className="max-w-xs mx-auto rounded-lg shadow-lg"
            />
            <button
              onClick={handleSearch}
              disabled={isLoading}
              className="mt-4 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
            >
              {isLoading ? '검색 중...' : '🔍 검색 시작'}
            </button>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {/* Results */}
      {searchResults && (
        <div>
          <h2 className="text-2xl font-bold mb-4">
            검색 결과 ({searchResults.metrics.total_results}개)
          </h2>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-gray-100 p-4 rounded-lg">
              <div className="text-gray-600">검색 시간</div>
              <div className="text-2xl font-bold">
                {searchResults.metrics.search_time_ms}ms
              </div>
            </div>
            <div className="bg-gray-100 p-4 rounded-lg">
              <div className="text-gray-600">평균 유사도</div>
              <div className="text-2xl font-bold">
                {searchResults.stats.avg_score.toFixed(3)}
              </div>
            </div>
            <div className="bg-gray-100 p-4 rounded-lg">
              <div className="text-gray-600">최고 유사도</div>
              <div className="text-2xl font-bold">
                {searchResults.stats.max_score.toFixed(3)}
              </div>
            </div>
          </div>

          {/* Results Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {searchResults.results.map((item) => (
              <div
                key={item.rank}
                className="bg-white rounded-lg shadow-lg overflow-hidden hover:shadow-xl transition-shadow"
              >
                <div className="relative h-64">
                  <Image
                    src={item.image_url}
                    alt={item.title}
                    fill
                    className="object-cover"
                  />
                  <div className="absolute top-2 left-2 bg-blue-600 text-white px-3 py-1 rounded-full text-sm font-bold">
                    #{item.rank}
                  </div>
                  <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white px-2 py-1 rounded text-sm">
                    {item.score.toFixed(3)}
                  </div>
                </div>
                <div className="p-4">
                  <div className="text-sm text-gray-500 mb-1">
                    {item.kfashion_category}
                  </div>
                  <h3 className="font-semibold mb-2 line-clamp-2">
                    {item.title}
                  </h3>
                  <div className="text-lg font-bold text-blue-600">
                    ₩{item.price.toLocaleString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

---

### 3. Server-Side API Route (Option)

```typescript
// app/api/fashion/search/route.ts

import { NextRequest, NextResponse } from 'next/server';

const FASHION_SEARCH_API = process.env.FASHION_SEARCH_API ||
  'http://localhost:8001';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;

    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }

    // Forward to Fashion Search API
    const searchFormData = new FormData();
    searchFormData.append('file', file);

    const topK = request.nextUrl.searchParams.get('top_k') || '10';
    const category = request.nextUrl.searchParams.get('category');

    const params = new URLSearchParams();
    params.append('top_k', topK);
    if (category) {
      params.append('category', category);
    }

    const response = await fetch(
      `${FASHION_SEARCH_API}/search/upload?${params}`,
      {
        method: 'POST',
        body: searchFormData,
      }
    );

    if (!response.ok) {
      throw new Error('Fashion search failed');
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('Fashion search error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
```

Client에서 사용:
```typescript
const response = await fetch('/api/fashion/search?top_k=10', {
  method: 'POST',
  body: formData,
});
```

---

## 🔧 환경 변수 설정

### .env.local
```bash
# 개발 환경 (로컬)
NEXT_PUBLIC_FASHION_SEARCH_API=http://localhost:8001

# 또는 Server-Side에서만 사용
FASHION_SEARCH_API=http://localhost:8001
```

### .env.production
```bash
# 프로덕션 환경
NEXT_PUBLIC_FASHION_SEARCH_API=https://fashion-search-abc123-uc.a.run.app

# 또는 Spring Boot를 통한 프록시
NEXT_PUBLIC_API_BASE_URL=https://api.yourproject.com
```

---

## 🎨 UI/UX 권장사항

### 1. 로딩 상태
```tsx
{isLoading && (
  <div className="flex items-center justify-center">
    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
    <span className="ml-3">유사한 상품을 찾고 있습니다...</span>
  </div>
)}
```

### 2. 에러 처리
```tsx
{error && (
  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
    <div className="flex">
      <div className="flex-shrink-0">
        <svg className="h-5 w-5 text-red-400" /* ... */ />
      </div>
      <div className="ml-3">
        <h3 className="text-sm font-medium text-red-800">검색 실패</h3>
        <div className="mt-2 text-sm text-red-700">{error}</div>
      </div>
    </div>
  </div>
)}
```

### 3. 빈 결과
```tsx
{searchResults && searchResults.results.length === 0 && (
  <div className="text-center py-12">
    <div className="text-6xl mb-4">🔍</div>
    <h3 className="text-xl font-semibold mb-2">검색 결과가 없습니다</h3>
    <p className="text-gray-600">다른 이미지로 시도해보세요</p>
  </div>
)}
```

---

## 📱 반응형 디자인

```tsx
// Tailwind CSS 클래스 예시
<div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
  {/* Results */}
</div>
```

---

## 🔐 CORS 설정

Fashion Search API에서 CORS 허용이 필요합니다:

```python
# api/search_api.py (이미 설정됨)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 구체적으로 지정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

프로덕션에서는:
```python
allow_origins=[
    "https://yourproject.com",
    "https://www.yourproject.com"
]
```

---

## 🧪 테스트

```typescript
// __tests__/fashionSearch.test.ts

import { searchByImage } from '@/lib/fashionSearch';

describe('Fashion Search API', () => {
  it('should search by image', async () => {
    const mockFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });

    const result = await searchByImage(mockFile, 10);

    expect(result.results).toBeDefined();
    expect(result.results.length).toBeGreaterThan(0);
    expect(result.metrics.total_results).toBeGreaterThan(0);
  });

  it('should handle errors gracefully', async () => {
    const invalidFile = new File([''], 'invalid.txt', { type: 'text/plain' });

    await expect(searchByImage(invalidFile, 10)).rejects.toThrow();
  });
});
```

---

## 📝 체크리스트

### Next.js팀이 할 일
- [ ] Fashion Search API 연동 코드 작성
- [ ] UI 컴포넌트 구현
- [ ] 에러 처리 및 로딩 상태 처리
- [ ] 반응형 디자인 적용
- [ ] 로컬 테스트

### 협업 사항
- [ ] API Base URL 공유
- [ ] 에러 응답 형식 협의
- [ ] Rate limiting 확인
- [ ] 이미지 파일 크기 제한 확인

---

## 🆘 문의사항

**Fashion Search 담당자:** [당신 이름]
**Slack:** #fashion-search
**API Docs:** http://localhost:8001/docs
