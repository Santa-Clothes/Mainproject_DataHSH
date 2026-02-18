# Fashion Search - Spring Boot 통합 가이드

## 🎯 아키텍처 개요

```
[Next.js Frontend]
        ↓
[Spring Boot Backend]  ←→  [Supabase PostgreSQL]
        ↓
        ↓ HTTP REST API
        ↓
[Fashion Search Microservice] (Python/FastAPI)
        ↓
        └── FashionCLIP + FAISS
```

**Fashion Search = 독립 Microservice**
- Spring Boot가 HTTP로 호출
- 이미지 검색 전담
- 독립 배포/스케일링

---

## 📡 API 명세

### Base URL
```
개발: http://localhost:8001
프로덕션: https://fashion-search-abc123-uc.a.run.app
```

### 1. 이미지 업로드 검색

**Endpoint:** `POST /search/upload`

**Request:**
```http
POST /search/upload?top_k=10&category=BL HTTP/1.1
Content-Type: multipart/form-data

file: (binary image data)
```

**cURL 예시:**
```bash
curl -X POST "http://localhost:8001/search/upload?top_k=10" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

**Response:**
```json
{
  "query": {
    "query_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2026-02-13T10:30:00Z",
    "image_info": {
      "filename": "test.jpg",
      "size": 45678,
      "dimensions": "800x600",
      "format": "JPEG"
    }
  },
  "results": [
    {
      "rank": 1,
      "product_id": "12345",
      "title": "여성 블라우스",
      "price": 29900,
      "image_url": "https://shopping-phinf.pstatic.net/...",
      "category_id": "BL",
      "kfashion_category": "블라우스",
      "score": 0.856
    }
  ],
  "metrics": {
    "total_results": 10,
    "search_time_ms": 512,
    "total_time_ms": 678,
    "category_filter": null,
    "faiss_enabled": true
  },
  "stats": {
    "avg_score": 0.623,
    "max_score": 0.856,
    "min_score": 0.445,
    "score_distribution": {
      "0.8-1.0": 2,
      "0.6-0.8": 5,
      "0.4-0.6": 3
    }
  }
}
```

---

### 2. Nine Oz 인덱스 검색

**Endpoint:** `GET /search`

**Request:**
```http
GET /search?query_index=0&top_k=10&category=BL HTTP/1.1
```

**Parameters:**
- `query_index` (required): Nine Oz 데이터셋 인덱스 (0~233)
- `top_k` (optional): 결과 개수 (default: 10, max: 100)
- `category` (optional): 카테고리 필터 (BL, SK, OP 등)

**Response:** (동일한 형식)

---

### 3. 헬스 체크

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-13T10:30:00Z",
  "version": "2.0.0",
  "model_loaded": true,
  "faiss_enabled": true,
  "database": {
    "nineoz_count": 234,
    "naver_count": 7538
  }
}
```

---

## 🔧 Spring Boot 통합 코드

### 1. RestTemplate 사용

#### Configuration
```java
@Configuration
public class FashionSearchConfig {

    @Value("${fashion.search.base-url}")
    private String fashionSearchBaseUrl;

    @Bean
    public RestTemplate restTemplate() {
        RestTemplate restTemplate = new RestTemplate();

        // 타임아웃 설정 (검색은 느릴 수 있음)
        HttpComponentsClientHttpRequestFactory factory =
            new HttpComponentsClientHttpRequestFactory();
        factory.setConnectTimeout(5000);  // 5초
        factory.setReadTimeout(30000);    // 30초
        restTemplate.setRequestFactory(factory);

        return restTemplate;
    }
}
```

#### application.yml
```yaml
fashion:
  search:
    base-url: http://localhost:8001
    # 프로덕션: https://fashion-search-abc123-uc.a.run.app
    timeout:
      connect: 5000
      read: 30000
```

#### DTO 정의
```java
// Request
@Data
public class FashionSearchRequest {
    private Integer topK = 10;
    private String category;
}

// Response
@Data
public class FashionSearchResponse {
    private QueryInfo query;
    private List<SearchResult> results;
    private SearchMetrics metrics;
    private SearchStats stats;

    @Data
    public static class QueryInfo {
        private String queryId;
        private String timestamp;
        private ImageInfo imageInfo;
    }

    @Data
    public static class SearchResult {
        private Integer rank;
        private String productId;
        private String title;
        private Integer price;
        private String imageUrl;
        private String categoryId;
        private String kfashionCategory;
        private Double score;
    }

    @Data
    public static class SearchMetrics {
        private Integer totalResults;
        private Integer searchTimeMs;
        private Integer totalTimeMs;
        private String categoryFilter;
        private Boolean faissEnabled;
    }

    @Data
    public static class SearchStats {
        private Double avgScore;
        private Double maxScore;
        private Double minScore;
        private Map<String, Integer> scoreDistribution;
    }
}
```

#### Service 구현
```java
@Service
@Slf4j
public class FashionSearchService {

    @Value("${fashion.search.base-url}")
    private String baseUrl;

    @Autowired
    private RestTemplate restTemplate;

    /**
     * 이미지 파일로 검색
     */
    public FashionSearchResponse searchByImage(
        MultipartFile imageFile,
        Integer topK,
        String category
    ) {
        try {
            // MultiValueMap으로 multipart 데이터 구성
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", imageFile.getResource());

            // Query parameters
            UriComponentsBuilder builder = UriComponentsBuilder
                .fromHttpUrl(baseUrl + "/search/upload")
                .queryParam("top_k", topK);

            if (category != null) {
                builder.queryParam("category", category);
            }

            // Request entity
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);
            HttpEntity<MultiValueMap<String, Object>> requestEntity =
                new HttpEntity<>(body, headers);

            // API 호출
            ResponseEntity<FashionSearchResponse> response = restTemplate.exchange(
                builder.toUriString(),
                HttpMethod.POST,
                requestEntity,
                FashionSearchResponse.class
            );

            log.info("Fashion search completed: query_id={}",
                response.getBody().getQuery().getQueryId());

            return response.getBody();

        } catch (Exception e) {
            log.error("Fashion search failed", e);
            throw new FashionSearchException("검색 중 오류가 발생했습니다", e);
        }
    }

    /**
     * 이미지 URL로 검색 (URL 다운로드 후 검색)
     */
    public FashionSearchResponse searchByImageUrl(
        String imageUrl,
        Integer topK,
        String category
    ) {
        try {
            // 1. 이미지 다운로드
            byte[] imageBytes = downloadImage(imageUrl);

            // 2. MultipartFile로 변환
            MultipartFile multipartFile = new MockMultipartFile(
                "file",
                "image.jpg",
                "image/jpeg",
                imageBytes
            );

            // 3. 검색 수행
            return searchByImage(multipartFile, topK, category);

        } catch (Exception e) {
            log.error("Image URL search failed: url={}", imageUrl, e);
            throw new FashionSearchException("이미지 URL 검색 실패", e);
        }
    }

    /**
     * Nine Oz 인덱스로 검색
     */
    public FashionSearchResponse searchByIndex(
        Integer queryIndex,
        Integer topK,
        String category
    ) {
        try {
            UriComponentsBuilder builder = UriComponentsBuilder
                .fromHttpUrl(baseUrl + "/search")
                .queryParam("query_index", queryIndex)
                .queryParam("top_k", topK);

            if (category != null) {
                builder.queryParam("category", category);
            }

            ResponseEntity<FashionSearchResponse> response = restTemplate.getForEntity(
                builder.toUriString(),
                FashionSearchResponse.class
            );

            return response.getBody();

        } catch (Exception e) {
            log.error("Index search failed: index={}", queryIndex, e);
            throw new FashionSearchException("인덱스 검색 실패", e);
        }
    }

    /**
     * 헬스 체크
     */
    public boolean isHealthy() {
        try {
            ResponseEntity<Map> response = restTemplate.getForEntity(
                baseUrl + "/health",
                Map.class
            );
            return "healthy".equals(response.getBody().get("status"));
        } catch (Exception e) {
            log.warn("Fashion search health check failed", e);
            return false;
        }
    }

    private byte[] downloadImage(String imageUrl) throws IOException {
        // 이미지 다운로드 로직
        RestTemplate imageRestTemplate = new RestTemplate();
        return imageRestTemplate.getForObject(imageUrl, byte[].class);
    }
}

@ResponseStatus(HttpStatus.SERVICE_UNAVAILABLE)
class FashionSearchException extends RuntimeException {
    public FashionSearchException(String message, Throwable cause) {
        super(message, cause);
    }
}
```

#### Controller 예시
```java
@RestController
@RequestMapping("/api/fashion")
@Slf4j
public class FashionController {

    @Autowired
    private FashionSearchService fashionSearchService;

    /**
     * 이미지 업로드 검색
     */
    @PostMapping("/search/upload")
    public ResponseEntity<FashionSearchResponse> uploadSearch(
        @RequestParam("file") MultipartFile file,
        @RequestParam(value = "top_k", defaultValue = "10") Integer topK,
        @RequestParam(value = "category", required = false) String category
    ) {
        log.info("Fashion search request: file={}, topK={}, category={}",
            file.getOriginalFilename(), topK, category);

        FashionSearchResponse response = fashionSearchService.searchByImage(
            file, topK, category
        );

        return ResponseEntity.ok(response);
    }

    /**
     * 이미지 URL로 검색
     */
    @PostMapping("/search/url")
    public ResponseEntity<FashionSearchResponse> urlSearch(
        @RequestParam("image_url") String imageUrl,
        @RequestParam(value = "top_k", defaultValue = "10") Integer topK,
        @RequestParam(value = "category", required = false) String category
    ) {
        log.info("Fashion search by URL: url={}, topK={}, category={}",
            imageUrl, topK, category);

        FashionSearchResponse response = fashionSearchService.searchByImageUrl(
            imageUrl, topK, category
        );

        return ResponseEntity.ok(response);
    }
}
```

---

### 2. WebClient 사용 (Spring WebFlux)

```java
@Configuration
public class WebClientConfig {

    @Value("${fashion.search.base-url}")
    private String baseUrl;

    @Bean
    public WebClient fashionSearchWebClient() {
        return WebClient.builder()
            .baseUrl(baseUrl)
            .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
            .build();
    }
}

@Service
@Slf4j
public class FashionSearchWebClientService {

    @Autowired
    private WebClient fashionSearchWebClient;

    public Mono<FashionSearchResponse> searchByImage(
        MultipartFile imageFile,
        Integer topK,
        String category
    ) {
        return Mono.fromCallable(() -> {
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", imageFile.getResource());
            return body;
        })
        .flatMap(body -> fashionSearchWebClient
            .post()
            .uri(uriBuilder -> {
                uriBuilder.path("/search/upload")
                    .queryParam("top_k", topK);
                if (category != null) {
                    uriBuilder.queryParam("category", category);
                }
                return uriBuilder.build();
            })
            .contentType(MediaType.MULTIPART_FORM_DATA)
            .body(BodyInserters.fromMultipartData(body))
            .retrieve()
            .bodyToMono(FashionSearchResponse.class)
        )
        .doOnSuccess(response ->
            log.info("Fashion search completed: query_id={}",
                response.getQuery().getQueryId())
        )
        .doOnError(error ->
            log.error("Fashion search failed", error)
        );
    }
}
```

---

## 🗄️ PostgreSQL 연동

### 검색 로그 저장

```java
@Entity
@Table(name = "fashion_search_logs")
@Data
public class FashionSearchLog {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String queryId;  // Fashion Search의 query_id

    @Column(nullable = false)
    private LocalDateTime timestamp;

    private Long userId;  // 사용자 ID (로그인 시)

    @Column(length = 500)
    private String imageUrl;

    private String category;
    private Integer topK;

    // 검색 결과 통계
    private Integer totalResults;
    private Integer searchTimeMs;
    private Double avgScore;

    // 결과 (JSON)
    @Column(columnDefinition = "jsonb")
    private String resultsJson;
}

@Repository
public interface FashionSearchLogRepository extends JpaRepository<FashionSearchLog, Long> {
    List<FashionSearchLog> findByUserId(Long userId);
    List<FashionSearchLog> findByTimestampBetween(LocalDateTime start, LocalDateTime end);
}

@Service
public class FashionSearchService {

    @Autowired
    private FashionSearchLogRepository logRepository;

    public FashionSearchResponse searchByImage(
        MultipartFile imageFile,
        Integer topK,
        String category,
        Long userId
    ) {
        // 1. Fashion Search API 호출
        FashionSearchResponse response = ... ;

        // 2. 로그 저장
        FashionSearchLog log = new FashionSearchLog();
        log.setQueryId(response.getQuery().getQueryId());
        log.setTimestamp(LocalDateTime.now());
        log.setUserId(userId);
        log.setCategory(category);
        log.setTopK(topK);
        log.setTotalResults(response.getMetrics().getTotalResults());
        log.setSearchTimeMs(response.getMetrics().getSearchTimeMs());
        log.setAvgScore(response.getStats().getAvgScore());
        log.setResultsJson(objectMapper.writeValueAsString(response.getResults()));

        logRepository.save(log);

        return response;
    }
}
```

---

## 🔄 에러 처리

### Fallback 전략

```java
@Service
public class FashionSearchService {

    private static final int MAX_RETRIES = 3;
    private static final int RETRY_DELAY_MS = 1000;

    public FashionSearchResponse searchByImageWithRetry(
        MultipartFile imageFile,
        Integer topK,
        String category
    ) {
        int attempt = 0;
        Exception lastException = null;

        while (attempt < MAX_RETRIES) {
            try {
                return searchByImage(imageFile, topK, category);
            } catch (Exception e) {
                lastException = e;
                attempt++;

                if (attempt < MAX_RETRIES) {
                    log.warn("Fashion search failed (attempt {}), retrying...", attempt);
                    try {
                        Thread.sleep(RETRY_DELAY_MS * attempt);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    }
                }
            }
        }

        log.error("Fashion search failed after {} attempts", MAX_RETRIES, lastException);

        // Fallback: 빈 결과 반환 또는 캐시된 결과 반환
        return createFallbackResponse();
    }

    private FashionSearchResponse createFallbackResponse() {
        FashionSearchResponse response = new FashionSearchResponse();
        response.setResults(Collections.emptyList());
        response.setMetrics(new FashionSearchResponse.SearchMetrics());
        // ... fallback 데이터 설정
        return response;
    }
}
```

---

## 🐳 로컬 개발 환경

### docker-compose.yml

```yaml
version: '3.8'

services:
  # Spring Boot Backend
  spring-boot:
    build: ./backend
    ports:
      - "8080:8080"
    environment:
      - SPRING_DATASOURCE_URL=jdbc:postgresql://postgres:5432/fashion_db
      - FASHION_SEARCH_BASE_URL=http://fashion-search:8001
    depends_on:
      - postgres
      - fashion-search

  # PostgreSQL
  postgres:
    image: postgres:15
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=fashion_db
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres-data:/var/lib/postgresql/data

  # Fashion Search Microservice
  fashion-search:
    build: ./fashion-search
    ports:
      - "8001:8001"
    environment:
      - DEVICE=cpu
      - USE_FAISS=true
    volumes:
      - ./fashion-search/data:/app/data
      - ./fashion-search/checkpoints:/app/checkpoints

volumes:
  postgres-data:
```

실행:
```bash
docker-compose up -d
```

---

## 📝 체크리스트

### Spring Boot팀이 할 일
- [ ] Fashion Search API 연동 코드 작성
- [ ] PostgreSQL 스키마 생성 (search_logs)
- [ ] 검색 로그 저장 로직 구현
- [ ] 에러 처리 및 Fallback 구현
- [ ] 로컬 docker-compose 테스트

### Fashion Search팀이 할 일 (당신)
- [ ] Cloud Run 배포
- [ ] API 문서 최신화
- [ ] 헬스 체크 엔드포인트 확인
- [ ] 성능 모니터링
- [ ] Spring Boot팀에 Base URL 공유

---

## 🆘 문의사항

**Fashion Search 담당자:** [당신 이름]
**Email:** your-email@example.com
**GitHub:** https://github.com/your-repo

**API 이슈/버그:** GitHub Issues에 등록
**긴급 문의:** Slack #fashion-search 채널
