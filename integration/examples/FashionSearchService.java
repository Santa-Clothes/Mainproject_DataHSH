package com.yourproject.service;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.util.UriComponentsBuilder;

import java.util.List;
import java.util.Map;

/**
 * Fashion Search Service
 * =======================
 *
 * Spring Boot에서 바로 사용할 수 있는 Fashion Search API 클라이언트
 *
 * application.yml 설정:
 * ```yaml
 * fashion:
 *   search:
 *     base-url: http://localhost:8001
 * ```
 */
@Service
@Slf4j
public class FashionSearchService {

    @Value("${fashion.search.base-url:http://localhost:8001}")
    private String baseUrl;

    private final RestTemplate restTemplate;

    public FashionSearchService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    /**
     * 이미지 파일로 패션 검색
     *
     * @param imageFile 검색할 이미지 파일
     * @param topK 반환할 결과 개수 (기본: 10)
     * @param category 카테고리 필터 (선택사항)
     * @return 검색 결과
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
                .queryParam("top_k", topK != null ? topK : 10);

            if (category != null && !category.isEmpty()) {
                builder.queryParam("category", category);
            }

            // Request entity
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);
            HttpEntity<MultiValueMap<String, Object>> requestEntity =
                new HttpEntity<>(body, headers);

            // API 호출
            log.info("Fashion search request: filename={}, topK={}, category={}",
                imageFile.getOriginalFilename(), topK, category);

            ResponseEntity<FashionSearchResponse> response = restTemplate.exchange(
                builder.toUriString(),
                HttpMethod.POST,
                requestEntity,
                FashionSearchResponse.class
            );

            log.info("Fashion search completed: query_id={}, results={}",
                response.getBody().getQuery().getQueryId(),
                response.getBody().getMetrics().getTotalResults());

            return response.getBody();

        } catch (Exception e) {
            log.error("Fashion search failed: {}", e.getMessage(), e);
            throw new FashionSearchException("검색 중 오류가 발생했습니다", e);
        }
    }

    /**
     * API 헬스 체크
     *
     * @return 헬스 상태
     */
    public HealthResponse checkHealth() {
        try {
            ResponseEntity<HealthResponse> response = restTemplate.getForEntity(
                baseUrl + "/health",
                HealthResponse.class
            );
            return response.getBody();
        } catch (Exception e) {
            log.warn("Fashion search health check failed", e);
            return null;
        }
    }

    /**
     * API가 정상 작동하는지 확인
     *
     * @return true if healthy
     */
    public boolean isHealthy() {
        HealthResponse health = checkHealth();
        return health != null && "healthy".equals(health.getStatus());
    }

    // ===== Response DTOs =====

    @Data
    public static class FashionSearchResponse {
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
        public static class ImageInfo {
            private String filename;
            private Integer size;
            private String dimensions;
            private String format;
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

    @Data
    public static class HealthResponse {
        private String status;
        private Boolean modelLoaded;
        private Integer nineozCount;
        private Integer naverCount;
    }

    /**
     * Fashion Search 예외
     */
    public static class FashionSearchException extends RuntimeException {
        public FashionSearchException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
