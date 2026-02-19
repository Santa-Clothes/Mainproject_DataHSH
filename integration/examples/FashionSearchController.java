package com.yourproject.controller;

import com.yourproject.service.FashionSearchService;
import com.yourproject.service.FashionSearchService.FashionSearchResponse;
import com.yourproject.service.FashionSearchService.HealthResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * Fashion Search Controller
 * ==========================
 *
 * Spring Boot에서 바로 사용할 수 있는 Fashion Search API 컨트롤러
 *
 * 엔드포인트:
 * - POST /api/fashion/search/upload - 이미지 업로드 검색
 * - GET /api/fashion/health - API 헬스 체크
 */
@RestController
@RequestMapping("/api/fashion")
@Slf4j
@CrossOrigin(origins = "*") // 프로덕션에서는 구체적으로 지정
public class FashionSearchController {

    @Autowired
    private FashionSearchService fashionSearchService;

    /**
     * 이미지 업로드 검색
     *
     * @param file 이미지 파일
     * @param topK 결과 개수 (기본: 10)
     * @param category 카테고리 필터 (선택사항)
     * @return 검색 결과
     */
    @PostMapping("/search/upload")
    public ResponseEntity<FashionSearchResponse> uploadSearch(
        @RequestParam("file") MultipartFile file,
        @RequestParam(value = "top_k", defaultValue = "10") Integer topK,
        @RequestParam(value = "category", required = false) String category
    ) {
        log.info("Fashion search request: file={}, topK={}, category={}",
            file.getOriginalFilename(), topK, category);

        // 파일 검증
        if (file.isEmpty()) {
            return ResponseEntity.badRequest().build();
        }

        // 이미지 파일 타입 검증
        String contentType = file.getContentType();
        if (contentType == null || !contentType.startsWith("image/")) {
            log.warn("Invalid file type: {}", contentType);
            return ResponseEntity.badRequest().build();
        }

        try {
            FashionSearchResponse response = fashionSearchService.searchByImage(
                file, topK, category
            );

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("Fashion search error", e);
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * API 헬스 체크
     *
     * @return 헬스 상태
     */
    @GetMapping("/health")
    public ResponseEntity<HealthResponse> healthCheck() {
        HealthResponse health = fashionSearchService.checkHealth();

        if (health == null) {
            return ResponseEntity.status(503).build();
        }

        return ResponseEntity.ok(health);
    }
}
