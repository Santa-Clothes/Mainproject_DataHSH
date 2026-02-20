package com.yourproject.controller;

import com.yourproject.service.EmbeddingApiService;
import com.yourproject.service.NaverProductService;
import com.yourproject.service.NaverProductService.NaverProduct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

/**
 * Fashion Search Controller
 * ==========================
 *
 * 전체 흐름:
 *   1. 프론트에서 이미지 받기
 *   2. FastAPI에 임베딩 요청 (EmbeddingApiService)
 *   3. 임베딩으로 Supabase 유사 상품 검색 (NaverProductService)
 *   4. 결과 반환
 *
 * 엔드포인트:
 *   POST /api/fashion/search  — 이미지 검색
 *   GET  /api/fashion/health  — 상태 확인
 */
@RestController
@RequestMapping("/api/fashion")
@Slf4j
@CrossOrigin(origins = "*") // 프로덕션에서는 구체적으로 지정
public class FashionSearchController {

    private final EmbeddingApiService embeddingApiService;
    private final NaverProductService naverProductService;

    public FashionSearchController(
        EmbeddingApiService embeddingApiService,
        NaverProductService naverProductService
    ) {
        this.embeddingApiService = embeddingApiService;
        this.naverProductService = naverProductService;
    }

    /**
     * 이미지 업로드 → 유사 상품 검색
     *
     * @param file  이미지 파일 (multipart/form-data)
     * @param topK  반환할 결과 수 (기본: 10)
     * @return      유사 상품 목록
     */
    @PostMapping("/search")
    public ResponseEntity<List<NaverProduct>> search(
        @RequestParam("file") MultipartFile file,
        @RequestParam(value = "top_k", defaultValue = "10") int topK
    ) {
        if (file.isEmpty()) {
            return ResponseEntity.badRequest().build();
        }

        String contentType = file.getContentType();
        if (contentType == null || !contentType.startsWith("image/")) {
            log.warn("이미지 파일만 허용: {}", contentType);
            return ResponseEntity.badRequest().build();
        }

        log.info("[FashionSearch] 요청: file={}, topK={}", file.getOriginalFilename(), topK);

        // 1. 이미지 → 임베딩 (FastAPI)
        float[] embedding = embeddingApiService.getEmbedding(file);

        // 2. 임베딩 → 유사 상품 (Supabase pgvector)
        List<NaverProduct> results = naverProductService.searchSimilar(embedding, topK);

        log.info("[FashionSearch] 완료: {}개 결과", results.size());
        return ResponseEntity.ok(results);
    }

    /**
     * 상태 확인
     */
    @GetMapping("/health")
    public ResponseEntity<String> health() {
        boolean aiHealthy = embeddingApiService.isHealthy();
        if (aiHealthy) {
            return ResponseEntity.ok("{\"status\":\"healthy\",\"ai_server\":true}");
        } else {
            return ResponseEntity.status(503)
                .body("{\"status\":\"ai_server_down\",\"ai_server\":false}");
        }
    }
}
