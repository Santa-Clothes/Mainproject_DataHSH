package com.yourproject.service;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.*;

/**
 * Supabase pgvector 유사도 검색 서비스
 * ======================================
 *
 * 역할: 임베딩 벡터로 naver_products 테이블에서 유사 상품 검색
 * Spring이 직접 Supabase REST API 호출 (DB 접근은 여기서만)
 *
 * 사전 조건: README의 Supabase SQL 설정 완료 필요
 *
 * application.yml:
 *   supabase:
 *     url: https://fjoylosbfvojioljibku.supabase.co
 *     key: YOUR_SERVICE_ROLE_KEY
 */
@Service
@Slf4j
public class NaverProductService {

    @Value("${supabase.url}")
    private String supabaseUrl;

    @Value("${supabase.key}")
    private String supabaseKey;

    private final RestTemplate restTemplate;

    public NaverProductService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    /**
     * 임베딩 벡터로 유사 상품 검색 (Supabase pgvector RPC)
     *
     * @param embedding 768차원 float 벡터 (FastAPI에서 받은 값)
     * @param topK      반환할 상품 수
     * @return 유사도 순 정렬된 상품 목록
     */
    public List<NaverProduct> searchSimilar(float[] embedding, int topK) {
        try {
            // float[] → List<Float> (JSON 직렬화용)
            List<Float> embeddingList = new ArrayList<>(embedding.length);
            for (float v : embedding) {
                embeddingList.add(v);
            }

            // Supabase RPC 요청 body
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("query_embedding", embeddingList);
            requestBody.put("match_count", topK);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("apikey", supabaseKey);
            headers.set("Authorization", "Bearer " + supabaseKey);

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);

            log.info("[NaverSearch] pgvector 검색: topK={}", topK);

            ResponseEntity<NaverProduct[]> response = restTemplate.exchange(
                supabaseUrl + "/rest/v1/rpc/match_naver_products",
                HttpMethod.POST,
                requestEntity,
                NaverProduct[].class
            );

            NaverProduct[] results = response.getBody();
            if (results == null) {
                return Collections.emptyList();
            }

            log.info("[NaverSearch] 결과: {}개", results.length);
            return Arrays.asList(results);

        } catch (Exception e) {
            log.error("[NaverSearch] 실패: {}", e.getMessage(), e);
            throw new RuntimeException("유사 상품 검색 실패: " + e.getMessage(), e);
        }
    }

    // DTO
    @Data
    public static class NaverProduct {
        private String productId;
        private String title;
        private Integer price;
        private String imageUrl;
        private String categoryId;
        private String kfashionCategory;
        private Double similarity;
    }
}
