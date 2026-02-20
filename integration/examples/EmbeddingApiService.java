package com.yourproject.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Map;

/**
 * FastAPI 임베딩 서버 클라이언트
 * ================================
 *
 * 역할: 이미지 파일을 FastAPI(/embed)로 전송 → 768차원 벡터 수신
 * DB 접근 없음. 임베딩 생성만 담당.
 *
 * application.yml:
 *   fashion:
 *     embed:
 *       url: http://localhost:8002
 */
@Service
@Slf4j
public class EmbeddingApiService {

    @Value("${fashion.embed.url:http://localhost:8002}")
    private String embedApiUrl;

    private final RestTemplate restTemplate;

    public EmbeddingApiService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    /**
     * 이미지 파일 → 768차원 임베딩 벡터
     *
     * @param imageFile 업로드된 이미지 파일
     * @return float[] 768차원 벡터 (L2 정규화 완료)
     */
    public float[] getEmbedding(MultipartFile imageFile) {
        try {
            // multipart body 구성
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            ByteArrayResource fileResource = new ByteArrayResource(imageFile.getBytes()) {
                @Override
                public String getFilename() {
                    return imageFile.getOriginalFilename();
                }
            };
            body.add("file", fileResource);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);
            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            log.info("[Embedding] 요청: {}", imageFile.getOriginalFilename());

            ResponseEntity<EmbeddingResponse> response = restTemplate.exchange(
                embedApiUrl + "/embed",
                HttpMethod.POST,
                requestEntity,
                EmbeddingResponse.class
            );

            EmbeddingResponse result = response.getBody();
            if (result == null || result.getEmbedding() == null) {
                throw new RuntimeException("FastAPI 응답이 비어있음");
            }

            log.info("[Embedding] 완료: dimension={}", result.getDimension());

            // List<Double> → float[]
            List<Double> embeddingList = result.getEmbedding();
            float[] embedding = new float[embeddingList.size()];
            for (int i = 0; i < embeddingList.size(); i++) {
                embedding[i] = embeddingList.get(i).floatValue();
            }
            return embedding;

        } catch (Exception e) {
            log.error("[Embedding] 실패: {}", e.getMessage(), e);
            throw new EmbeddingException("임베딩 생성 실패: " + e.getMessage(), e);
        }
    }

    /**
     * FastAPI 서버 상태 확인
     */
    public boolean isHealthy() {
        try {
            ResponseEntity<Map> response = restTemplate.getForEntity(
                embedApiUrl + "/health", Map.class
            );
            return response.getStatusCode().is2xxSuccessful();
        } catch (Exception e) {
            log.warn("[Embedding] 헬스체크 실패: {}", e.getMessage());
            return false;
        }
    }

    // DTO
    @lombok.Data
    public static class EmbeddingResponse {
        private List<Double> embedding;
        private Integer dimension;
    }

    public static class EmbeddingException extends RuntimeException {
        public EmbeddingException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
