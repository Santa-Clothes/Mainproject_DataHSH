package com.yourproject.config;

import org.apache.hc.client5.http.classic.HttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClientBuilder;
import org.apache.hc.client5.http.impl.io.PoolingHttpClientConnectionManager;
import org.apache.hc.core5.util.Timeout;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.HttpComponentsClientHttpRequestFactory;
import org.springframework.web.client.RestTemplate;

/**
 * RestTemplate 설정
 * =================
 *
 * Fashion Search API 호출을 위한 RestTemplate 설정
 * 타임아웃 및 연결 풀 설정 포함
 */
@Configuration
public class RestTemplateConfig {

    @Bean
    public RestTemplate restTemplate() {
        // Connection pool 설정
        PoolingHttpClientConnectionManager connectionManager =
            new PoolingHttpClientConnectionManager();
        connectionManager.setMaxTotal(100);
        connectionManager.setDefaultMaxPerRoute(20);

        // HttpClient 생성
        HttpClient httpClient = HttpClientBuilder.create()
            .setConnectionManager(connectionManager)
            .build();

        // Request factory 설정
        HttpComponentsClientHttpRequestFactory factory =
            new HttpComponentsClientHttpRequestFactory(httpClient);

        // 타임아웃 설정 (패션 검색은 시간이 걸릴 수 있음)
        factory.setConnectTimeout(5000);  // 5초
        factory.setConnectionRequestTimeout(Timeout.ofSeconds(30));  // 30초

        return new RestTemplate(factory);
    }
}
