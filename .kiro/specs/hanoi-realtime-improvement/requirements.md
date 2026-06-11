# Requirements Document - Hệ thống Real-time Giám sát Chất lượng Không khí Hà Nội (Simplified MVP)

## Introduction

Xây dựng hệ thống giám sát chất lượng không khí Hà Nội đơn giản, thực tế với kiến trúc MVP có thể scale up sau này. Tập trung vào giải pháp đơn giản với ít dependencies: API polling + WebSocket/SSE + Redis (optional) thay vì kiến trúc phức tạp với Kafka.

## Glossary

- **Data_Collector**: Service thu thập dữ liệu từ API bên ngoài theo schedule
- **Scheduler**: Cron job hoặc simple scheduler để chạy data collection
- **Local_Storage**: SQLite hoặc PostgreSQL để lưu dữ liệu
- **Data_Processor**: Service xử lý và tính toán dữ liệu
- **WebSocket_Server**: Server phục vụ dữ liệu real-time cho client
- **REST_API**: HTTP API để client query dữ liệu
- **Redis_Cache**: Optional caching layer để tăng performance
- **Circuit_Breaker**: Cơ chế bảo vệ khi API bên ngoài lỗi
- **Hanoi_Station**: Trạm đo chất lượng không khí tại Hà Nội

## Requirements

### Requirement 1: Thu thập Dữ liệu Đơn giản

**User Story:** Là một data engineer, tôi muốn hệ thống thu thập dữ liệu đơn giản và đáng tin cậy, để có dữ liệu liên tục mà không cần infrastructure phức tạp.

#### Acceptance Criteria

1. THE Data_Collector SHALL run as scheduled job every 5 minutes using cron or simple scheduler
2. THE Data_Collector SHALL fetch data from external APIs with timeout of 30 seconds
3. WHEN an API call fails, THE Data_Collector SHALL retry up to 3 times with exponential backoff
4. THE Data_Collector SHALL implement Circuit_Breaker pattern for each external API
5. THE Data_Collector SHALL store raw data in Local_Storage with timestamp and source metadata
6. THE Data_Collector SHALL validate basic data structure before storage
7. THE Data_Collector SHALL log all operations with structured format
8. THE Data_Collector SHALL continue processing other sources if one source fails

### Requirement 2: Tổ chức lại Codebase

**User Story:** Là một developer, tôi muốn codebase được tổ chức rõ ràng và dễ maintain, để có thể phát triển và debug hiệu quả.

#### Acceptance Criteria

1. THE System SHALL separate concerns into distinct modules: collectors, processors, storage, and API layers
2. THE System SHALL implement consistent error handling across all modules
3. THE System SHALL use configuration files for all environment-specific settings
4. THE System SHALL implement comprehensive logging with structured format
5. THE System SHALL include health check endpoints for all services
6. THE System SHALL use dependency injection for external service connections
7. THE System SHALL implement proper resource cleanup in all async operations
8. THE System SHALL follow consistent naming conventions for files, classes, and functions

### Requirement 3: Xử lý Dữ liệu In-Memory

**User Story:** Là một system administrator, tôi muốn xử lý dữ liệu đơn giản trong memory, để có hiệu suất tốt mà không cần message queue phức tạp.

#### Acceptance Criteria

1. THE Data_Processor SHALL process new data within 5 seconds of collection
2. THE Data_Processor SHALL calculate 5-minute and 15-minute trends for PM2.5 values
3. THE Data_Processor SHALL maintain in-memory sliding windows for trend calculations
4. THE Data_Processor SHALL detect anomalous data using simple threshold rules
5. WHEN anomalous data is detected, THE Data_Processor SHALL log alert and notify WebSocket clients
6. THE Data_Processor SHALL update processed data in Local_Storage
7. IF Redis_Cache is available, THE Data_Processor SHALL cache latest processed data with 1-hour TTL
8. THE Data_Processor SHALL expose health check endpoint

### Requirement 4: WebSocket/SSE Real-time cho Client

**User Story:** Là một frontend developer, tôi muốn nhận dữ liệu real-time qua WebSocket hoặc Server-Sent Events, để hiển thị thông tin cập nhật cho người dùng.

#### Acceptance Criteria

1. THE WebSocket_Server SHALL accept client connections on port 8765
2. THE WebSocket_Server SHALL allow clients to subscribe to specific Hanoi_Station data
3. WHEN new processed data is available, THE WebSocket_Server SHALL broadcast to subscribed clients within 2 seconds
4. THE WebSocket_Server SHALL support subscription filtering by data type (weather, air_quality, alerts)
5. THE WebSocket_Server SHALL implement connection heartbeat with 30-second ping interval
6. THE WebSocket_Server SHALL handle client disconnections gracefully
7. THE WebSocket_Server SHALL limit concurrent connections to 50 clients for MVP
8. THE WebSocket_Server SHALL fallback to Server-Sent Events if WebSocket is not supported

### Requirement 5: Monitoring Cơ bản

**User Story:** Là một operations engineer, tôi muốn giám sát cơ bản hệ thống, để phát hiện vấn đề sớm mà không cần infrastructure monitoring phức tạp.

#### Acceptance Criteria

1. THE System SHALL provide health check endpoints for all services
2. THE System SHALL log errors and warnings to structured log files
3. THE System SHALL track basic metrics: data collection success rate, processing time, active connections
4. WHEN data collection fails for more than 15 minutes, THE System SHALL log critical error
5. THE System SHALL detect when PM2.5 values exceed 150 μg/m³ and log health alert
6. THE System SHALL provide simple dashboard endpoint showing system status
7. THE System SHALL rotate log files to prevent disk space issues
8. THE System SHALL expose metrics in simple JSON format for external monitoring

### Requirement 6: Caching Optional và Performance

**User Story:** Là một system user, tôi muốn hệ thống phản hồi nhanh, để có trải nghiệm tốt với kiến trúc đơn giản.

#### Acceptance Criteria

1. THE REST_API SHALL respond to queries within 200ms for cached data
2. THE System SHALL serve data from Local_Storage when real-time data is unavailable
3. IF Redis_Cache is available, THE System SHALL cache latest data for each Hanoi_Station
4. THE System SHALL use connection pooling for database connections
5. THE System SHALL implement simple rate limiting for API endpoints (100 requests/minute per IP)
6. THE System SHALL optimize database queries with proper indexing
7. THE System SHALL limit in-memory data windows to last 24 hours
8. THE System SHALL compress HTTP responses when client supports it

### Requirement 7: Configuration Management

**User Story:** Là một DevOps engineer, tôi muốn quản lý cấu hình dễ dàng cho các môi trường khác nhau, để triển khai và maintain hiệu quả.

#### Acceptance Criteria

1. THE System SHALL load configuration from YAML files with environment variable override
2. THE System SHALL validate configuration on startup and fail fast if invalid
3. THE System SHALL support different configurations for development, staging, and production
4. THE System SHALL allow runtime configuration updates for non-critical settings
5. THE System SHALL encrypt sensitive configuration values (API keys, passwords)
6. THE System SHALL provide configuration schema documentation
7. THE System SHALL log configuration changes with timestamp and user
8. THE System SHALL backup configuration before applying changes

### Requirement 8: Testing Đơn giản

**User Story:** Là một quality assurance engineer, tôi muốn có test coverage cơ bản để đảm bảo chất lượng code với effort hợp lý.

#### Acceptance Criteria

1. THE System SHALL achieve 60% unit test coverage for core business logic
2. THE System SHALL include integration tests for external API interactions
3. THE System SHALL include end-to-end tests for critical data flow paths
4. THE System SHALL use mocks for external dependencies in unit tests
5. THE System SHALL include basic performance tests for API endpoints
6. THE System SHALL run tests automatically on code changes using simple CI
7. THE System SHALL test error handling scenarios
8. THE System SHALL include smoke tests for deployment verification

### Requirement 9: Deployment Đơn giản

**User Story:** Là một platform engineer, tôi muốn hệ thống dễ deploy với minimal dependencies, để triển khai nhanh chóng và maintain dễ dàng.

#### Acceptance Criteria

1. THE System SHALL provide Docker containers for all services
2. THE System SHALL include Docker Compose for local development with minimal services
3. THE System SHALL support single-server deployment for MVP
4. THE System SHALL implement graceful shutdown for all services
5. THE System SHALL include health checks for container orchestration
6. THE System SHALL provide clear deployment instructions for production
7. THE System SHALL include environment-specific configuration files
8. THE System SHALL support easy scaling by adding more instances behind load balancer

### Requirement 10: Documentation Thực tế

**User Story:** Là một new team member, tôi muốn có documentation đủ dùng để setup và hiểu hệ thống nhanh chóng.

#### Acceptance Criteria

1. THE System SHALL include API documentation for all HTTP endpoints
2. THE System SHALL provide setup instructions for local development
3. THE System SHALL include troubleshooting guide for common issues
4. THE System SHALL document configuration options and their effects
5. THE System SHALL include simple architecture diagram showing data flow
6. THE System SHALL provide code examples for common integration patterns
7. THE System SHALL include deployment guide with step-by-step instructions
8. THE System SHALL document scaling options for future growth

### Requirement 11: REST API cho Data Access

**User Story:** Là một client application developer, tôi muốn truy cập dữ liệu qua REST API, để tích hợp với các ứng dụng khác nhau.

#### Acceptance Criteria

1. THE REST_API SHALL provide endpoint to get latest air quality data for Hanoi stations
2. THE REST_API SHALL provide endpoint to get historical data with time range filtering
3. THE REST_API SHALL provide endpoint to get trend data (5-min, 15-min averages)
4. THE REST_API SHALL return data in JSON format with consistent schema
5. THE REST_API SHALL implement pagination for historical data queries
6. THE REST_API SHALL include CORS headers for browser-based clients
7. THE REST_API SHALL provide OpenAPI/Swagger documentation
8. THE REST_API SHALL implement basic authentication for write operations (if any)