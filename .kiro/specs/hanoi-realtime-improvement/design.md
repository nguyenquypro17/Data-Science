# Design Document - Hệ thống Real-time Giám sát Chất lượng Không khí Hà Nội (Simplified MVP)

## Overview

Thiết kế hệ thống giám sát chất lượng không khí Hà Nội với kiến trúc đơn giản, thực tế cho giai đoạn MVP. Hệ thống sử dụng polling-based data collection, in-memory processing, và WebSocket/SSE để cung cấp dữ liệu real-time mà không cần infrastructure phức tạp như Kafka.

### Key Design Principles

- **Simplicity First**: Ưu tiên giải pháp đơn giản, dễ hiểu và maintain
- **Minimal Dependencies**: Sử dụng ít external services nhất có thể
- **Scalable Architecture**: Thiết kế có thể scale up khi cần thiết
- **Practical MVP**: Tập trung vào tính năng cốt lõi, bỏ qua complexity không cần thiết

## Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "External APIs"
        API1[Weather API]
        API2[Air Quality API]
        API3[Government API]
    end
    
    subgraph "Core System"
        DC[Data Collector]
        DB[(Local Database<br/>SQLite/PostgreSQL)]
        DP[Data Processor]
        WS[WebSocket Server]
        REST[REST API Server]
        CACHE[(Redis Cache<br/>Optional)]
    end
    
    subgraph "Clients"
        WEB[Web Dashboard]
        MOBILE[Mobile App]
        API_CLIENT[API Clients]
    end
    
    subgraph "Monitoring"
        LOGS[Log Files]
        HEALTH[Health Checks]
    end
    
    API1 --> DC
    API2 --> DC
    API3 --> DC
    
    DC --> DB
    DB --> DP
    DP --> DB
    DP --> CACHE
    DP --> WS
    
    DB --> REST
    CACHE --> REST
    
    WS --> WEB
    WS --> MOBILE
    REST --> API_CLIENT
    
    DC --> LOGS
    DP --> LOGS
    WS --> LOGS
    REST --> LOGS
    
    DC --> HEALTH
    DP --> HEALTH
    WS --> HEALTH
    REST --> HEALTH
```

### Component Interaction Flow

```mermaid
sequenceDiagram
    participant Scheduler
    participant DataCollector
    participant Database
    participant DataProcessor
    participant WebSocketServer
    participant Client
    participant RESTApi
    
    loop Every 5 minutes
        Scheduler->>DataCollector: Trigger collection
        DataCollector->>External APIs: Fetch data
        DataCollector->>Database: Store raw data
        DataCollector->>DataProcessor: Notify new data
        DataProcessor->>Database: Read raw data
        DataProcessor->>DataProcessor: Calculate trends
        DataProcessor->>Database: Store processed data
        DataProcessor->>WebSocketServer: Broadcast update
        WebSocketServer->>Client: Send real-time data
    end
    
    Client->>RESTApi: Request historical data
    RESTApi->>Database: Query data
    RESTApi->>Client: Return JSON response
```

## Components and Interfaces

### 1. Data Collector Service

**Responsibility**: Thu thập dữ liệu từ external APIs theo schedule

**Key Features**:
- Cron-based scheduling (5-minute intervals)
- Circuit breaker pattern cho mỗi API
- Retry logic với exponential backoff
- Data validation cơ bản

**Interface**:
```python
class DataCollector:
    def collect_from_source(self, source_config: SourceConfig) -> CollectionResult
    def validate_data(self, raw_data: dict) -> bool
    def store_raw_data(self, data: dict, source: str) -> bool
    def get_health_status(self) -> HealthStatus
```

**Configuration**:
```yaml
data_collector:
  schedule_interval: "*/5 * * * *"  # Every 5 minutes
  sources:
    - name: "weather_api"
      url: "https://api.weather.com/hanoi"
      timeout: 30
      retry_count: 3
    - name: "air_quality_api"
      url: "https://api.airquality.com/hanoi"
      timeout: 30
      retry_count: 3
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 300
```

### 2. Data Processor Service

**Responsibility**: Xử lý dữ liệu thô thành dữ liệu có ý nghĩa

**Key Features**:
- In-memory sliding window calculations
- Trend analysis (5-min, 15-min averages)
- Anomaly detection với simple thresholds
- Real-time notification cho WebSocket clients

**Interface**:
```python
class DataProcessor:
    def process_new_data(self, raw_data: RawData) -> ProcessedData
    def calculate_trends(self, station_id: str) -> TrendData
    def detect_anomalies(self, data: ProcessedData) -> List[Alert]
    def get_sliding_window(self, station_id: str, duration: timedelta) -> WindowData
```

**Processing Logic**:
- PM2.5 trend calculation using simple moving average
- Alert generation khi PM2.5 > 150 μg/m³
- Data quality scoring dựa trên completeness

### 3. WebSocket Server

**Responsibility**: Cung cấp real-time data cho clients

**Key Features**:
- WebSocket connections với heartbeat
- Subscription-based filtering
- Fallback to Server-Sent Events
- Connection limit (50 concurrent cho MVP)

**Interface**:
```python
class WebSocketServer:
    def handle_connection(self, websocket: WebSocket) -> None
    def handle_subscription(self, client_id: str, filters: dict) -> None
    def broadcast_update(self, data: ProcessedData) -> None
    def send_heartbeat(self) -> None
```

**Message Format**:
```json
{
  "type": "air_quality_update",
  "timestamp": "2024-01-15T10:30:00Z",
  "station_id": "hanoi_central",
  "data": {
    "pm25": 45.2,
    "pm10": 78.1,
    "aqi": 112,
    "trend_5min": "increasing",
    "trend_15min": "stable"
  }
}
```

### 4. REST API Server

**Responsibility**: Cung cấp HTTP API cho data access

**Key Features**:
- RESTful endpoints cho current và historical data
- Pagination cho large datasets
- CORS support
- Rate limiting (100 req/min per IP)

**Endpoints**:
```
GET /api/v1/stations                    # List all stations
GET /api/v1/stations/{id}/current       # Current data for station
GET /api/v1/stations/{id}/history       # Historical data with pagination
GET /api/v1/stations/{id}/trends        # Trend data (5min, 15min)
GET /api/v1/health                      # System health check
GET /api/v1/metrics                     # Basic metrics
```

**Response Format**:
```json
{
  "status": "success",
  "data": {
    "station_id": "hanoi_central",
    "timestamp": "2024-01-15T10:30:00Z",
    "measurements": {
      "pm25": 45.2,
      "pm10": 78.1,
      "temperature": 28.5,
      "humidity": 65.0
    },
    "aqi": 112,
    "quality_level": "moderate"
  },
  "metadata": {
    "last_updated": "2024-01-15T10:30:00Z",
    "data_quality": 0.95
  }
}
```

### 5. Database Layer

**Responsibility**: Persistent storage cho raw và processed data

**Schema Design**:
```sql
-- Raw data table
CREATE TABLE raw_measurements (
    id SERIAL PRIMARY KEY,
    station_id VARCHAR(50) NOT NULL,
    source VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Processed data table
CREATE TABLE processed_measurements (
    id SERIAL PRIMARY KEY,
    station_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    pm25 FLOAT,
    pm10 FLOAT,
    temperature FLOAT,
    humidity FLOAT,
    aqi INTEGER,
    quality_level VARCHAR(20),
    trend_5min VARCHAR(20),
    trend_15min VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Alerts table
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    station_id VARCHAR(50) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    message TEXT,
    severity VARCHAR(20),
    timestamp TIMESTAMP NOT NULL,
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_processed_station_timestamp ON processed_measurements(station_id, timestamp DESC);
CREATE INDEX idx_raw_station_timestamp ON raw_measurements(station_id, timestamp DESC);
CREATE INDEX idx_alerts_station_timestamp ON alerts(station_id, timestamp DESC);
```

### 6. Optional Redis Cache

**Responsibility**: Caching layer để tăng performance

**Usage**:
- Cache latest measurements (TTL: 1 hour)
- Cache trend calculations (TTL: 15 minutes)
- Cache API responses (TTL: 5 minutes)

**Key Patterns**:
```
hanoi:station:{station_id}:current     -> Latest measurement
hanoi:station:{station_id}:trends      -> Trend data
hanoi:api:cache:{endpoint}:{params}    -> API response cache
```

## Data Models

### Core Data Structures

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class RawMeasurement:
    station_id: str
    source: str
    timestamp: datetime
    data: Dict[str, Any]
    
@dataclass
class ProcessedMeasurement:
    station_id: str
    timestamp: datetime
    pm25: Optional[float]
    pm10: Optional[float]
    temperature: Optional[float]
    humidity: Optional[float]
    aqi: Optional[int]
    quality_level: str
    trend_5min: str
    trend_15min: str
    
@dataclass
class Alert:
    station_id: str
    alert_type: str
    message: str
    severity: str
    timestamp: datetime
    resolved: bool = False
    
@dataclass
class TrendData:
    station_id: str
    timestamp: datetime
    avg_5min: float
    avg_15min: float
    direction_5min: str  # "increasing", "decreasing", "stable"
    direction_15min: str
    
@dataclass
class HealthStatus:
    service_name: str
    status: str  # "healthy", "degraded", "unhealthy"
    last_check: datetime
    details: Dict[str, Any]
```

### Data Validation Rules

```python
class DataValidator:
    @staticmethod
    def validate_pm25(value: float) -> bool:
        return 0 <= value <= 1000
    
    @staticmethod
    def validate_temperature(value: float) -> bool:
        return -50 <= value <= 60
    
    @staticmethod
    def validate_humidity(value: float) -> bool:
        return 0 <= value <= 100
    
    @staticmethod
    def calculate_aqi(pm25: float, pm10: float) -> int:
        # Simplified AQI calculation
        if pm25 <= 12:
            return min(50, int(pm25 * 4.17))
        elif pm25 <= 35.4:
            return min(100, int(50 + (pm25 - 12) * 2.14))
        elif pm25 <= 55.4:
            return min(150, int(100 + (pm25 - 35.4) * 2.5))
        else:
            return min(300, int(150 + (pm25 - 55.4) * 1.04))
```

## Error Handling

### Error Categories và Response Strategies

1. **External API Failures**
   - Circuit breaker activation
   - Fallback to cached data
   - Retry với exponential backoff
   - Graceful degradation

2. **Database Connection Issues**
   - Connection pooling với retry
   - Fallback to in-memory storage (temporary)
   - Health check failures

3. **Processing Errors**
   - Skip invalid data points
   - Log errors với context
   - Continue processing other data

4. **WebSocket Connection Issues**
   - Automatic reconnection for clients
   - Heartbeat monitoring
   - Graceful connection cleanup

### Error Response Format

```json
{
  "status": "error",
  "error": {
    "code": "DATA_UNAVAILABLE",
    "message": "Air quality data temporarily unavailable",
    "details": "External API timeout after 3 retries",
    "timestamp": "2024-01-15T10:30:00Z",
    "retry_after": 300
  }
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

After analyzing the acceptance criteria, the following properties have been identified for property-based testing. Some redundant properties have been consolidated to avoid duplication:

### Property 1: Retry Logic Consistency

*For any* API call failure scenario, the Data_Collector should retry exactly 3 times with exponential backoff intervals, regardless of the failure type or timing.

**Validates: Requirements 1.3**

### Property 2: Circuit Breaker State Transitions

*For any* sequence of API success/failure responses, the Circuit_Breaker should transition between CLOSED, OPEN, and HALF_OPEN states according to the configured thresholds and timeouts.

**Validates: Requirements 1.4**

### Property 3: Data Storage with Metadata

*For any* valid raw data input, the Data_Collector should store it in Local_Storage with proper timestamp and source metadata, ensuring data integrity and traceability.

**Validates: Requirements 1.5**

### Property 4: Data Validation Consistency

*For any* input data structure, the validation logic should correctly classify it as valid or invalid based on the defined schema rules, with consistent behavior across all data types.

**Validates: Requirements 1.6**

### Property 5: Error Isolation Between Sources

*For any* combination of source failures, the Data_Collector should continue processing data from functioning sources without being affected by failed sources.

**Validates: Requirements 1.8**

### Property 6: Structured Logging Format

*For any* system operation or event, the generated logs should follow the defined structured format with consistent fields and proper JSON serialization.

**Validates: Requirements 1.7, 2.4**

### Property 7: Trend Calculation Accuracy

*For any* sequence of PM2.5 measurements, the Data_Processor should calculate 5-minute and 15-minute trends correctly using the defined moving average algorithms.

**Validates: Requirements 3.2**

### Property 8: Sliding Window Maintenance

*For any* continuous data stream, the in-memory sliding windows should maintain the correct size and contain the most recent data points for trend calculations.

**Validates: Requirements 3.3**

### Property 9: Anomaly Detection Threshold Rules

*For any* PM2.5 measurement value, the anomaly detection should correctly identify values exceeding the defined thresholds (e.g., PM2.5 > 150 μg/m³) and generate appropriate alerts.

**Validates: Requirements 3.4, 3.5**

### Property 10: WebSocket Subscription Filtering

*For any* client subscription request with specific filters (station_id, data_type), the WebSocket_Server should only broadcast matching data to that client, ensuring proper data isolation.

**Validates: Requirements 4.2, 4.4**

### Property 11: Connection Limit Enforcement

*For any* number of concurrent connection attempts, the WebSocket_Server should enforce the 50-client limit for MVP, accepting connections up to the limit and rejecting additional ones gracefully.

**Validates: Requirements 4.7**

### Property 12: REST API Response Schema Consistency

*For any* valid API request, the response should conform to the defined JSON schema with consistent field names, data types, and structure across all endpoints.

**Validates: Requirements 11.4**

### Property 13: Historical Data Filtering

*For any* time range query parameters, the REST API should return only the historical data that falls within the specified time bounds, with proper boundary handling.

**Validates: Requirements 11.2**

### Property 14: Pagination Correctness

*For any* large dataset query, the pagination logic should correctly divide the results into pages of the specified size, with proper offset calculations and consistent ordering.

**Validates: Requirements 11.5**

## Testing Strategy

### Testing Approach

Hệ thống sử dụng dual testing approach với focus vào practical testing:

1. **Unit Tests (60% coverage target)**
   - Core business logic testing
   - Data validation functions
   - Trend calculation algorithms
   - Error handling scenarios

2. **Property-Based Tests**
   - Universal properties across all inputs (minimum 100 iterations per test)
   - Each property test references its design document property
   - Tag format: **Feature: hanoi-realtime-improvement, Property {number}: {property_text}**
   - Uses appropriate PBT library for the target language (e.g., Hypothesis for Python, fast-check for JavaScript)

3. **Integration Tests**
   - External API interactions với mocks
   - Database operations
   - WebSocket connection handling
   - End-to-end data flow

4. **Performance Tests**
   - API response time benchmarks
   - WebSocket connection limits
   - Database query performance
   - Memory usage under load

### Test Configuration

- **Unit tests**: Run với mocks cho external dependencies
- **Integration tests**: Use test database và mock external APIs
- **Performance tests**: Simulate realistic load patterns
- **Smoke tests**: Verify deployment health

### Example Test Cases

```python
# Unit Test Example
class TestDataProcessor:
    def test_calculate_5min_trend_increasing(self):
        # Test trend calculation with increasing values
        data_points = [10, 15, 20, 25, 30]
        result = processor.calculate_trend(data_points, window=5)
        assert result.direction == "increasing"
    
    def test_anomaly_detection_high_pm25(self):
        # Test alert generation for high PM2.5
        measurement = ProcessedMeasurement(pm25=200)
        alerts = processor.detect_anomalies(measurement)
        assert len(alerts) == 1
        assert alerts[0].alert_type == "high_pm25"

# Property-Based Test Example
from hypothesis import given, strategies as st

class TestDataProcessorProperties:
    @given(st.lists(st.floats(min_value=0, max_value=500), min_size=5, max_size=100))
    def test_trend_calculation_accuracy(self, pm25_values):
        """
        Feature: hanoi-realtime-improvement, Property 7: Trend Calculation Accuracy
        For any sequence of PM2.5 measurements, trends should be calculated correctly
        """
        result = processor.calculate_trends(pm25_values)
        
        # Verify 5-minute average is within expected range
        assert 0 <= result.avg_5min <= max(pm25_values)
        # Verify trend direction is consistent with data
        if len(pm25_values) >= 2:
            last_values = pm25_values[-5:]
            if all(last_values[i] <= last_values[i+1] for i in range(len(last_values)-1)):
                assert result.direction_5min in ["increasing", "stable"]
    
    @given(st.floats(min_value=0, max_value=1000))
    def test_anomaly_detection_threshold_rules(self, pm25_value):
        """
        Feature: hanoi-realtime-improvement, Property 9: Anomaly Detection Threshold Rules
        For any PM2.5 value, anomaly detection should correctly apply threshold rules
        """
        measurement = ProcessedMeasurement(pm25=pm25_value)
        alerts = processor.detect_anomalies(measurement)
        
        if pm25_value > 150:
            assert len(alerts) > 0
            assert any(alert.alert_type == "high_pm25" for alert in alerts)
        else:
            assert not any(alert.alert_type == "high_pm25" for alert in alerts)
```

## Deployment Architecture

### Single-Server MVP Deployment

```mermaid
graph TB
    subgraph "Production Server"
        subgraph "Docker Containers"
            DC_CONT[Data Collector Container]
            DP_CONT[Data Processor Container]
            WS_CONT[WebSocket Server Container]
            API_CONT[REST API Container]
        end
        
        subgraph "Data Layer"
            PG[(PostgreSQL)]
            REDIS[(Redis - Optional)]
        end
        
        subgraph "Infrastructure"
            NGINX[Nginx Load Balancer]
            LOGS[Log Files]
        end
    end
    
    NGINX --> WS_CONT
    NGINX --> API_CONT
    
    DC_CONT --> PG
    DP_CONT --> PG
    DP_CONT --> REDIS
    API_CONT --> PG
    API_CONT --> REDIS
    
    DC_CONT --> LOGS
    DP_CONT --> LOGS
    WS_CONT --> LOGS
    API_CONT --> LOGS
```

### Docker Compose Configuration

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: hanoi_air_quality
      POSTGRES_USER: app_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  data-collector:
    build: ./services/data-collector
    environment:
      - DATABASE_URL=postgresql://app_user:${DB_PASSWORD}@postgres:5432/hanoi_air_quality
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
  
  data-processor:
    build: ./services/data-processor
    environment:
      - DATABASE_URL=postgresql://app_user:${DB_PASSWORD}@postgres:5432/hanoi_air_quality
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
  
  websocket-server:
    build: ./services/websocket-server
    environment:
      - DATABASE_URL=postgresql://app_user:${DB_PASSWORD}@postgres:5432/hanoi_air_quality
      - REDIS_URL=redis://redis:6379
    ports:
      - "8765:8765"
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
  
  rest-api:
    build: ./services/rest-api
    environment:
      - DATABASE_URL=postgresql://app_user:${DB_PASSWORD}@postgres:5432/hanoi_air_quality
      - REDIS_URL=redis://redis:6379
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - websocket-server
      - rest-api

volumes:
  postgres_data:
  redis_data:
```

### Scaling Strategy

**Horizontal Scaling Options**:

1. **Load Balancer Setup**
   - Multiple REST API instances behind Nginx
   - Multiple WebSocket server instances với sticky sessions
   - Database connection pooling

2. **Database Scaling**
   - Read replicas cho historical data queries
   - Connection pooling với PgBouncer
   - Partitioning cho large historical tables

3. **Cache Scaling**
   - Redis Cluster cho high availability
   - Cache warming strategies
   - TTL optimization

4. **Future Migration Path**
   - Message queue introduction (Redis Streams → Kafka)
   - Microservices decomposition
   - Cloud-native deployment (Kubernetes)

### Resource Requirements

**MVP Single Server**:
- CPU: 4 cores
- RAM: 8GB
- Storage: 100GB SSD
- Network: 100Mbps

**Expected Load**:
- 50 concurrent WebSocket connections
- 1000 API requests/hour
- 5-minute data collection intervals
- 1 year data retention

### Monitoring và Health Checks

**Health Check Endpoints**:
```
GET /health/data-collector    -> Service health + last collection time
GET /health/data-processor    -> Processing status + queue length
GET /health/websocket-server  -> Active connections + uptime
GET /health/rest-api          -> API status + response times
GET /health/system            -> Overall system health
```

**Log Structure**:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "data-collector",
  "message": "Successfully collected data from weather_api",
  "context": {
    "source": "weather_api",
    "records_count": 15,
    "response_time_ms": 245,
    "data_quality": 0.95
  }
}
```

**Basic Metrics**:
- Data collection success rate
- API response times
- Active WebSocket connections
- Database query performance
- Memory và CPU usage

Thiết kế này cung cấp foundation solid cho MVP với khả năng scale up khi cần thiết, đồng thời giữ complexity ở mức tối thiểu cho giai đoạn hiện tại.