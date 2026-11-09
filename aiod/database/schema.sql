-- =====================================================================
-- AIOD Structured Database Schema (MySQL & SQLite Compatible)
-- =====================================================================

-- Table 1: Detection History (Master Record)
CREATE TABLE IF NOT EXISTS detection_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_path VARCHAR(500) NOT NULL,
    annotated_image_path VARCHAR(500),
    model_name VARCHAR(100) DEFAULT 'yolo26n.pt',
    confidence_threshold FLOAT DEFAULT 0.25,
    inference_time_ms FLOAT DEFAULT 0.0,
    total_detected_objects INT DEFAULT 0,
    detected_objects TEXT, -- Preserved for 100% backward compatibility
    detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_detection_time (detection_time),
    INDEX idx_image_path (image_path(255))
);

-- Table 2: Detected Objects (Granular Child Records)
CREATE TABLE IF NOT EXISTS detected_objects (
    id INT AUTO_INCREMENT PRIMARY KEY,
    detection_id INT NOT NULL,
    class_id INT NOT NULL,
    class_name VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x1 FLOAT NOT NULL,
    bbox_y1 FLOAT NOT NULL,
    bbox_x2 FLOAT NOT NULL,
    bbox_y2 FLOAT NOT NULL,
    INDEX idx_detection_id (detection_id),
    INDEX idx_class_name (class_name),
    INDEX idx_confidence (confidence),
    FOREIGN KEY (detection_id) REFERENCES detection_history(id) ON DELETE CASCADE
);
