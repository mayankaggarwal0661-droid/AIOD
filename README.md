# AIOD — AI Image Object Detection & Vision Analytics

AIOD (AI Image Object Detection) is a modular, high-performance computer vision and vision analytics platform powered by the **YOLO26n** neural network. It offers multiple interfaces: a web dashboard served on localhost, standalone desktop applications, and automated folder-level batch processing, backed by a relational database for detection persistence and vision analytics.

---

## 1. Project Overview

AIOD allows developers and computer vision practitioners to:
* Detect and classify objects in individual images or entire image directories.
* Extract granular bounding-box coordinates, individual confidence scores, and class counts.
* Store detections into a structured relational database (MySQL with automatic SQLite fallback).
* Generate visual scene descriptions and insights using Google Gemini multimodal vision.
* Access the platform via a responsive Web Dashboard (`http://localhost:5000`) or native Tkinter desktop applications.

---

## 2. Main Features

* **High-Accuracy Object Detection**: Powered by `yolo26n.pt` with 80 standard COCO object classes.
* **Thread-Safe Shared Detection Engine**: Single cached neural network instance prevents duplicate model loads in memory.
* **Configurable Detection Parameters**:
  * Real-time confidence threshold tuning (`0.0` to `1.0`).
  * Intersection-over-Union (IoU) threshold tuning for Non-Maximum Suppression (NMS).
  * Flexible class filtering (e.g. `person, car, dog`) at inference time.
* **Structured Detection Output**:
  * Bounding-box coordinates: `(x1, y1, x2, y2)`, `.width`, `.height`, and `.area`.
  * Total detected object count and class-wise frequency breakdown.
  * Inference latency tracking (`preprocess`, `inference`, `postprocess`).
* **Batch Image Processing**:
  * Automatic discovery of supported image formats (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`).
  * Pre-validation of file headers to catch corrupted images before inference.
  * Sequential execution to protect GPU VRAM and CPU memory.
  * Resilient error recovery: faulty images are logged as failed while processing continues for valid images.
  * Automatic generation and saving of annotated output images.
* **Relational Database Architecture**:
  * Master-child tables: `detection_history` (master metadata) and `detected_objects` (individual bounding boxes).
  * Parameterized queries prevent SQL injection.
  * Atomic transactions with automatic commit and rollback.
  * Non-destructive schema migration ensures existing tables and legacy queries are preserved without data loss.
  * Automatic fallback to local SQLite (`aiod_database.db`) if MySQL is offline.
* **Multimodal AI Scene Insights**:
  * Integrated Google Gemini Vision for deep scene captioning and visual QA on detected images.
* **Multi-Interface Access**:
  * **Web Application**: Flask dashboard at `http://localhost:5000`.
  * **Standalone Desktop App**: Tkinter GUI (`AIOD.py`).
  * **Database-Connected Desktop App**: Tkinter GUI (`AIOD_Database.py`).

---

## 3. Technologies Used

* **Language**: Python 3.10+
* **Deep Learning & Inference**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), PyTorch, Torchvision
* **Computer Vision & Image Processing**: OpenCV (`opencv-python`), Pillow (`PIL`)
* **Databases**: MySQL (via `mysql-connector-python`), SQLite3 (embedded standard library)
* **Web Framework**: Flask, Werkzeug
* **Multimodal AI**: Google Gemini Generative Language API
* **Automated Testing**: `pytest`, `unittest`

---

## 4. Python Version Requirements

* **Python Version**: **Python 3.10, 3.11, or 3.12** (64-bit recommended)
* **Operating System**: Windows 10/11, Linux, or macOS

Verify your Python version:
```bash
python --version
```

---

## 5. Project Folder Structure

```
AIOD/
├── aiod/
│   ├── core/
│   │   ├── config.py             # Runtime configuration & environment variables
│   │   ├── models.py             # DetectionBox, DetectionResult domain dataclasses
│   │   └── detector.py           # Thread-safe YOLODetector singleton engine
│   ├── database/
│   │   ├── connection.py         # MySQL connection manager with SQLite fallback
│   │   ├── repository.py         # Relational CRUD and atomic transaction operations
│   │   └── schema.sql            # Master-child DDL schema definitions
│   ├── services/
│   │   ├── detection_service.py  # Coordinates image detection, filtering, and persistence
│   │   ├── batch_service.py      # Sequential batch processor and summary reporter
│   │   └── gemini_service.py     # Multimodal AI vision scene analysis
│   ├── analytics/
│   │   └── vision_analytics.py   # Historical statistics, distributions, and metrics
│   ├── utils/
│   │   ├── exceptions.py         # Custom exceptions (AIODError, InvalidImageError, etc.)
│   │   └── image_utils.py        # Image discovery, validation, and thumbnailing
│   └── tests/                    # 90+ automated tests (pytest & unittest)
│       ├── test_detector.py      # Detection engine unit tests (mock & actual YOLO)
│       ├── test_database.py      # Database integration tests
│       ├── test_database_mock.py # Parameterized queries & rollback tests
│       ├── test_batch_service.py # Batch processing resilience tests
│       ├── test_services.py      # Service layer class filtering & threshold tests
│       ├── test_web_api.py       # Web REST API integration tests
│       ├── test_gemini_service.py# Gemini Vision tests
│       ├── test_analytics.py     # Analytics aggregation tests
│       └── test_stability_pytest.py # Comprehensive pytest regression suite
├── templates/
│   └── index.html                # Responsive web dashboard
├── static/
│   ├── uploads/                  # Uploaded input images
│   └── annotated/                # Generated annotated output images
├── AIOD.py                       # Standalone desktop launch script
├── AIOD_Database.py              # Desktop launch script with MySQL persistence
├── app.py                        # Flask web application server
├── yolo26n.pt                    # Primary YOLO model weights checkpoint (5.5 MB)
├── for_the_database              # Historical schema reference
├── requirements.txt              # Pinned Python package dependencies
├── .env.example                  # Environment variable configuration template
├── .env                          # Local environment variables (ignored by git)
└── .gitignore                    # Git ignore file for secrets and environments
```

---

## 6. Installation Instructions

### Step 1: Clone or Navigate to the Repository
```bash
git clone https://github.com/ghoshaayush13-create/AIOD.git
cd AIOD
```

### Step 2: Create a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 7. YOLO Model Setup

AIOD includes the primary weight checkpoint **`yolo26n.pt`** (5.5 MB) in the project root directory.

* The model is automatically detected at `yolo26n.pt`.
* To point to a custom checkpoint or a different YOLO model weight file, set the `AIOD_MODEL_PATH` environment variable in your `.env` file:
  ```env
  AIOD_MODEL_PATH=path/to/your/weights.pt
  ```
* **Hardware Acceleration**: The engine automatically detects and uses an NVIDIA GPU (CUDA) if compatible drivers and PyTorch with CUDA are installed; otherwise, it safely runs on the CPU.

---

## 8. MySQL Database Configuration

AIOD persists detection runs into a structured relational schema:

1. Ensure MySQL Server is running locally or remotely (default port: `3306`).
2. Create the target database (if it doesn't already exist):
   ```sql
   CREATE DATABASE IF NOT EXISTS object_detection_db;
   ```
3. Update database credentials in your `.env` file (see Section 9).
4. **Automatic Schema Migration**: When AIOD starts, it automatically executes a safe, non-destructive migration. It checks existing columns and creates the `detection_history` and `detected_objects` tables without dropping or modifying existing data.
5. **No MySQL Installed?** No problem. If MySQL is offline or not installed, AIOD automatically falls back to an embedded SQLite database (`aiod_database.db`). The app will continue running without crashing.

---

## 9. Environment Variable Setup

Copy the `.env.example` template to `.env`:

```bash
# Windows PowerShell
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` in any text editor and adjust the settings:

```env
# Optional: Google Gemini API Key for AI Scene Insights
GEMINI_API_KEY=your_gemini_api_key_here

# MySQL Connection Details
AIOD_DB_HOST=localhost
AIOD_DB_PORT=3306
AIOD_DB_USER=root
AIOD_DB_PASSWORD=your_mysql_password_here
AIOD_DB_NAME=object_detection_db

# Default Detection Thresholds
AIOD_CONFIDENCE=0.25
AIOD_IOU=0.45

# Device Selection: Leave empty for auto-detection, or specify 'cpu' / 'cuda'
AIOD_DEVICE=

# SQLite Fallback: Set to 'true' to enable local fallback when MySQL is unreachable
AIOD_SQLITE_FALLBACK=true
```

> **Security Note**: Never commit your `.env` file containing real passwords or API keys to version control. `.env` is listed in `.gitignore`.

---

## 10. How to Launch the Application

### Option A: Web Dashboard (Recommended)
Starts the web server on localhost with drag-and-drop detection, batch folder processing, Gemini AI insights, and analytics history:

```bash
python app.py
```
Open your browser and navigate to:
**[http://localhost:5000](http://localhost:5000)** (or `http://127.0.0.1:5000`)

### Option B: Standalone Desktop GUI
Launches the native Tkinter window for single-image detection without database requirement:
```bash
python AIOD.py
```

### Option C: Database-Connected Desktop GUI
Launches the native Tkinter window with automatic database logging:
```bash
python AIOD_Database.py
```

---

## 11. How to Run Automated Tests

AIOD includes a comprehensive test suite (96+ automated tests) covering unit logic, mock database connections, batch resilience, and end-to-end integration:

### Run with `pytest`:
```bash
# Run the complete test suite
pytest aiod/tests -v

# Run the dedicated stability and regression tests
pytest aiod/tests/test_stability_pytest.py -v
```

### Run with standard Python `unittest`:
```bash
python -m unittest discover -s aiod/tests
```

---

## 12. Common Errors and Their Solutions

### 1. `Model weight file not found at: yolo26n.pt`
* **Cause**: The model checkpoint is missing from the working directory.
* **Solution**: Ensure `yolo26n.pt` is in the project root folder, or set the exact path in `.env`:
  ```env
  AIOD_MODEL_PATH=C:/path/to/yolo26n.pt
  ```

### 2. `Could not establish MySQL connection`
* **Cause**: MySQL server is stopped, port 3306 is blocked, or credentials in `.env` are incorrect.
* **Solution**: 
  1. Verify MySQL is running: `mysqladmin -u root -p status`.
  2. Check `AIOD_DB_USER` and `AIOD_DB_PASSWORD` in `.env`.
  3. Ensure `AIOD_SQLITE_FALLBACK=true` in `.env` so the application smoothly uses SQLite while MySQL is offline.

### 3. `Port 5000 is already in use`
* **Cause**: Another process or background instance of Flask is already running on port 5000.
* **Solution**: Stop the existing process:
  * **Windows**:
    ```powershell
    Get-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess | Stop-Process -Force
    ```
  * **Linux / macOS**:
    ```bash
    fuser -k 5000/tcp
    ```

### 4. `Unsupported file format` or `Corrupted or invalid image data`
* **Cause**: A non-image file (.txt, .pdf) or a truncated/empty image was selected.
* **Solution**: AIOD validates images before inference. Select a valid image file formatted as `.jpg`, `.jpeg`, `.png`, `.webp`, or `.bmp`.

### 5. `Detection failed: DetectionService.detect_and_save() got an unexpected keyword argument`
* **Cause**: Calling outdated function signatures.
* **Solution**: `DetectionService` accepts `classes`, `annotated_image_path`, and `conf`. Pass classes as a list of strings (e.g. `classes=['person', 'car']`) or comma-separated in the web interface. Leaving the class filter blank detects all supported classes.

---

## License

This project is licensed under the Apache 2.0 License.
