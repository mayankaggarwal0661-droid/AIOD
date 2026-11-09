"""Flask Web Application for AIOD Vision Analytics."""

import os
import uuid
import shutil
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from aiod.core.detector import get_detector
from aiod.services.detection_service import detection_service
from aiod.services.batch_service import batch_processor
from aiod.services.gemini_service import gemini_service
from aiod.database.repository import detection_repository
from aiod.analytics.vision_analytics import vision_analytics
from aiod.utils.exceptions import InvalidImageError, ModelError, DatabaseError
from aiod.utils.image_utils import SUPPORTED_EXTENSIONS

logger = logging.getLogger("aiod.web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
ANNOTATED_DIR = BASE_DIR / "static" / "annotated"
BATCH_DIR = BASE_DIR / "static" / "batch_temp"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB max upload limit


@app.route("/")
def index():
    """Render the main dashboard interface."""
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def detect_single_image():
    """Handle single image detection request."""
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Selected file is empty"}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": f"Unsupported format '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"}), 400

    try:
        conf = float(request.form.get("confidence", 0.25))
    except (ValueError, TypeError):
        conf = 0.25

    # Parse and validate class filter
    classes_raw = request.form.get("classes", "").strip()
    class_filter = [c.strip() for c in classes_raw.split(",") if c.strip()] if classes_raw else None

    # Save incoming upload
    unique_name = f"{uuid.uuid4().hex[:10]}_{secure_filename(file.filename)}"
    upload_path = UPLOAD_DIR / unique_name
    file.save(upload_path)

    annotated_name = f"annotated_{unique_name}"
    annotated_path = ANNOTATED_DIR / annotated_name

    try:
        # Run detection and persist to DB via backend service with class filter
        result, saved_db, backend = detection_service.detect_and_save(
            image_path=upload_path,
            conf=conf,
            classes=class_filter,
            annotated_image_path=annotated_path,
        )

        # Save annotated image array to public web directory
        result.save_annotated_image(annotated_path)

        return jsonify({
            "success": True,
            "image_name": file.filename,
            "total_count": result.total_count,
            "counts": result.counts,
            "inference_time_ms": round(result.inference_time_ms, 2),
            "preprocess_time_ms": round(result.preprocess_time_ms, 2),
            "postprocess_time_ms": round(result.postprocess_time_ms, 2),
            "annotated_image_url": f"/static/annotated/{annotated_name}",
            "original_image_url": f"/static/uploads/{unique_name}",
            "detections": [box.to_dict() for box in result.boxes],
            "saved_to_db": saved_db,
            "db_backend": backend,
        })

    except InvalidImageError as e:
        logger.warning(f"Invalid image error: {e}")
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400
    except ModelError as e:
        logger.error(f"Detection model error: {e}", exc_info=True)
        return jsonify({"error": f"Model inference error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected detection error: {e}", exc_info=True)
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500


@app.route("/api/batch", methods=["POST"])
def process_batch_images():
    """Handle batch image processing request."""
    files = request.files.getlist("images")
    if not files or len(files) == 0:
        return jsonify({"error": "No images provided for batch processing"}), 400

    try:
        conf = float(request.form.get("confidence", 0.25))
    except (ValueError, TypeError):
        conf = 0.25

    classes_raw = request.form.get("classes", "").strip()
    class_filter = [c.strip() for c in classes_raw.split(",") if c.strip()] if classes_raw else None

    # Create an isolated batch subfolder
    batch_session_id = uuid.uuid4().hex[:8]
    batch_in = BATCH_DIR / f"in_{batch_session_id}"
    batch_out = BATCH_DIR / f"out_{batch_session_id}"
    batch_in.mkdir(parents=True, exist_ok=True)
    batch_out.mkdir(parents=True, exist_ok=True)

    try:
        # Save batch files
        for f in files:
            if f.filename:
                safe_name = secure_filename(f.filename)
                f.save(batch_in / safe_name)

        # Execute batch processor
        summary = batch_processor.process_directory(
            input_directory=batch_in,
            output_directory=batch_out,
            conf=conf,
            classes=class_filter,
            save_annotated=True,
        )

        # Build response with accessible URLs
        item_responses = []
        for item in summary.items:
            annotated_url = None
            if item.annotated_output_path and item.annotated_output_path.exists():
                public_annotated = ANNOTATED_DIR / item.annotated_output_path.name
                shutil.copy2(item.annotated_output_path, public_annotated)
                annotated_url = f"/static/annotated/{item.annotated_output_path.name}"

            item_responses.append({
                "image_name": item.image_path.name,
                "status": item.status,
                "error_message": item.error_message,
                "total_objects": item.total_objects,
                "counts": item.counts,
                "processing_time_ms": item.processing_time_ms,
                "annotated_output_url": annotated_url,
            })

        return jsonify({
            "success": True,
            "total_discovered": summary.total_discovered,
            "total_successful": summary.total_successful,
            "total_failed": summary.total_failed,
            "total_objects_detected": summary.total_objects_detected,
            "aggregated_class_counts": summary.aggregated_class_counts,
            "total_processing_time_ms": summary.total_processing_time_ms,
            "average_time_per_image_ms": summary.average_time_per_image_ms,
            "items": item_responses,
        })

    except Exception as e:
        logger.error(f"Batch processing error: {e}", exc_info=True)
        return jsonify({"error": f"Batch run failed: {str(e)}"}), 500


@app.route("/api/gemini/analyze", methods=["POST"])
def analyze_with_gemini():
    """Analyze uploaded or previously detected image with Gemini Vision."""
    data = request.get_json(silent=True) or request.form
    image_url = data.get("image_url")
    user_prompt = data.get("prompt")
    detection_summary = data.get("detection_summary")

    if not image_url:
        return jsonify({"success": False, "error": "No image_url provided"}), 400

    # Resolve local path from static URL
    rel_path = image_url.lstrip("/")
    local_path = BASE_DIR / rel_path
    if not local_path.exists():
        return jsonify({"success": False, "error": f"Image file not found on server: {rel_path}"}), 404

    result = gemini_service.analyze_scene(
        image_path=local_path,
        user_prompt=user_prompt,
        detection_summary=detection_summary,
    )
    return jsonify(result)


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return recent detection history from database."""
    try:
        history = detection_repository.get_recent_history(limit=50)
        return jsonify({"success": True, "history": history})
    except Exception as e:
        logger.error(f"Failed to retrieve history: {e}", exc_info=True)
        return jsonify({"error": f"Failed to retrieve history: {str(e)}"}), 500


@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """Return aggregated vision analytics summary."""
    try:
        history = detection_repository.get_recent_history(limit=100)
        analytics_summary = vision_analytics.aggregate_history(history)
        return jsonify(analytics_summary)
    except Exception as e:
        logger.error(f"Failed to compute analytics: {e}", exc_info=True)
        return jsonify({"error": f"Failed to compute analytics: {str(e)}"}), 500


if __name__ == "__main__":
    # Preload the YOLO model once on startup
    print("Pre-loading shared YOLO detector...")
    get_detector()
    print("AIOD Vision Analytics Web Platform started!")
    print("Serving on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
