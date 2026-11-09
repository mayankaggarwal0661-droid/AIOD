"""Data access layer for persisting and retrieving structured detection records."""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from aiod.core.models import DetectionResult, DetectionBox
from aiod.database.connection import db_manager, DatabaseConnectionManager
from aiod.utils.exceptions import DatabaseError


class DetectionRepository:
    """
    Repository for managing detection history and granular detected objects.
    Enforces parameterized queries, transaction atomicity, and safe connection lifecycle.
    """

    def __init__(self, connection_manager: Optional[DatabaseConnectionManager] = None):
        """
        Initialize the repository.

        Args:
            connection_manager: Optional connection manager instance (defaults to global db_manager).
        """
        self.conn_mgr = connection_manager if connection_manager is not None else db_manager

    def init_schema(self) -> bool:
        """
        Ensure database schema and tables exist with safe migration.

        Returns:
            True if schema is verified or created.
        """
        return self.conn_mgr.init_schema()

    def save_detection(
        self,
        image_path: Optional[Union[str, Path, DetectionResult]] = None,
        annotated_image_path: Optional[Union[str, Path]] = None,
        model_name: str = "yolo26n.pt",
        confidence_threshold: float = 0.25,
        inference_time_ms: float = 0.0,
        boxes: Optional[List[DetectionBox]] = None,
        result: Optional[DetectionResult] = None,
    ) -> Tuple[int, str]:
        """
        Save a detection and all associated bounding boxes in a single atomic transaction.

        Supports passing either individual parameters or a DetectionResult domain object.

        Args:
            image_path: Path to original input image, or DetectionResult instance.
            annotated_image_path: Path to annotated output image.
            model_name: Neural network model identifier.
            confidence_threshold: Confidence threshold used.
            inference_time_ms: Inference execution time in milliseconds.
            boxes: List of DetectionBox instances.
            result: Optional DetectionResult object (overrides individual arguments if provided).

        Returns:
            Tuple of (detection_id: int, backend_used: str).

        Raises:
            DatabaseError: If database insert or transaction fails.
        """
        # Handle DetectionResult passed as first positional argument
        if isinstance(image_path, DetectionResult):
            result = image_path
            image_path = None

        # Populate from DetectionResult if provided
        if result is not None:
            resolved_image_path = str(result.image_path) if result.image_path else (str(image_path) if image_path else "")
            resolved_boxes = result.boxes
            resolved_inference_time = result.inference_time_ms
            legacy_string = result.legacy_detected_objects_string
        else:
            resolved_image_path = str(image_path) if image_path else ""
            resolved_boxes = boxes if boxes is not None else []
            resolved_inference_time = inference_time_ms
            legacy_string = ", ".join(b.label for b in resolved_boxes)

        annotated_str = str(annotated_image_path) if annotated_image_path else None
        total_objects = len(resolved_boxes)

        conn = None
        cursor = None
        backend = "unknown"

        try:
            # Ensure schema exists once before inserting
            self.conn_mgr.ensure_schema()
            conn, backend = self.conn_mgr.get_connection()

            cursor = conn.cursor()

            # 1. Parameterized insert into detection_history
            if backend == "mysql":
                history_query = """
                    INSERT INTO detection_history 
                    (image_path, annotated_image_path, model_name, confidence_threshold, 
                     inference_time_ms, total_detected_objects, detected_objects)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                """
            else:
                history_query = """
                    INSERT INTO detection_history 
                    (image_path, annotated_image_path, model_name, confidence_threshold, 
                     inference_time_ms, total_detected_objects, detected_objects)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                """

            history_params = (
                resolved_image_path,
                annotated_str,
                model_name,
                float(confidence_threshold),
                float(resolved_inference_time),
                int(total_objects),
                legacy_string,
            )

            cursor.execute(history_query, history_params)
            detection_id = int(cursor.lastrowid)

            # 2. Parameterized batch insert into detected_objects child table
            if detection_id and resolved_boxes:
                if backend == "mysql":
                    objects_query = """
                        INSERT INTO detected_objects 
                        (detection_id, class_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """
                else:
                    objects_query = """
                        INSERT INTO detected_objects 
                        (detection_id, class_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """

                objects_data = [
                    (
                        detection_id,
                        int(box.class_id),
                        str(box.class_name),
                        float(box.confidence),
                        float(box.bbox[0]),
                        float(box.bbox[1]),
                        float(box.bbox[2]),
                        float(box.bbox[3]),
                    )
                    for box in resolved_boxes
                ]

                cursor.executemany(objects_query, objects_data)

            # Commit the atomic transaction
            conn.commit()
            return detection_id, backend

        except Exception as e:
            # Transaction rollback on any error
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise DatabaseError(f"Failed to persist detection in {backend}: {e}") from e

        finally:
            # Safe resource closure
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def get_recent_detections(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve recent detection history records with all columns.

        Args:
            limit: Maximum number of rows to retrieve.

        Returns:
            List of dictionaries representing master detection records.
        """
        conn = None
        cursor = None
        try:
            conn, backend = self.conn_mgr.get_connection()
            cursor = conn.cursor()

            query = f"""
                SELECT id, image_path, annotated_image_path, model_name, confidence_threshold, 
                       inference_time_ms, total_detected_objects, detected_objects, detection_time 
                FROM detection_history 
                ORDER BY id DESC 
                LIMIT {int(limit)};
            """
            cursor.execute(query)
            rows = cursor.fetchall()

            history: List[Dict[str, Any]] = []
            for row in rows:
                if hasattr(row, "keys"):
                    history.append({
                        "id": row["id"],
                        "image_path": row["image_path"],
                        "annotated_image_path": row["annotated_image_path"],
                        "model_name": row["model_name"],
                        "confidence_threshold": row["confidence_threshold"],
                        "inference_time_ms": row["inference_time_ms"],
                        "total_detected_objects": row["total_detected_objects"],
                        "detected_objects": row["detected_objects"],
                        "detection_time": str(row["detection_time"]),
                    })
                else:
                    history.append({
                        "id": row[0],
                        "image_path": row[1],
                        "annotated_image_path": row[2],
                        "model_name": row[3],
                        "confidence_threshold": row[4],
                        "inference_time_ms": row[5],
                        "total_detected_objects": row[6],
                        "detected_objects": row[7],
                        "detection_time": str(row[8]),
                    })

            return history

        except Exception as e:
            raise DatabaseError(f"Failed to fetch detection history: {e}") from e

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    # Backward-compatible alias
    get_recent_history = get_recent_detections

    def get_detection_by_id(self, detection_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single detection record by its ID.

        Args:
            detection_id: Primary key of detection_history.

        Returns:
            Dictionary with detection fields or None if not found.
        """
        conn = None
        cursor = None
        try:
            conn, backend = self.conn_mgr.get_connection()
            cursor = conn.cursor()

            query = """
                SELECT id, image_path, annotated_image_path, model_name, confidence_threshold, 
                       inference_time_ms, total_detected_objects, detected_objects, detection_time 
                FROM detection_history 
                WHERE id = %s;
            """ if backend == "mysql" else """
                SELECT id, image_path, annotated_image_path, model_name, confidence_threshold, 
                       inference_time_ms, total_detected_objects, detected_objects, detection_time 
                FROM detection_history 
                WHERE id = ?;
            """

            cursor.execute(query, (int(detection_id),))
            row = cursor.fetchone()

            if not row:
                return None

            if hasattr(row, "keys"):
                return {
                    "id": row["id"],
                    "image_path": row["image_path"],
                    "annotated_image_path": row["annotated_image_path"],
                    "model_name": row["model_name"],
                    "confidence_threshold": row["confidence_threshold"],
                    "inference_time_ms": row["inference_time_ms"],
                    "total_detected_objects": row["total_detected_objects"],
                    "detected_objects": row["detected_objects"],
                    "detection_time": str(row["detection_time"]),
                }
            return {
                "id": row[0],
                "image_path": row[1],
                "annotated_image_path": row[2],
                "model_name": row[3],
                "confidence_threshold": row[4],
                "inference_time_ms": row[5],
                "total_detected_objects": row[6],
                "detected_objects": row[7],
                "detection_time": str(row[8]),
            }

        except Exception as e:
            raise DatabaseError(f"Failed to fetch detection by id {detection_id}: {e}") from e

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def get_objects_for_detection(self, detection_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve all granular detected objects associated with a detection ID.

        Args:
            detection_id: Foreign key referencing detection_history.

        Returns:
            List of dictionaries for each detected bounding box.
        """
        conn = None
        cursor = None
        try:
            conn, backend = self.conn_mgr.get_connection()
            cursor = conn.cursor()

            query = """
                SELECT id, detection_id, class_id, class_name, confidence, 
                       bbox_x1, bbox_y1, bbox_x2, bbox_y2 
                FROM detected_objects 
                WHERE detection_id = %s 
                ORDER BY id ASC;
            """ if backend == "mysql" else """
                SELECT id, detection_id, class_id, class_name, confidence, 
                       bbox_x1, bbox_y1, bbox_x2, bbox_y2 
                FROM detected_objects 
                WHERE detection_id = ? 
                ORDER BY id ASC;
            """

            cursor.execute(query, (int(detection_id),))
            rows = cursor.fetchall()

            objects: List[Dict[str, Any]] = []
            for row in rows:
                if hasattr(row, "keys"):
                    objects.append({
                        "id": row["id"],
                        "detection_id": row["detection_id"],
                        "class_id": row["class_id"],
                        "class_name": row["class_name"],
                        "confidence": float(row["confidence"]),
                        "bbox": (
                            float(row["bbox_x1"]),
                            float(row["bbox_y1"]),
                            float(row["bbox_x2"]),
                            float(row["bbox_y2"]),
                        ),
                        "bbox_x1": float(row["bbox_x1"]),
                        "bbox_y1": float(row["bbox_y1"]),
                        "bbox_x2": float(row["bbox_x2"]),
                        "bbox_y2": float(row["bbox_y2"]),
                    })
                else:
                    objects.append({
                        "id": row[0],
                        "detection_id": row[1],
                        "class_id": row[2],
                        "class_name": row[3],
                        "confidence": float(row[4]),
                        "bbox": (float(row[5]), float(row[6]), float(row[7]), float(row[8])),
                        "bbox_x1": float(row[5]),
                        "bbox_y1": float(row[6]),
                        "bbox_x2": float(row[7]),
                        "bbox_y2": float(row[8]),
                    })

            return objects

        except Exception as e:
            raise DatabaseError(f"Failed to fetch objects for detection {detection_id}: {e}") from e

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def get_detection_with_objects(self, detection_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve master detection record combined with its child detected objects.

        Args:
            detection_id: Primary key of detection_history.

        Returns:
            Dictionary containing detection data and a 'objects' list, or None.
        """
        detection = self.get_detection_by_id(detection_id)
        if not detection:
            return None

        detection["objects"] = self.get_objects_for_detection(detection_id)
        return detection


# Global default repository instance
detection_repository = DetectionRepository()
