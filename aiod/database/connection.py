"""Database connection management supporting MySQL with automatic SQLite fallback."""

import sqlite3
from typing import Optional, Tuple, Any, Set
from pathlib import Path
import mysql.connector
from mysql.connector import Error as MySQLError

from aiod.core.config import settings
from aiod.utils.exceptions import DatabaseError


class DatabaseConnectionManager:
    """Manages secure connections to MySQL with seamless fallback to SQLite."""

    def __init__(self):
        self._is_mysql: Optional[bool] = None
        self._schema_initialized: bool = False

    def ensure_schema(self, conn: Optional[Any] = None, backend: Optional[str] = None):
        """Ensure schema is initialized once."""
        if not self._schema_initialized:
            self.init_schema(conn=conn, backend=backend)
            self._schema_initialized = True

    def test_mysql_connection(self) -> Tuple[bool, Optional[str]]:
        """
        Test whether MySQL is reachable with current environment credentials.

        Returns:
            Tuple of (is_connected: bool, error_message: Optional[str]).
        """
        conn = None
        try:
            conn = mysql.connector.connect(
                host=settings.db_host,
                port=settings.db_port,
                user=settings.db_user,
                password=settings.db_password,
                database=settings.db_name,
                connect_timeout=3,
            )
            if conn.is_connected():
                return True, None
            return False, "Could not establish MySQL connection."
        except MySQLError as err:
            return False, str(err)
        except Exception as e:
            return False, str(e)
        finally:
            if conn and conn.is_connected():
                conn.close()

    def get_connection(self) -> Tuple[Any, str]:
        """
        Get an active database connection.
        Attempts MySQL first; falls back to SQLite if MySQL is unreachable.

        Returns:
            Tuple of (connection_object, backend_type: 'mysql' | 'sqlite').

        Raises:
            DatabaseError: If connection to both MySQL and SQLite fails.
        """
        # 1. Attempt MySQL connection
        try:
            conn = mysql.connector.connect(
                host=settings.db_host,
                port=settings.db_port,
                user=settings.db_user,
                password=settings.db_password,
                database=settings.db_name,
                connect_timeout=3,
                autocommit=False,  # Enforce transaction safety
            )
            if conn.is_connected():
                self._is_mysql = True
                return conn, "mysql"
        except Exception:
            # If MySQL connection fails and fallback is disabled, raise error
            if not settings.enable_sqlite_fallback:
                raise DatabaseError(
                    f"MySQL connection to {settings.db_host}:{settings.db_port} failed "
                    f"and SQLite fallback is disabled."
                )

        # 2. Fallback to SQLite
        try:
            conn = sqlite3.connect(str(settings.sqlite_db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            # Enable foreign keys in SQLite
            conn.execute("PRAGMA foreign_keys = ON;")
            self._is_mysql = False
            return conn, "sqlite"
        except Exception as e:
            raise DatabaseError(f"Failed to connect to SQLite fallback database: {e}") from e

    def init_schema(self, conn: Optional[Any] = None, backend: Optional[str] = None) -> bool:
        """
        Non-destructively initialize or migrate the database schema.
        Preserves all existing tables and data, safely adding any missing columns.

        Args:
            conn: Optional existing connection. If None, acquires a new connection.
            backend: Optional backend identifier ('mysql' | 'sqlite').

        Returns:
            True if schema initialization and migration succeeded.
        """
        should_close = False
        if conn is None:
            conn, backend = self.get_connection()
            should_close = True

        cursor = None
        try:
            cursor = conn.cursor()

            if backend == "mysql":
                self._init_mysql_schema(cursor)
            else:
                self._init_sqlite_schema(cursor)

            conn.commit()
            self._schema_initialized = True
            return True
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise DatabaseError(f"Schema initialization/migration failed on {backend}: {e}") from e
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if should_close and conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def _init_mysql_schema(self, cursor: Any):
        """Execute non-destructive MySQL schema creation and column migration."""
        # 1. Create master detection_history table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_path VARCHAR(500) NOT NULL,
                annotated_image_path VARCHAR(500),
                model_name VARCHAR(100) DEFAULT 'yolo26n.pt',
                confidence_threshold FLOAT DEFAULT 0.25,
                inference_time_ms FLOAT DEFAULT 0.0,
                total_detected_objects INT DEFAULT 0,
                detected_objects TEXT,
                detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 2. Safe migration: check existing columns and add missing ones without data loss
        cursor.execute("SHOW COLUMNS FROM detection_history;")
        existing_cols: Set[str] = {row[0].lower() for row in cursor.fetchall()}

        col_definitions = [
            ("annotated_image_path", "ALTER TABLE detection_history ADD COLUMN annotated_image_path VARCHAR(500);"),
            ("model_name", "ALTER TABLE detection_history ADD COLUMN model_name VARCHAR(100) DEFAULT 'yolo26n.pt';"),
            ("confidence_threshold", "ALTER TABLE detection_history ADD COLUMN confidence_threshold FLOAT DEFAULT 0.25;"),
            ("inference_time_ms", "ALTER TABLE detection_history ADD COLUMN inference_time_ms FLOAT DEFAULT 0.0;"),
            ("total_detected_objects", "ALTER TABLE detection_history ADD COLUMN total_detected_objects INT DEFAULT 0;"),
            ("detected_objects", "ALTER TABLE detection_history ADD COLUMN detected_objects TEXT;"),
        ]

        for col_name, alter_stmt in col_definitions:
            if col_name.lower() not in existing_cols:
                cursor.execute(alter_stmt)

        # 3. Create granular detected_objects table with foreign key
        cursor.execute("""
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
                FOREIGN KEY (detection_id) REFERENCES detection_history(id) ON DELETE CASCADE
            );
        """)

    def _init_sqlite_schema(self, cursor: Any):
        """Execute non-destructive SQLite schema creation and column migration."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                annotated_image_path TEXT,
                model_name TEXT DEFAULT 'yolo26n.pt',
                confidence_threshold REAL DEFAULT 0.25,
                inference_time_ms REAL DEFAULT 0.0,
                total_detected_objects INTEGER DEFAULT 0,
                detected_objects TEXT,
                detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cursor.execute("PRAGMA table_info(detection_history);")
        existing_cols: Set[str] = {row[1].lower() for row in cursor.fetchall()}

        col_definitions = [
            ("annotated_image_path", "ALTER TABLE detection_history ADD COLUMN annotated_image_path TEXT;"),
            ("model_name", "ALTER TABLE detection_history ADD COLUMN model_name TEXT DEFAULT 'yolo26n.pt';"),
            ("confidence_threshold", "ALTER TABLE detection_history ADD COLUMN confidence_threshold REAL DEFAULT 0.25;"),
            ("inference_time_ms", "ALTER TABLE detection_history ADD COLUMN inference_time_ms REAL DEFAULT 0.0;"),
            ("total_detected_objects", "ALTER TABLE detection_history ADD COLUMN total_detected_objects INTEGER DEFAULT 0;"),
            ("detected_objects", "ALTER TABLE detection_history ADD COLUMN detected_objects TEXT;"),
        ]

        for col_name, alter_stmt in col_definitions:
            if col_name.lower() not in existing_cols:
                cursor.execute(alter_stmt)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detected_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER NOT NULL,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox_x1 REAL NOT NULL,
                bbox_y1 REAL NOT NULL,
                bbox_x2 REAL NOT NULL,
                bbox_y2 REAL NOT NULL,
                FOREIGN KEY (detection_id) REFERENCES detection_history(id) ON DELETE CASCADE
            );
        """)


# Global connection manager instance
db_manager = DatabaseConnectionManager()
