import sqlite3
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("factoryeye.database")

DB_DIR = Path("data")
DB_PATH = DB_DIR / "factoryeye_audit.db"

class DefectAuditDatabase:
    """
    Persistent SQLite database for storing factory defect inspection records,
    providing historical audit trails and real-time quality assurance analytics.
    """
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inspection_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc REAL NOT NULL,
                    datetime_iso TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    defect_count INTEGER NOT NULL,
                    defect_detected INTEGER NOT NULL,
                    inference_ms REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS defect_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id INTEGER,
                    timestamp_utc REAL NOT NULL,
                    station_id TEXT NOT NULL,
                    defect_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    FOREIGN KEY (inspection_id) REFERENCES inspection_events(id)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_defect_class ON defect_records(defect_class)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_defect_timestamp ON defect_records(timestamp_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspection_timestamp ON inspection_events(timestamp_utc)")
            conn.commit()
            logger.info(f"✓ Defect audit database initialized at {self.db_path}")

    def log_inspection(
        self,
        defect_count: int,
        detections: List[Dict],
        inference_ms: float,
        source: str = "REST API",
        station_id: str = "STATION_01"
    ) -> int:
        """Logs an inspection event and individual detected defect bounding boxes."""
        now = time.time()
        datetime_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO inspection_events 
                (timestamp_utc, datetime_iso, station_id, source, defect_count, defect_detected, inference_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (now, datetime_iso, station_id, source, defect_count, 1 if defect_count > 0 else 0, inference_ms))
            inspection_id = cursor.lastrowid

            for d in detections:
                label = d.get("label", "") if isinstance(d, dict) else getattr(d, "label", "")
                conf = d.get("confidence", 0.0) if isinstance(d, dict) else getattr(d, "confidence", 0.0)
                bbox = d.get("bbox", [0, 0, 0, 0]) if isinstance(d, dict) else getattr(d, "bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox if len(bbox) == 4 else (0, 0, 0, 0)

                cursor.execute("""
                    INSERT INTO defect_records
                    (inspection_id, timestamp_utc, station_id, defect_class, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (inspection_id, now, station_id, label, conf, x1, y1, x2, y2))
            
            conn.commit()
            return inspection_id

    def query_defects(
        self,
        limit: int = 50,
        offset: int = 0,
        defect_class: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> Tuple[List[Dict], int]:
        """Queries historical defect records with optional filtering and pagination."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            conditions = ["confidence >= ?"]
            params = [min_confidence]

            if defect_class:
                conditions.append("defect_class = ?")
                params.append(defect_class)

            where_clause = " AND ".join(conditions)

            # Count total matching
            cursor.execute(f"SELECT COUNT(*) FROM defect_records WHERE {where_clause}", params)
            total = cursor.fetchone()[0]

            # Fetch records
            query = f"""
                SELECT id, inspection_id, timestamp_utc, station_id, defect_class, confidence,
                       bbox_x1, bbox_y1, bbox_x2, bbox_y2
                FROM defect_records
                WHERE {where_clause}
                ORDER BY timestamp_utc DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
            cursor.execute(query, params)
            rows = cursor.fetchall()

            results = []
            for r in rows:
                results.append({
                    "id": r["id"],
                    "inspection_id": r["inspection_id"],
                    "timestamp_utc": r["timestamp_utc"],
                    "datetime_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r["timestamp_utc"])),
                    "station_id": r["station_id"],
                    "defect_class": r["defect_class"],
                    "confidence": round(r["confidence"], 4),
                    "bbox": [r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]]
                })
            return results, total

    def get_summary_stats(self) -> Dict:
        """Calculates real-time quality assurance line statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total inspections
            cursor.execute("SELECT COUNT(*), SUM(defect_detected), AVG(inference_ms) FROM inspection_events")
            row = cursor.fetchone()
            total_inspections = row[0] or 0
            defective_inspections = row[1] or 0
            avg_latency = row[2] or 0.0

            defect_rate = round((defective_inspections / total_inspections * 100), 2) if total_inspections > 0 else 0.0
            quality_yield = round(100.0 - defect_rate, 2) if total_inspections > 0 else 100.0

            # Defect breakdown by class
            cursor.execute("""
                SELECT defect_class, COUNT(*) as count, AVG(confidence) as avg_conf
                FROM defect_records
                GROUP BY defect_class
                ORDER BY count DESC
            """)
            class_breakdown = [
                {"class": r["defect_class"], "count": r["count"], "avg_confidence": round(r["avg_conf"], 4)}
                for r in cursor.fetchall()
            ]

            return {
                "total_inspections": total_inspections,
                "clean_inspections": total_inspections - defective_inspections,
                "defective_inspections": defective_inspections,
                "defect_rate_percent": defect_rate,
                "quality_yield_percent": quality_yield,
                "mean_inference_ms": round(avg_latency, 2),
                "defect_class_breakdown": class_breakdown
            }

audit_db = DefectAuditDatabase()
