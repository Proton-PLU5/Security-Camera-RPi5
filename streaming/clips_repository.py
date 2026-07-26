import sqlite3 as sqlite

class ClipsRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_clip_started_at(self, clip_id) -> float | None:
        conn = sqlite.connect(self.db_path, timeout=5.0)
        try:
            cur = conn.cursor()
            cur.execute("SELECT started_at FROM clips WHERE id = ?", (clip_id,))
            row = cur.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def get_detections_for_clip(self, clip_id):
        conn = sqlite.connect(self.db_path, timeout=5.0)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT timestamp, class_name, confidence, bbox_x, bbox_y, bbox_width, bbox_height "
                "FROM detections WHERE clip_id = ? ORDER BY timestamp ASC",
                (clip_id,),
            )
            return cur.fetchall()
        finally:
            conn.close()

    def get_clips_before(self, before_timestamp_ms: float):
        conn = sqlite.connect(self.db_path)
        conn.row_factory = lambda cur, row: {
            col[0]: val for col, val in zip(cur.description, row)
        }
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM clips WHERE ended_at < ?", (before_timestamp_ms,))
            return cur.fetchall()
        finally:
            conn.close()