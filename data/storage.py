import threading
import sqlite3
import queue
from dataclasses import dataclass
import time
from typing import Optional
import uuid
import logging
import heapq

logger = logging.getLogger(__name__)

@dataclass
class Command:
    type: str
    payload: dict

class StorageThread(threading.Thread):
    def __init__(self, db_path: str = 'storage.db', max_retry_attempts: int = 3, retry_backoff: float = 0.1):
        super().__init__(daemon=True)
        self.db_path = db_path
        self.stopped = threading.Event()
        self.max_retry_attempts = max_retry_attempts
        self.retry_backoff = retry_backoff # Backoff time in seconds between retries
        self._queue: "queue.Queue[Command]" = queue.Queue()

    # ---------- Public API ----------

    def start_clip(self, file_path: str, start_time: float, trigger: str = 'continuous') -> str:
        clip_id = uuid.uuid4().hex  # Generate a unique identifier for the clip
        self.enqueue('start_clip', {'clip_id': clip_id, 'file_path': file_path, 'start_time': start_time, 'trigger': trigger})
        return clip_id

    def end_clip(self, clip_id: str, ended_at: float) -> None:
        self.enqueue('end_clip', {'clip_id': clip_id, 'ended_at': ended_at})

    def insert_detection(self, clip_id: str, timestamp: float, class_name: str, confidence: float,
                          bbox_x: int, bbox_y: int, bbox_width: int, bbox_height: int) -> None:
        self.enqueue('insert_detection', dict(
            clip_id=clip_id, timestamp=timestamp, class_name=class_name, confidence=confidence,
            bbox_x=bbox_x, bbox_y=bbox_y, bbox_width=bbox_width, bbox_height=bbox_height
        ))

    def stop(self) -> None:
        self.enqueue('stop', {})

    def enqueue(self, cmd_type: str, payload: dict) -> None:
        if self.stopped.is_set():
            logger.warning("StorageThread already stopped; dropping %s command", cmd_type)
            return
        
        self._queue.put(Command(cmd_type, payload))

    # ---------- Internals ----------

    def run(self):
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute('PRAGMA foreign_keys = ON')  # Enable foreign key support
        conn.execute('PRAGMA journal_mode = WAL')  # Use Write-Ahead Logging for better concurrency
        
        cursor = conn.cursor()
        self.create_tables(cursor)
        conn.commit()

        retry_heap = []  # list of (ready_at, attempt_count, command) managed as a min-heap

        while True:
            # Use monotonic time to avoid issues with system clock changes
            now = time.monotonic()

            # Process any commands that are ready to be retried
            while retry_heap and retry_heap[0][0] <= now:
                _, attempt_count, cmd = heapq.heappop(retry_heap)
                self.handle_command(conn, cursor, cmd, retry_heap, attempt_count)

            # Wait until one of the retry commands is ready or a new command is enqueued
            # This ensures that, if no new tasks are enqueued, the retry commands will be processed as soon 
            # as they are ready.
            timeout = (retry_heap[0][0] - now) if retry_heap else None
            try:
                cmd = self._queue.get(timeout=timeout)
            except queue.Empty:
                continue
            
            # If the command is a stop command, break the loop and exit
            if cmd.type == 'stop':
                break
            
            self.handle_command(conn, cursor, cmd, retry_heap, 0)

        conn.close()
        self.stopped.set()

    def handle_command(self, conn, cursor, cmd, retry_heap, attempt):
        try:
            self.execute(cursor, cmd)
            conn.commit()
        except sqlite3.OperationalError as e:
            attempt += 1
            conn.rollback()

            # If the maximum retry attempts have been reached, log the error and move the command to the dead letter.
            if attempt >= self.max_retry_attempts:
                logger.error("Max retry attempts reached for command %s: %s", cmd.type, e)
                self.dead_letter(cmd, conn, cursor, e)
                return
            
            # Schedule the command for retry after a backoff period
            ready_at = time.monotonic() + self.retry_backoff * attempt
            heapq.heappush(retry_heap, (ready_at, attempt, cmd))
            logger.info("Command %s scheduled for retry (attempt %d): %s", cmd.type, attempt, e)
        except Exception as e:
            conn.rollback()
            self.dead_letter(cmd, conn, cursor, e)

    def dead_letter(self, cmd: Command, conn, cursor, error: Exception):
        cursor.execute(
            'INSERT INTO dead_letter (cmd_type, cmd_payload, error_message) VALUES (?, ?, ?)',
            (cmd.type, str(cmd.payload), str(error))
        )
        conn.commit()

    def execute(self, cursor, cmd: Command):
        p = cmd.payload
        if cmd.type == 'start_clip':
            cursor.execute(
                'INSERT INTO clips (id, started_at, file_path, trigger) VALUES (?, ?, ?, ?)',
                (p['clip_id'], p['start_time'], p['file_path'], p['trigger'])
            )
        elif cmd.type == 'end_clip':
            cursor.execute("UPDATE clips SET ended_at = ? WHERE id = ?", (p['ended_at'], p['clip_id']))
        elif cmd.type == 'insert_detection':
            cursor.execute(
                '''INSERT INTO detections
                (clip_id, timestamp, class_name, confidence, bbox_x, bbox_y, bbox_width, bbox_height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (p['clip_id'], p['timestamp'], p['class_name'], p['confidence'],
                p['bbox_x'], p['bbox_y'], p['bbox_width'], p['bbox_height'])
            )
        else:
            raise ValueError(f'Unknown command type: {cmd.type}')

    def create_tables(self, cursor):
        cursor.execute('''CREATE TABLE IF NOT EXISTS clips (
                            id TEXT PRIMARY KEY,
                            started_at REAL NOT NULL,
                            ended_at REAL,
                            file_path TEXT NOT NULL,
                            trigger TEXT NOT NULL DEFAULT 'continuous'
                        )''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS detections (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            clip_id TEXT NOT NULL REFERENCES clips (id) ON DELETE CASCADE,
                            timestamp REAL NOT NULL,
                            class_name TEXT NOT NULL,
                            confidence REAL NOT NULL,
                            bbox_x INTEGER NOT NULL,
                            bbox_y INTEGER NOT NULL,
                            bbox_width INTEGER NOT NULL,
                            bbox_height INTEGER NOT NULL
                        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS dead_letter (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            cmd_type TEXT NOT NULL,
                            cmd_payload TEXT NOT NULL,
                            error_message TEXT NOT NULL,
                            timestamp DATETIME NOT NULL DEFAULT (datetime('now'))
                        )''')
        
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_detections_clip_id_ts ON detections (clip_id, timestamp)''')

        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_clips_id ON clips (id)''')