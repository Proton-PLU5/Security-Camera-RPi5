import heapq
from multiprocessing import Event, Process, Queue
import os
import sqlite3
from dataclasses import dataclass
import time
import logging
import queue

@dataclass
class Task:
    type: str
    payload: dict

logger = logging.getLogger(__name__)

"""
    A class that assists in creating tasks for the StorageProcess.
"""
class TaskFactory:
    def __init__(self):
        pass

    def start_clip(self, clip_id : str, file_path: str, start_time: float, trigger: str = 'continuous') -> Task:
        command = Task('start_clip', {'clip_id': clip_id, 'file_path': file_path, 'start_time': start_time, 'trigger': trigger})
        return command

    def end_clip(self, clip_id: str, ended_at: float, trigger: str) -> Task:
        command = Task('end_clip', {'clip_id': clip_id, 'ended_at': ended_at, 'trigger': trigger})
        return command

    def insert_detection(self, clip_id: str, timestamp: float, class_name: str, confidence: float,
                          bbox_x: int, bbox_y: int, bbox_width: int, bbox_height: int) -> Task:
        return Task('insert_detection', dict(
            clip_id=clip_id, timestamp=timestamp, class_name=class_name, confidence=confidence,
            bbox_x=bbox_x, bbox_y=bbox_y, bbox_width=bbox_width, bbox_height=bbox_height
        ))

    def delete_clip(self, clip_id: str) -> Task:
        return Task('delete_clip', dict(clip_id=clip_id))

    def insert_recognition(self, clip_id: str, timestamp: float, name: str) -> Task:
        return Task('insert_recognition', dict(
            clip_id=clip_id, timestamp=timestamp, name=name
        ))

class StorageProcess(Process):
    def __init__(self, 
                 task_queue: "Queue[Task]", 
                 stop_event: Event,  # type: ignore
                 db_path: str = 'storage.db', 
                 max_retry_attempts: int = 3, 
                 retry_backoff: float = 0.1,
                 max_clips: int = 1000):
        super().__init__(daemon=True)
        self.db_path = db_path
        self.stop_event = stop_event
        self.task_queue = task_queue
        self.max_retry_attempts = max_retry_attempts
        self.retry_backoff = retry_backoff
        self.max_number_of_clips = max_clips  # Set a maximum number of clips to store
        self.current_clip_count = 0  # Initialize the current clip count

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
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS recognitions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            clip_id TEXT NOT NULL REFERENCES clips (id) ON DELETE CASCADE,
                            timestamp REAL NOT NULL,
                            name TEXT NOT NULL
                        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS devices (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            pairing_secret TEXT NOT NULL UNIQUE,
                            access_level INTEGER NOT NULL DEFAULT 0,
                            created_at DATETIME NOT NULL DEFAULT (datetime('now'))
                        )''')
        
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_detections_clip_id_ts ON detections (clip_id, timestamp)''')

        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_clips_id ON clips (id)''')

    def handle_command(self, conn, cursor, cmd, retry_heap, attempt):
        try:
            cursor.execute("BEGIN")
            self.execute(cursor, cmd)
            conn.commit()
        except sqlite3.OperationalError as e:
            attempt += 1
            conn.rollback()  # Rolls back the failed command execution

            # If max retries are reached, log and record to dead letter atomically
            if attempt >= self.max_retry_attempts:
                logger.error("Max retry attempts reached for command %s: %s", cmd.type, e)
                try:
                    cursor.execute("BEGIN")
                    self.dead_letter(cmd, cursor, e)
                    conn.commit()
                except Exception as inner_e:
                    conn.rollback()
                    logger.error(f"Failed to write to dead_letter table: {inner_e}", exc_info=True)
                return
            
            # Schedule the command for retry after a backoff period
            ready_at = time.monotonic() + self.retry_backoff * attempt
            heapq.heappush(retry_heap, (ready_at, attempt, cmd))
            logger.info("Command %s scheduled for retry (attempt %d): %s", cmd.type, attempt, e)
            
        except Exception as e:
            conn.rollback()
            try:
                cursor.execute("BEGIN")
                self.dead_letter(cmd, cursor, e)
                conn.commit()
            except Exception as inner_e:
                conn.rollback()
                logger.error(f"Failed to write to dead_letter table: {inner_e}", exc_info=True)

    def dead_letter(self, cmd: Task, cursor, error: Exception):
        # Removed 'conn' parameter since commit is now handled safely outside
        cursor.execute(
            'INSERT INTO dead_letter (cmd_type, cmd_payload, error_message) VALUES (?, ?, ?)',
            (cmd.type, str(cmd.payload), str(error))
        )

    def execute(self, cursor, cmd: Task):
        p = cmd.payload
        if cmd.type == 'start_clip':
            cursor.execute(
                'INSERT INTO clips (id, started_at, file_path, trigger) VALUES (?, ?, ?, ?)',
                (p['clip_id'], p['start_time'], p['file_path'], p['trigger'])
            )
            self.current_clip_count += 1  # Increment the current clip count after insertion
        elif cmd.type == 'end_clip':
            cursor.execute("UPDATE clips SET ended_at = ?, trigger = ? WHERE id = ?", (p['ended_at'], p['trigger'], p['clip_id']))

            # If the number of clips exceeds the maximum allowed, delete the oldest clip(s)
            if self.current_clip_count < self.max_number_of_clips:
                return  # No need to delete clips if we are under the limit

            num_to_delete = self.current_clip_count - self.max_number_of_clips + 1
            logger.info(f"Clips count exceeded limit. Deleting {num_to_delete} oldest clip(s).")

            cursor.execute("SELECT id, file_path FROM clips ORDER BY started_at ASC LIMIT ?", (num_to_delete,))
            rows = cursor.fetchall()
            if not rows:
                return

            for row in rows:
                oldest_id, oldest_path = row

                cursor.execute("DELETE FROM clips WHERE id = ?", (oldest_id,))

                if oldest_path and os.path.exists(oldest_path):
                    os.remove(oldest_path)

        elif cmd.type == 'insert_detection':
            cursor.execute(
                '''INSERT INTO detections
                (clip_id, timestamp, class_name, confidence, bbox_x, bbox_y, bbox_width, bbox_height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (p['clip_id'], p['timestamp'], p['class_name'], p['confidence'],
                p['bbox_x'], p['bbox_y'], p['bbox_width'], p['bbox_height'])
            )
        elif cmd.type == 'insert_recognition':
            cursor.execute(
                '''INSERT INTO recognitions
                (clip_id, timestamp, name)
                VALUES (?, ?, ?)''',
                (p['clip_id'], p['timestamp'], p['name'])
            )
        elif cmd.type == 'delete_clip':
            cursor.execute("DELETE FROM clips WHERE id = ?", (p['clip_id'],))
            self.current_clip_count -= 1  # Decrement the current clip count after deletion
        else:
            raise ValueError(f'Unknown command type: {cmd.type}')

    def run(self):
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute('PRAGMA foreign_keys = ON')  # Enable foreign key support
        conn.execute('PRAGMA journal_mode = WAL')  # Use Write-Ahead Logging for better concurrency
        
        cursor = conn.cursor()
        self.create_tables(cursor)
        conn.commit()

        # Initialize the number of clips currently in the database
        cursor.execute("SELECT COUNT(*) FROM clips")
        self.current_clip_count = cursor.fetchone()[0]

        retry_heap = []  # list of (ready_at, attempt_count, command) managed as a min-heap
        while not self.stop_event.is_set():
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
                cmd = self.task_queue.get(timeout=timeout)
            except queue.Empty:
                continue  # No new command, loop back to check retry_heap
            except Exception as e:
                logger.error(f"Error occurred while fetching command from queue: {e}", exc_info=True)
                continue
            
            # If the command is a stop command, break the loop and exit
            if cmd.type == 'stop':
                break
            
            self.handle_command(conn, cursor, cmd, retry_heap, 0)
