from multiprocessing import Event, Queue

from mp.capture import CaptureProcess, CaptureBuffer
from mp.storage import StorageProcess
from mp.stream import StreamProcess

def main():
    storage_task_queue = Queue()
    stop_event = Event()
    lowres_size = (960, 540)  # Width, Height
    video_size = (1920, 1080)  # Width, Height
    capture_buffer = CaptureBuffer(shape = (lowres_size[1], lowres_size[0], 3)) # Height, Width, Channels

    capture_process = CaptureProcess(
        storage_task_queue=storage_task_queue,
        stop_event=stop_event,
        db_path='storage.db',
        capture_buffer=capture_buffer,
        lowres_size=lowres_size,
        video_size=video_size
    )

    storage_process = StorageProcess(storage_task_queue, stop_event, db_path='storage.db')
    stream_process = StreamProcess(
        buffer=capture_buffer,
        storage_db_path='storage.db',
        stop_event=stop_event)

    storage_process.start()
    capture_process.start()
    stream_process.start()

    while True:
        print(capture_process.is_alive(), storage_process.is_alive(), stream_process.is_alive())

    stop_event.set()  # Signal the storage process to stop
    storage_process.join()  # Wait for the storage process to finish
    capture_process.join()  # Wait for the capture process to finish
    stream_process.join()  # Wait for the stream process to finish
if __name__ == "__main__":
    main()