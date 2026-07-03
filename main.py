import logging

import numpy as np
import ncnn
from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.status import status
from rich.table import Table, table
from rich.logging import RichHandler
from data.storage import StorageThread
from capture.capture import CaptureThread
from capture.mailbox import MailBox
from capture.detection.detect import DetectionThread, DetectionStore
from stream.frame_buffer import FrameBuffer
from stream.webrtc_stream import StreamThread
from ultralytics import YOLO
from data.metrics import metrics
_orig_load_param = ncnn.Net.load_param # type: ignore

console = Console()

# NCNN THREAD LIMITING
def _load_param_with_thread_limit(self, path):
    self.opt.num_threads = 2  # set before weights get packed, not after
    return _orig_load_param(self, path)

ncnn.Net.load_param = _load_param_with_thread_limit # type: ignore

def build_commands(threads: dict) -> dict:
    def handle_metrics():
        metrics.report()
    
    def handle_help():
        table = Table(title="Commands")

        table.add_column("Command")
        table.add_column("Description")

        table.add_row("status", "Show thread status")
        table.add_row("metrics", "Display runtime metrics")
        table.add_row("help", "Show this help menu")
        table.add_row("exit", "Shutdown application")

        console.print(table)

    def handle_status():
        return {
            "capture_thread_alive": threads["capture"].is_alive(),
            "detection_thread_alive": threads["detection"].is_alive(),
            "stream_thread_alive": threads["stream"].is_alive(),
            "storage_thread_alive": threads["storage"].is_alive(),
        }

    return {
        "metrics": handle_metrics,
        "status": handle_status,
        "help": handle_help,
    }


def main():
    with patch_stdout():
        logging.basicConfig(
            level=logging.INFO,
            format="[%(threadName)s] %(name)s: %(message)s",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        
        model = YOLO("./capture/detection/yolo26s_ncnn_model")  # Load the YOLO model

        storage_thread = StorageThread()
        storage_thread.start()

        detection_mailbox = MailBox()
        detection_store = DetectionStore()

        stream_buffer = FrameBuffer()

        capture_thread = CaptureThread(
            detection_mailbox=detection_mailbox, 
            stream_buffer=stream_buffer, 
            clip_dir="./clips", 
            clip_length=10, 
            storage_thread=storage_thread)
        
        capture_thread.start()

        detection_thread = DetectionThread(
            mailbox=detection_mailbox, 
            detection_store=detection_store, 
            model=model, 
            storage_thread=storage_thread)
        detection_thread.start()

        stream_thread = StreamThread(
            buffer=stream_buffer, 
            detection_store=detection_store)
        stream_thread.start()

        threads = {
            "capture": capture_thread,
            "detection": detection_thread,
            "stream": stream_thread,
            "storage": storage_thread,
        }
        commands = build_commands(threads)

        command_completer = WordCompleter(
            [
                "help",
                "metrics",
                "status",
                "exit",
            ],
            ignore_case=True,
        )

        session = PromptSession(
            HTML(
                "<ansicyan><b>vision-edge</b></ansicyan> "
                "<ansigreen>❯</ansigreen> "
            ),
            completer=command_completer,
            complete_while_typing=True,
            history=FileHistory(".vision_history")
        )
        
        console.print()

        console.print(
            Panel.fit(
                "[bold cyan]Vision Edge[/bold cyan]\n\n"
                "[green]✓[/green] Capture Thread Running\n"
                "[green]✓[/green] Detection Thread Running\n"
                "[green]✓[/green] Stream Thread Running\n"
                "[green]✓[/green] Storage Thread Running",
                title="System Ready",
                border_style="cyan",
            )
        )

        console.print(
            "[dim]Type 'help' for available commands.[/dim]\n"
        )

        while True:
            try:
                user_input = session.prompt()
            except (EOFError, KeyboardInterrupt):
                break
 
            cmd = user_input.strip().lower()
            if not cmd:
                continue
            if cmd == "exit":
                break
            elif cmd == "status":
                status = commands["status"]()

                table = Table(title="System Status")

                table.add_column("Component")
                table.add_column("State")

                for name, alive in status.items():
                    state = "[green]Running[/green]" if alive else "[red]Stopped[/red]"

                    pretty_name = (
                        name.replace("_thread_alive", "")
                            .replace("_", " ")
                            .title()
                    )

                    table.add_row(pretty_name, state)

                console.print(table)
            elif cmd in commands:
                commands[cmd]()
            else:
                console.print(
                    f"[red]Unknown command:[/red] {cmd!r}\n"
                    "[dim]Type 'help' for available commands.[/dim]"
                )
        
        print("Shutting down threads...")
        capture_thread.stop()
        capture_thread.join()
        detection_thread.stop()
        detection_thread.join()
        stream_thread.stop()
        stream_thread.join()
        storage_thread.stop()
        storage_thread.join()

if __name__ == "__main__":
    main()