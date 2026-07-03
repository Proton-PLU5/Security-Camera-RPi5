import logging

import pyfiglet
import psutil
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import RichLog, Static, Input


class TextualLogHandler(logging.Handler):
    """
    Routes stdlib `logging` records into the app's log widget instead of
    stdout/stderr. A full-screen Textual app owns the whole terminal, so
    writing directly to real stdout/stderr would corrupt rendering -
    call_from_thread() is Textual's sanctioned way to safely touch
    widgets from a non-UI thread (CaptureThread, DetectionThread, etc.
    all call logger.info() from their own threads).

    Note: call_from_thread() only works once the app's event loop is
    actually running (i.e. after app.run() has started). If a thread
    logs something in the brief window between thread.start() and the
    app finishing its first mount, that message is silently dropped
    rather than crashing - in practice this window is a few
    milliseconds and rarely matters, but it's why emit() swallows
    exceptions instead of raising.
    """

    def __init__(self, app: "VisionApp"):
        super().__init__()
        self.app = app

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.app.call_from_thread(self.app.log_message, msg)
        except Exception:
            pass


class VisionApp(App):
    """Vision System TUI (Textual-based)"""

    CSS = """
    Screen {
        layout: vertical;
    }

    #header {
        height: 5;
    }

    #log {
        height: 1fr;
        border: solid #444444;
    }

    #input {
        height: 3;
        border-top: solid #333333;
    }
    """

    def __init__(self, threads: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        # passed in from main.py, which owns creating/starting the
        # actual pipeline threads - the app just reads from this dict
        # for the status command and the header's live status label.
        self.threads: dict = threads or {}

    # -----------------------------
    # LIFECYCLE
    # -----------------------------
    def compose(self) -> ComposeResult:
        yield Static("", id="header")
        yield RichLog(id="log", markup=False, wrap=True, auto_scroll=True)
        yield Input(placeholder="vision-system \u276f ", id="input")

    def on_mount(self):
        self.set_interval(1, self.update_header)

        ascii_title = pyfiglet.figlet_format("VISION SYSTEM", font="mono12")
        title = Text(ascii_title, style="bold #D97757", justify="center")
        panel = Panel(
            title,
            subtitle="Author: Proton-PLU5",
            border_style="#D97757",
            padding=(1, 1),
        )
        self.query_one("#log", RichLog).write(panel)
        self.log_message("Vision system started \u2714")

    # -----------------------------
    # HEADER PANEL
    # -----------------------------
    def update_header(self):
        cpu_percent = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent

        if self.threads:
            all_alive = all(t.is_alive() for t in self.threads.values())
            status_label, status_style = ("RUNNING", "green") if all_alive else ("DEGRADED", "red")
        else:
            status_label, status_style = ("STARTING", "yellow")

        header_text = Text.assemble(
            ("VISION SYSTEM", "bold #D97757"),
            ("   |   ", "dim"),
            (f"CPU: {cpu_percent:.0f}%", "cyan"),
            ("   |   ", "dim"),
            (f"RAM: {ram:.1f}%", "magenta"),
            ("   |   ", "dim"),
            (f"STATUS: {status_label}", status_style),
            justify="center",
        )

        panel = Panel(header_text, border_style="#D97757", padding=(1, 1))
        self.query_one("#header", Static).update(panel)

    # -----------------------------
    # INPUT HANDLING
    # -----------------------------
    def on_input_submitted(self, event: Input.Submitted):
        cmd = event.value.strip()
        event.input.value = ""
        log = self.query_one("#log", RichLog)

        if not cmd:
            return

        log.write(Text(f"> {cmd}", style="bold cyan"))

        if cmd == "exit":
            self.exit()
        elif cmd == "status":
            self._handle_status(log)
        elif cmd == "metrics":
            from data.metrics import metrics
            metrics.report()  # routes through logging -> TextualLogHandler -> log_message
        elif cmd == "reset_data":
            # Deletes the database, clips, and resets metrics.
            from data.metrics import metrics
            import os
            import shutil
            if os.path.exists("storage.db"):
                os.remove("storage.db")
            if os.path.exists("clips"):
                shutil.rmtree("clips")
            metrics.reset()
            log.write(Text("Data reset: storage.db deleted, clips removed, metrics reset.", style="red"))
        elif cmd == "help":
            self._handle_help(log)
        else:
            log.write(Text(f"Unknown command: {cmd!r}", style="red"))

    def _handle_status(self, log: RichLog):
        table = Table(title="System Status")
        table.add_column("Component")
        table.add_column("State")
        if not self.threads:
            log.write(Text("No threads registered yet.", style="yellow"))
            return
        for name, t in self.threads.items():
            state = Text("Running", style="green") if t.is_alive() else Text("Stopped", style="red")
            table.add_row(name.title(), state)
        log.write(table)

    def _handle_help(self, log: RichLog):
        table = Table(title="Commands")
        table.add_column("Command")
        table.add_column("Description")
        table.add_row("status", "Show thread status")
        table.add_row("metrics", "Display runtime metrics")
        table.add_row("help", "Show this help menu")
        table.add_row("exit", "Shutdown application")
        log.write(table)

    # -----------------------------
    # THREAD-SAFE LOGGING API
    # -----------------------------
    def log_message(self, message):
        """
        Call from the UI thread directly, or from any other thread via:
            app.call_from_thread(app.log_message, "msg")
        """
        self.query_one("#log", RichLog).write(message)