import subprocess
import sys
import threading
import numpy as np
import time
import zmq
import msgpack
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Label, Button, Static, Select
from textual.containers import Vertical, Horizontal, ScrollableContainer, Grid
from gr1_config import COMPACT_WIRE_JOINTS


class JointControl(Static):
    """A widget for controlling a single robot joint using an Input field."""

    def __init__(
        self, joint_name: str, index: int, initial_value: float, on_change_callback
    ):
        super().__init__()
        self.joint_name = joint_name
        self.index = index
        self.initial_value = initial_value
        self.on_change_callback = on_change_callback

    def compose(self) -> ComposeResult:
        with Horizontal(classes="joint-row"):
            yield Label(
                f"[{self.index:02}] {self.joint_name:<25}", classes="joint-label"
            )
            # Format initial value, handling NaN as 0.00
            val_str = (
                f"{self.initial_value:.2f}"
                if not np.isnan(self.initial_value)
                else "0.00"
            )
            yield Input(
                placeholder="0.00",
                value=val_str,
                id=f"input_{self.index}",
                classes="joint-input",
            )
            yield Button("Clear", id=f"clear_{self.index}", classes="btn-clear")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == f"input_{self.index}":
            try:
                val = float(event.value)
                # Clamp for safety
                val = max(-1.0, min(1.0, val))
                self.on_change_callback(self.index, val)
                event.input.styles.color = "#9ece6a"  # Green for valid
            except ValueError:
                event.input.styles.color = "#f7768e"  # Red for invalid

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == f"clear_{self.index}":
            input_widget = self.query_one(f"#input_{self.index}", Input)
            input_widget.value = "0.00"
            self.on_change_callback(self.index, 0.0)


class GR1TeleopApp(App):
    """A Textual app to teleoperate the GR1 robot via a separate Simulation Process."""

    CSS = """
    Screen {
        background: #1a1b26;
    }
    #selection-pane {
        height: auto;
        padding: 1 2 0 2;
        background: #1f2335;
        border-bottom: double #7aa2f7;
    }
    .selection-label {
        width: auto;
        height: 1;
        margin: 1 1 0 0;
        color: #bb9af7;
        text-style: bold;
    }
    .selection-row {
        height: 3;
        margin: 0 0 1 0;
        padding: 0;
        align: left middle;
    }
    #joint-select {
        width: 40;
    }
    #selection-pane Button {
        height: 1;
        min-width: 8;
        border: none;
        background: #bb9af7;
        color: #1a1b26;
        text-style: bold;
        margin: 1 0 0 1;
    }
    #joint-list {
        background: #1a1b26;
        padding: 1 2;
        min-height: 10;
        scrollbar-gutter: stable;
    }
    JointControl {
        height: 3;
        margin: 0 0 1 0;
    }
    .joint-row {
        height: 3;
        padding: 0 2;
        background: #24283b;
        border: solid #414868;
        align: left middle;
    }
    .joint-label {
        width: 40;
        content-align: left middle;
        color: #7aa2f7;
        text-style: bold;
    }
    .joint-input, Input {
        height: 1;
        background: #1f2335;
        border: none;
        color: #c0caf5;
        text-style: bold;
    }
    .joint-input {
        width: 15;
    }
    Select {
        height: 3;
        border: none;
        background: #1f2335;
        color: #c0caf5;
    }
    .btn-clear {
        margin-left: 2;
        min-width: 10;
        height: 1;
        background: #e3892d;
        color: #ffffff;
        border: none;
        text-style: bold;
    }
    #footer-controls {
        height: 4;
        dock: bottom;
        background: #1f2335;
        border-top: double #7aa2f7;
        padding: 0 2;
        align: left top;
    }
    #footer-controls Button {
        height: 1;
        margin: 0;
        border: none;
        text-style: bold;
    }
    .btn-submit {
        width: 20;
        background: #9ece6a;
        color: #1a1b26;
    }
    #clear_all_btn {
        background: #f7768e;
        color: #ffffff;
    }
    .status-label {
        color: #565f89;
        text-style: italic;
    }
    .footer-spacer {
        width: 2;
    }
    """

    BINDINGS = [
        ("h", "home_all", "Home All"),
        ("escape", "quit", "Quit"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        # 1. ZMQ Setup (Publisher)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.target_buffer = np.full(32, np.nan, dtype=np.float32)
        self.staging_buffer = np.full(32, np.nan, dtype=np.float32)

        # 3. State Management
        self.active_joints = set()  # Set of indices
        self.pending_updates = False

    def on_mount(self) -> None:
        self.title = "GR1 Advanced Teleop Dashboard"
        self.socket.connect("tcp://127.0.0.1:5556")

    def send_command(self) -> None:
        """Publishes the current target buffer to the simulation."""
        payload = {"target": self.target_buffer.tolist()}
        self.socket.send(msgpack.packb(payload, use_bin_type=True))

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="selection-pane"):
            with Horizontal(classes="selection-row"):
                yield Label("Select Joint:", classes="selection-label")
                yield Select(
                    [(name, i) for i, name in enumerate(COMPACT_WIRE_JOINTS)],
                    id="joint-select",
                    prompt="Pick a joint...",
                )
                yield Button(
                    "Add", id="add_select_btn", variant="primary", classes="btn-add"
                )

        with ScrollableContainer(id="joint-list"):
            # Initially empty, populated dynamically
            pass

        with Horizontal(id="footer-controls"):
            yield Button(
                "Submit Request",
                id="submit_btn",
                variant="success",
                classes="btn-submit",
            )
            yield Static(classes="footer-spacer")
            yield Button("Clear All", id="clear_all_btn", variant="error")
            yield Static(expand=True)
            with Vertical():
                yield Label(
                    "Backend: Genesis (Metal) | Mode: Multi-Process ZMQ",
                    classes="status-label",
                )
                yield Label("Press 'H' to reset all to zero", classes="status-label")
        yield Footer()

    def update_target(self, index: int, value: float) -> None:
        """Updates the local staging buffer. Scalar values are normalized [-1, 1]."""
        self.staging_buffer[index] = value

    def action_home_all(self) -> None:
        """Resets all joints to 0.0 immediately and clears the active list."""
        self.target_buffer.fill(0.0)
        self.staging_buffer.fill(0.0)
        # Reset all inputs in the UI
        for control in self.query(JointControl):
            input_widget = control.query_one(Input)
            input_widget.value = "0.00"

        # Optional: Clear active joints from view?
        # self.active_joints.clear()
        # self.rebuild_joint_list()

    async def rebuild_joint_list(self) -> None:
        """Updates the ScrollableContainer with active joints."""
        container = self.query_one("#joint-list", ScrollableContainer)
        # Clear existing cleanly
        await container.query("*").remove()

        if self.active_joints:
            widgets = []
            for idx in sorted(list(self.active_joints)):
                name = COMPACT_WIRE_JOINTS[idx]
                val = self.staging_buffer[idx]
                widgets.append(JointControl(name, idx, val, self.update_target))
            await container.mount_all(widgets)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add_select_btn":
            select = self.query_one("#joint-select", Select)
            if select.value is not Select.BLANK:
                self.active_joints.add(int(select.value))
                await self.rebuild_joint_list()

        elif event.button.id == "submit_btn":
            # Commit the staging buffer to the active target buffer
            self.target_buffer = np.copy(self.staging_buffer)
            self.send_command()
            event.button.label = "Sent!"

            def reset_label():
                event.button.label = "Submit Request"

            threading.Timer(1.0, reset_label).start()

        elif event.button.id == "clear_all_btn":
            self.active_joints.clear()
            await self.rebuild_joint_list()
            self.target_buffer.fill(np.nan)
            self.staging_buffer.fill(np.nan)

    def action_quit(self) -> None:
        self.exit()


if __name__ == "__main__":
    app = GR1TeleopApp()
    app.run()
