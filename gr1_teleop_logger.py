import subprocess
import sys
import threading
import numpy as np
import time
import zmq
import msgpack
import msgpack_numpy as m
import os
import shutil
from PIL import Image
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Label, Button, Static, Select
from textual.containers import Vertical, Horizontal, ScrollableContainer
from gr1_config import COMPACT_WIRE_JOINTS

import multiprocessing.resource_tracker
# Hack: macOS Python 3.12 has a bug where `multiprocessing.spawn` crashes if there are 
# complex file descriptors open (like Textual's UI pipes and zmq). Since huggingface `datasets` 
# lazily triggers the resource_tracker when saving the episode later, we pre-warm it here
# at script start when the file descriptors are still clean.
multiprocessing.resource_tracker.ensure_running()

# Attempt to import LeRobot (Will fail if lerobot isn't installed in the env)
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False


class JointControl(Static):
    """A widget for controlling a single robot joint using an Input field."""

    def __init__(self, joint_name: str, index: int, initial_value: float, on_change_callback):
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
                val = max(-1.0, min(1.0, val))
                self.on_change_callback(self.index, val)
                event.input.styles.color = "#9ece6a"
            except ValueError:
                event.input.styles.color = "#f7768e"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == f"clear_{self.index}":
            input_widget = self.query_one(f"#input_{self.index}", Input)
            input_widget.value = "0.00"
            self.on_change_callback(self.index, 0.0)


class GR1TeleopLoggerApp(App):
    """A Textual app to teleoperate AND log data to LeRobot format."""

    CSS = """
    Screen { background: #1a1b26; }
    #selection-pane { height: auto; padding: 0 2 0 2; background: #1f2335; border-bottom: double #7aa2f7; }
    .selection-label { width: auto; height: 1; margin: 1 1 0 0; color: #bb9af7; text-style: bold; }
    .instruction-label { width: auto; height: 1; margin: 0 1 0 0; color: #bb9af7; text-style: bold; }
    .selection-row { height: 3; margin: 0; padding: 0; align: left middle; }
    #joint-select { width: 40; }
    #instruction-input { height: 1; border: none; background: #1f2335; color: #c0caf5; margin-left: 2; width: 40; }
    #selection-pane Button { height: 1; min-width: 8; border: none; background: #bb9af7; color: #1a1b26; text-style: bold; margin: 1 0 0 1; }
    #joint-list { background: #1a1b26; padding: 1 2; min-height: 10; scrollbar-gutter: stable; }
    JointControl { height: 3; margin: 0 0 1 0; }
    .joint-row { height: 3; padding: 0 2; background: #24283b; border: solid #414868; align: left middle; }
    .joint-label { width: 40; content-align: left middle; color: #7aa2f7; text-style: bold; }
    .joint-input, Input { height: 1; background: #1f2335; border: none; color: #c0caf5; text-style: bold; }
    .joint-input { width: 15; }
    Select { height: 3; border: none; background: #1f2335; color: #c0caf5; }
    .btn-clear { margin-left: 2; min-width: 10; height: 1; background: #e3892d; color: #ffffff; border: none; text-style: bold; }
    #footer-controls { height: 4; dock: bottom; background: #1f2335; border-top: double #7aa2f7; padding: 0 2; align: left top; }
    #footer-controls Button { height: 1; margin: 0; border: none; text-style: bold; }
    .btn-submit { width: 20; background: #9ece6a; color: #1a1b26; }
    .btn-record { width: 20; background: #f7768e; color: #ffffff; margin-left: 2; margin-right: 2;}
    #clear_all_btn { background: #e0af68; color: #1a1b26; margin-left: 2; }
    .status-label { color: #565f89; text-style: italic; }
    .footer-spacer { width: 2; }
    """

    BINDINGS = [
        ("h", "home_all", "Home All"),
        ("escape", "quit", "Quit"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        m.patch() # Enable msgpack to serialize numpy arrays directly

        # ZMQ Setup (Publishes Actions)
        self.context = zmq.Context()
        self.socket_pub = self.context.socket(zmq.PUB)
        self.socket_pub.connect("tcp://127.0.0.1:5556")
        
        # ZMQ Setup (Subscribes to Observations)
        self.socket_sub = self.context.socket(zmq.SUB)
        self.socket_sub.connect("tcp://127.0.0.1:5557")
        self.socket_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # We fill missing actions with 0 instead of NaN to avoid dataset issues
        self.target_buffer = np.zeros(32, dtype=np.float32)
        self.staging_buffer = np.zeros(32, dtype=np.float32)

        self.active_joints = set()
        self.recording = False
        self.dataset = None
        self.recording_thread = None
        
        # Typically visual observations run at a fixed fps (e.g. 10 or 30Hz)
        self.fps = 10

    def on_mount(self) -> None:
        self.title = "GR1 Teleop & Data Logger (Real Images Edition)"
        
        if not LEROBOT_AVAILABLE:
            self.notify("LeRobot library not found! Recording will log dummy prints.", title="Missing Dependency", severity="warning")

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="selection-pane"):
            with Horizontal(classes="selection-row"):
                yield Label("Task Instruction:", classes="instruction-label")
                yield Input(placeholder="e.g. Pick up the red cube", id="instruction-input")
            
            with Horizontal(classes="selection-row"):
                yield Label("Select Joint:", classes="selection-label")
                yield Select(
                    [(name, i) for i, name in enumerate(COMPACT_WIRE_JOINTS)],
                    id="joint-select",
                    prompt="Pick a joint...",
                )
                yield Button("Add", id="add_select_btn", variant="primary", classes="btn-add")

        with ScrollableContainer(id="joint-list"):
            pass

        with Horizontal(id="footer-controls"):
            yield Button("Submit Request", id="submit_btn", variant="success", classes="btn-submit")
            yield Static(classes="footer-spacer")
            yield Button("Start Recording", id="record_btn", variant="error", classes="btn-record")
            yield Static(classes="footer-spacer")
            yield Button("Clear All", id="clear_all_btn", variant="warning")
            yield Static(expand=True)
            with Vertical():
                yield Label(f"FPS Logger: {self.fps} Hz", classes="status-label")
                yield Label("Press 'H' to reset all to zero", classes="status-label")
        yield Footer()

    def update_target(self, index: int, value: float) -> None:
        self.staging_buffer[index] = value

    def send_command(self) -> None:
        payload = {"target": self.target_buffer.tolist()}
        self.socket_pub.send(msgpack.packb(payload, use_bin_type=True))

    def action_home_all(self) -> None:
        self.target_buffer.fill(0.0)
        self.staging_buffer.fill(0.0)
        for control in self.query(JointControl):
            input_widget = control.query_one(Input)
            input_widget.value = "0.00"

    async def rebuild_joint_list(self) -> None:
        container = self.query_one("#joint-list", ScrollableContainer)
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
            self.target_buffer = np.copy(self.staging_buffer)
            self.send_command()
            event.button.label = "Sent!"
            def reset_label():
                event.button.label = "Submit Request"
            threading.Timer(1.0, reset_label).start()

        elif event.button.id == "clear_all_btn":
            self.active_joints.clear()
            await self.rebuild_joint_list()
            self.target_buffer.fill(0.0)
            self.staging_buffer.fill(0.0)
            
        elif event.button.id == "record_btn":
            if not self.recording:
                self.start_recording()
                event.button.label = "Stop Recording"
                event.button.styles.background = "#e0af68" # Orange
            else:
                self.stop_recording()
                event.button.label = "Start Recording"
                event.button.styles.background = "#f7768e" # Red

    # ------------------ LEROBOT DATASET INTEGRATION ------------------
    def start_recording(self):
        self.recording = True
        
        # 1. Initialize LeRobot Dataset Structure
        if LEROBOT_AVAILABLE:
            repo_id = "gr1_teleop_demo"
            dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", repo_id))
            # Check if dataset already exists to resume it
            resume = os.path.exists(dataset_dir)
            
            features = {
                "observation.images.world_left": {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.images.world_right": {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.images.world_center": {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.images.world_top": {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (32,),         # The proprioception vector (current joint poses)
                    "names": ["joints"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (32,),         # The desired actions to reach
                    "names": ["joints"],
                }
            }
            
            # Retrieve the instruction to save in tasks.jsonl manually
            self.current_task_instruction = self.query_one("#instruction-input", Input).value or "Do the task"
            
            # Either create a new dataset or load the existing one to append to it
            if not resume:
                self.dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=self.fps,
                    root=dataset_dir,
                    features=features,
                    use_videos=True,
                    image_writer_processes=0,  # CRITICAL: Fixes macOS multiprocessing FD error
                    image_writer_threads=4,    # Fall back to threading
                    batch_encoding_size=1      # Encode frame synchronously
                )
            else:
                self.dataset = LeRobotDataset(
                    repo_id=repo_id,
                    root=dataset_dir,
                )
            
        # 2. Start a background loop to sample data at target FPS
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()

    def stop_recording(self):
        self.recording = False
        if self.recording_thread:
            self.recording_thread.join()
            
        if self.dataset is not None:
            # Tell the dataset this episode is over and save the parquet buffers
            self.dataset.save_episode(parallel_encoding=False)
            self.dataset.finalize()
            self.notify("Dataset episode saved locally!", title="LeRobot", severity="information")
            self.dataset = None

    def _recording_loop(self):
        """
        Continuous loop capturing state and action at a fixed frequency (10Hz).
        Reads real observations from the ZMQ simple_simulation_logger.
        """
        # We need a small cache of the latest image in case ZMQ is lagging
        latest_obs = None

        while self.recording:
            loop_start = time.time()
            
            # --- GET CURRENT STATE (Proprioception + Vision from Simulation) ---
            # Try to pull the latest frame from ZMQ without blocking forever
            try:
                # We pull all pending messages to get the *freshest* frame
                while True:
                    msg = self.socket_sub.recv(flags=zmq.NOBLOCK)
                    latest_obs = msgpack.unpackb(msg, raw=False)
            except zmq.Again:
                pass # No more messages in queue

            if latest_obs is None:
                # No data yet, skip recording this tick or use dummy
                time.sleep(0.01)
                continue

            current_state = np.array(latest_obs["state"], dtype=np.float32)
            img_top = Image.fromarray(latest_obs["world_top"])
            img_left = Image.fromarray(latest_obs["world_left"])
            img_right = Image.fromarray(latest_obs["world_right"])
            img_center = Image.fromarray(latest_obs["world_center"])
            
            # --- GET CURRENT ACTION ---
            # The control being currently commanded by the TUI
            current_action = np.copy(self.target_buffer)
            
            # --- STORE IN DATASET ---
            if LEROBOT_AVAILABLE and self.dataset is not None:
                frame_data = {
                    "observation.images.world_left": img_left,
                    "observation.images.world_right": img_right,
                    "observation.images.world_center": img_center,
                    "observation.images.world_top": img_top,
                    "observation.state": current_state,
                    "action": current_action,
                    "task": getattr(self, "current_task_instruction", "Do the task")
                }
                # This automatically stores to an episode buffer and writes standard RLDS/Parquet formatting
                self.dataset.add_frame(frame_data)
                
            # Sleep to maintain the FPS loop strictly
            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / self.fps) - elapsed)
            time.sleep(sleep_time)

    def action_quit(self) -> None:
        if self.recording:
            self.stop_recording()
        self.exit()


if __name__ == "__main__":
    app = GR1TeleopLoggerApp()
    app.run()
