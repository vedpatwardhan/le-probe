import streamlit as st
import numpy as np
import zmq
import msgpack
import json
import os
from gr1_config import COMPACT_WIRE_JOINTS

st.set_page_config(page_title="GR1 Teleop Dashboard", layout="wide")


# --- Setup ZMQ Socket ---
@st.cache_resource
def get_zmq_socket():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5556")
    return socket


socket = get_zmq_socket()

# --- Initialize Session State ---
if "target_buffer" not in st.session_state:
    st.session_state.target_buffer = np.full(32, np.nan, dtype=np.float32)

if "staging_buffer" not in st.session_state:
    st.session_state.staging_buffer = np.full(32, np.nan, dtype=np.float32)

if "total_episodes" not in st.session_state:
    st.session_state.total_episodes = 0
if "upload_queue" not in st.session_state:
    st.session_state.upload_queue = 0
if "batch_status" not in st.session_state:
    st.session_state.batch_status = 0
if "ik_phase" not in st.session_state:
    st.session_state.ik_phase = None

# --- Load Default Active Joints ---
if "active_joints" not in st.session_state:
    st.session_state.active_joints = set()
    # Load default active joints from IK whitelist
    base_path = os.path.dirname(os.path.abspath(__file__))
    with open(f"{base_path}/ik_joints.txt", "r") as f:
        default_joint_names = [
            line.strip().split("#")[0].strip() for line in f if line.strip()
        ]
        for name in default_joint_names:
            if name in COMPACT_WIRE_JOINTS:
                idx = COMPACT_WIRE_JOINTS.index(name)
                st.session_state.active_joints.add(idx)
                st.session_state[f"input_{idx}"] = 0.0
                st.session_state.staging_buffer[idx] = 0.0


def send_command(payload):
    try:
        socket.send(msgpack.packb(payload, use_bin_type=True))
        resp = socket.recv()
        data = msgpack.unpackb(resp, raw=False)
        st.session_state.upload_queue = data.get("upload_queue", 0)
        st.session_state.total_episodes = data.get("total_episodes", 0)
        st.session_state.batch_status = data.get("batch_status", 0)
        return data

        # Track background sync progress
        if "upload_queue" in data:
            st.session_state.upload_queue = data["upload_queue"]
        if "total_episodes" in data:
            st.session_state.total_episodes = data["total_episodes"]

        return data
    except Exception as e:
        st.error(f"ZMQ Error: {e}")
        return None


# Automated status refresh on EVERY rerun to ensure parity with server
try:
    data = send_command({"command": "poll_status"})
    if data:
        st.session_state.total_episodes = data.get(
            "total_episodes", st.session_state.total_episodes
        )
        st.session_state.upload_queue = data.get(
            "upload_queue", st.session_state.upload_queue
        )
        st.session_state.batch_status = data.get(
            "batch_status", st.session_state.batch_status
        )
except Exception as e:
    print("Error while setting initial total episodes", e)


def sync_ui_to_joints(joints):
    """Updates the UI session state to match a 32-DOF joint vector."""
    st.session_state.staging_buffer = np.array(joints, dtype=np.float32)
    st.session_state.target_buffer = np.copy(st.session_state.staging_buffer)

    # Automatically activate sliders for important joints
    for idx, val in enumerate(joints):
        val = float(np.clip(val, -1, 1))
        if abs(val) > 1e-4:
            if idx not in st.session_state.active_joints:
                st.session_state.active_joints.add(idx)
            # Update the key associated with the slider so it jumps to the value
            st.session_state[f"input_{idx}"] = val


def handle_reset():
    resp = send_command({"command": "reset"})
    if resp and "joints" in resp:
        sync_ui_to_joints(resp["joints"])
        st.session_state.last_msg = ("Randomized! Sliders synced.", "🎲")


def handle_ik_pickup(offset_cm):
    st.session_state.ik_phase = 0
    st.session_state.ik_offset = offset_cm


def handle_start_recording(task_name):
    resp = send_command({"command": "start_recording", "task": task_name})
    if resp and resp.get("status") == "recording_started":
        st.session_state.is_recording = True
        st.session_state.last_msg = ("Recording Started!", "🔴")


def send_stop_recording():
    return send_command({"command": "stop_recording"})


def send_discard_recording():
    return send_command({"command": "discard_recording"})


def home_all():
    st.session_state.target_buffer.fill(0.0)
    st.session_state.staging_buffer.fill(0.0)
    for idx in st.session_state.active_joints:
        st.session_state[f"input_{idx}"] = 0.0


def clear_all():
    for idx in st.session_state.active_joints:
        if f"input_{idx}" in st.session_state:
            del st.session_state[f"input_{idx}"]
    st.session_state.active_joints.clear()
    st.session_state.target_buffer.fill(np.nan)
    st.session_state.staging_buffer.fill(np.nan)


# --- Phase Machine Sync Consumer ---
# This ensures that IK phase updates are applied to widgets BEFORE they are instantiated,
# while the actual phase command happened at the end of the previous run (after instantiation).
if "pending_sync" in st.session_state:
    sync_ui_to_joints(st.session_state.pending_sync)
    del st.session_state.pending_sync

st.title("GR1 Advanced Teleop Dashboard")

# Show any pending toasts from callbacks
if "last_msg" in st.session_state:
    msg, icon = st.session_state.last_msg
    st.toast(msg, icon=icon)
    del st.session_state.last_msg

# --- Recording Sidebar ---
with st.sidebar:
    st.header("🔴 Rec Manager")
    task_instruction = st.text_input("Task Instruction", value="Pick up the red cube")

    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False

    if not st.session_state.is_recording:
        st.button(
            "Start Recording",
            type="primary",
            use_container_width=True,
            on_click=handle_start_recording,
            args=(task_instruction,),
        )
    else:
        col_save, col_discard = st.columns(2)
        with col_save:
            if st.button(
                "✅ Save",
                use_container_width=True,
            ):
                send_stop_recording()
                st.session_state.is_recording = False
                st.rerun()
        with col_discard:
            if st.button(
                "❌ Discard",
                use_container_width=True,
            ):
                send_discard_recording()
                st.session_state.is_recording = False
                st.rerun()
        st.error("RECORDING...")

    st.divider()
    st.header("🎯 IK Configuration")
    reach_offset = st.slider("Reach Height (cm)", 5, 40, 5)

    st.divider()
    st.header("📊 Dataset Statistics")
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Total Episodes", st.session_state.total_episodes)
    with col_stat2:
        st.metric("Batch Status", f"{st.session_state.batch_status}/20")

    st.header("☁️ Cloud Sync Status")
    if st.session_state.upload_queue > 0:
        st.warning(f"Syncing: {st.session_state.upload_queue} episodes pending...")
    else:
        st.success("✅ All episodes synced to Hub.")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Refresh", icon="🔄", use_container_width=True):
            send_command({"command": "poll_status"})
            st.rerun()
    with col_btn2:
        if st.button(
            "Push to Hub", icon="☁️", type="primary", use_container_width=True
        ):
            send_command({"command": "sync"})
            st.rerun()

st.markdown("### Select Joint:")
col1, col2 = st.columns([3, 1])
with col1:
    selected_joint_name = st.selectbox(
        "Pick a joint...",
        options=COMPACT_WIRE_JOINTS,
        index=None,
        placeholder="Pick a joint...",
        label_visibility="collapsed",
    )
with col2:
    if st.button(
        "Add",
        type="primary",
        use_container_width=True,
    ):
        if selected_joint_name:
            idx = COMPACT_WIRE_JOINTS.index(selected_joint_name)
            st.session_state.active_joints.add(idx)
            val = st.session_state.staging_buffer[idx]
            if np.isnan(val):
                val = 0.0
                st.session_state.staging_buffer[idx] = val
            if f"input_{idx}" not in st.session_state:
                st.session_state[f"input_{idx}"] = val

st.divider()

if not st.session_state.active_joints:
    st.info("No active joints. Select a joint from the dropdown and click 'Add'.")
else:
    for idx in sorted(list(st.session_state.active_joints)):
        name = COMPACT_WIRE_JOINTS[idx]

        col_lbl, col_inp, col_clr = st.columns([3, 6, 1])
        with col_lbl:
            st.markdown(f"**[{idx:02}] {name}**")
        with col_inp:
            # Sliders pull value from keyed session state to reflect IK movement in real-time
            new_val = st.slider(
                f"Value for {name}",
                min_value=-1.0,
                max_value=1.0,
                step=0.01,
                label_visibility="collapsed",
                key=f"input_{idx}",
            )
            st.session_state.staging_buffer[idx] = new_val
        with col_clr:
            if st.button(
                "Remove",
                key=f"remove_btn_{idx}",
                use_container_width=True,
            ):
                st.session_state.active_joints.remove(idx)
                if f"input_{idx}" in st.session_state:
                    del st.session_state[f"input_{idx}"]
                st.session_state.target_buffer[idx] = np.nan
                st.session_state.staging_buffer[idx] = np.nan
                st.rerun()

st.divider()

col_sub, col_reach, col_reset, col_export, col_clr_all, col_home = st.columns(6)
with col_sub:
    if st.button(
        "Submit Request",
        type="primary",
        use_container_width=True,
    ):
        # Explicitly construct a clean 32-DOF packet to prevent cross-wiring
        final_packet = [float("nan")] * 32
        for idx in st.session_state.active_joints:
            final_packet[idx] = float(st.session_state.staging_buffer[idx])

        send_command({"target": final_packet})
        st.toast("Sent Action!", icon="🚀")

with col_reach:
    st.button(
        "🎯 IK Pickup",
        use_container_width=True,
        on_click=handle_ik_pickup,
        args=(reach_offset,),
    )

with col_reset:
    st.button(
        "Randomize Env",
        use_container_width=True,
        on_click=handle_reset,
    )

with col_export:
    export_data = {}
    for idx in sorted(list(st.session_state.active_joints)):
        name = COMPACT_WIRE_JOINTS[idx]
        val = st.session_state.staging_buffer[idx]
        export_data[name] = float(val)

    json_str = json.dumps(export_data, indent=2)
    st.download_button(
        label="Export JSON",
        data=json_str,
        file_name="teleop_settings.json",
        mime="application/json",
        use_container_width=True,
    )

with col_clr_all:
    if st.button(
        "Clear",
        use_container_width=True,
    ):
        clear_all()
        st.rerun()

with col_home:
    if st.button(
        "Home All",
        use_container_width=True,
    ):
        home_all()
        st.rerun()


# --- Phase Machine Executor (Producer) ---
# We execute the next phase at the end of the script so that the CURRENT state
# was rendered for the user before we request the next state and rerun.
if st.session_state.ik_phase is not None and st.session_state.ik_phase < 4:
    phase = st.session_state.ik_phase
    offset = st.session_state.get("ik_offset", 5)
    with st.spinner(f"Executing IK Phase {phase + 1} / 4..."):
        # Small delay to ensure the UI actually shows the state before moving to next
        import time

        time.sleep(0.1)
        resp = send_command(
            {"command": "ik_pickup", "phase": phase, "offset_cm": offset}
        )
        if resp and "joints" in resp:
            # Store for the NEXT run to avoid "modified after instantiation" errors
            st.session_state.pending_sync = resp["joints"]
            st.session_state.ik_phase += 1
            if st.session_state.ik_phase >= 4:
                st.session_state.ik_phase = None
                st.session_state.last_msg = (
                    "Pickup Complete! Sliders synced.",
                    "🎯",
                )
            st.rerun()
