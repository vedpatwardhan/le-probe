import streamlit as st
import numpy as np
import zmq
import msgpack
import json
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

if "upload_queue" not in st.session_state:
    st.session_state.upload_queue = 0

if "total_episodes" not in st.session_state:
    st.session_state.total_episodes = 0

# --- Load Default Active Joints ---
if "active_joints" not in st.session_state:
    st.session_state.active_joints = set()
    with open("gr1_gr00t/teleop_joints.txt", "r") as f:
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

        # Track background sync progress
        if "upload_queue" in data:
            st.session_state.upload_queue = data["upload_queue"]
        if "total_episodes" in data:
            st.session_state.total_episodes = data["total_episodes"]

        return data
    except Exception as e:
        st.error(f"ZMQ Error: {e}")
        return None


def sync_ui_to_joints(joints):
    """Updates the UI session state to match a 32-DOF joint vector."""
    st.session_state.staging_buffer = np.array(joints, dtype=np.float32)
    st.session_state.target_buffer = np.copy(st.session_state.staging_buffer)

    # Automatically activate and update sliders for important joints
    for idx, val in enumerate(joints):
        # Use a safe epsilon clip to prevent StreamlitValueAboveMaxErrors (1.0000004...)
        val = float(np.clip(val, -1, 1))
        if abs(val) > 1e-4:
            if idx not in st.session_state.active_joints:
                name = COMPACT_WIRE_JOINTS[idx]
                print(
                    f"[UI] Auto-activating '{name}' (idx {idx}) because value is {val:.6f}"
                )
                st.session_state.active_joints.add(idx)
            st.session_state[f"input_{idx}"] = val


def handle_reset():
    resp = send_command({"command": "reset"})
    if resp and "joints" in resp:
        sync_ui_to_joints(resp["joints"])
        st.session_state.last_msg = ("Randomized! Sliders synced.", "🎲")


def handle_auto_reach(offset_cm):
    resp = send_command({"command": "auto_reach", "offset_cm": offset_cm})
    if resp and "joints" in resp:
        sync_ui_to_joints(resp["joints"])
        st.session_state.last_msg = ("Reach Complete! Sliders synced.", "🎯")


def handle_start_recording(task_name):
    resp = send_command({"command": "start_recording", "task": task_name})
    if resp and resp.get("status") == "recording_started":
        st.session_state.is_recording = True
        st.session_state.last_msg = ("Recording Started!", "🔴")


def send_stop_recording():
    return send_command({"command": "stop_recording"})


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
        if st.button("Stop & Save", use_container_width=True):
            send_stop_recording()
            st.session_state.is_recording = False
            st.rerun()
        st.error("RECORDING...")

    st.divider()
    st.header("🎯 IK Configuration")
    reach_offset = st.slider("Reach Height (cm)", 5, 40, 5)

    st.divider()
    st.header("📊 Dataset Statistics")
    st.metric("Total Episodes", st.session_state.total_episodes)

    st.header("☁️ Cloud Sync Status")
    if st.session_state.upload_queue > 0:
        st.warning(f"Syncing: {st.session_state.upload_queue} episodes pending...")
    else:
        st.success("✅ All episodes synced to Hub.")

    if st.button("Refresh Sync Status"):
        # Dummy step command to poll status
        send_command({"command": "poll_status"})

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
    if st.button("Add", type="primary", use_container_width=True):
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
            if f"input_{idx}" not in st.session_state:
                st.session_state[f"input_{idx}"] = 0.0

            new_val = st.slider(
                f"Value for {name}",
                min_value=-1.0,
                max_value=1.0,
                step=0.01,
                key=f"input_{idx}",
                label_visibility="collapsed",
            )
            st.session_state.staging_buffer[idx] = new_val
        with col_clr:
            if st.button("Remove", key=f"remove_btn_{idx}", use_container_width=True):
                st.session_state.active_joints.remove(idx)
                if f"input_{idx}" in st.session_state:
                    del st.session_state[f"input_{idx}"]
                st.session_state.target_buffer[idx] = np.nan
                st.session_state.staging_buffer[idx] = np.nan
                st.rerun()

st.divider()

col_sub, col_reach, col_reset, col_export, col_clr_all, col_home = st.columns(6)
with col_sub:
    if st.button("Submit Request", type="primary", use_container_width=True):
        # Explicitly construct a clean 32-DOF packet to prevent cross-wiring
        final_packet = [float("nan")] * 32
        for idx in st.session_state.active_joints:
            final_packet[idx] = float(st.session_state.staging_buffer[idx])

        send_command({"target": final_packet})
        st.toast("Sent Action!", icon="🚀")

with col_reach:
    st.button(
        "🚀 Auto-Reach Cube",
        use_container_width=True,
        on_click=handle_auto_reach,
        args=(reach_offset,),
    )

with col_reset:
    st.button("Randomize Env", use_container_width=True, on_click=handle_reset)

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
    if st.button("Clear", use_container_width=True):
        clear_all()
        st.rerun()

with col_home:
    if st.button("Home All", use_container_width=True):
        home_all()
        st.rerun()
