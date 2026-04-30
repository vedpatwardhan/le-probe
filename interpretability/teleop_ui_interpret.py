import streamlit as st
import numpy as np
import zmq
import msgpack
import json
import os
import sys
import threading
import time

# --- Path Stabilization ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LEWM_DIR = os.path.join(ROOT_DIR, "lewm")
LE_WM_DIR = os.path.join(LEWM_DIR, "le_wm")

for p in [ROOT_DIR, LEWM_DIR, LE_WM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from gr1_config import COMPACT_WIRE_JOINTS

st.set_page_config(page_title="Le-Probe: Latent Explorer", layout="wide")

# --- Visual Polish: Reduce Top Padding ---
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 0rem;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Task-Centric Filter (Joints 16-31 Only) ---
TASK_JOINTS = COMPACT_WIRE_JOINTS[16:]


# --- Setup ZMQ Sockets ---
@st.cache_resource
def get_zmq_resources():
    context = zmq.Context()
    sim_socket = context.socket(zmq.REQ)
    sim_socket.connect("tcp://127.0.0.1:5556")
    latent_socket = context.socket(zmq.REQ)
    latent_socket.connect("tcp://127.0.0.1:5557")
    socket_lock = threading.Lock()
    return sim_socket, latent_socket, socket_lock


sim_socket, latent_socket, socket_lock = get_zmq_resources()

# --- Initialize Session State ---
if "staging_buffer" not in st.session_state:
    st.session_state.staging_buffer = np.full(32, 0.0, dtype=np.float32)
if "active_joints" not in st.session_state:
    st.session_state.active_joints = set()
    ik_path = f"{ROOT_DIR}/ik_joints.txt"
    if os.path.exists(ik_path):
        with open(ik_path, "r") as f:
            for line in f:
                name = line.strip().split("#")[0].strip()
                if name in TASK_JOINTS:
                    idx = COMPACT_WIRE_JOINTS.index(name)
                    st.session_state.active_joints.add(idx)
                    st.session_state[f"input_{idx}"] = 0.0


def send_command(payload):
    with socket_lock:
        try:
            if sim_socket.poll(0, zmq.POLLIN):
                sim_socket.recv(zmq.NOBLOCK)
            sim_socket.send(msgpack.packb(payload, use_bin_type=True))
            if sim_socket.poll(2000):
                resp = sim_socket.recv()
                data = msgpack.unpackb(resp, raw=False)
                if "image" in data:
                    st.session_state.last_image = data["image"]
                return data
            return None
        except Exception:
            return None


def get_latent_activations(action_32):
    if "last_image" not in st.session_state:
        return None
    with socket_lock:
        try:
            if latent_socket.poll(0, zmq.POLLIN):
                latent_socket.recv(zmq.NOBLOCK)
            payload = {
                "command": "get_activations",
                "image": st.session_state.last_image,
                "action": action_32.tolist(),
            }
            latent_socket.send(msgpack.packb(payload, use_bin_type=True))
            if latent_socket.poll(2000):
                resp = latent_socket.recv()
                return msgpack.unpackb(resp, raw=False)
            return None
        except Exception:
            return None


def update_feature_label(fid, label):
    try:
        payload = {"command": "update_label", "feature_id": fid, "label": label}
        latent_socket.send(msgpack.packb(payload, use_bin_type=True))
        latent_socket.recv()
        st.toast(f"Updated Label for Feature {fid}!", icon="🏷️")
    except Exception:
        pass


# --- Callbacks ---
def handle_restore(snap_idx):
    resp = send_command({"command": "load_snapshot", "index": snap_idx})
    if resp and resp.get("status") == "load_ok":
        # 1. Update Joint Sliders
        for idx, val in enumerate(resp["joints"]):
            if idx >= 16 and abs(val) > 1e-4:
                st.session_state.active_joints.add(idx)
                st.session_state[f"input_{idx}"] = float(np.clip(val, -1.0, 1.0))

        # 2. Update Action Buffer
        st.session_state.staging_buffer = np.array(resp["action_32"], dtype=np.float32)

        # 3. Trigger Audit
        st.session_state._trigger_audit = True
        st.toast(f"Reproduced Snapshot {snap_idx}!", icon="⏪")


def handle_randomize():
    resp = send_command({"command": "reset"})
    if resp and "joints" in resp:
        for idx, val in enumerate(resp["joints"]):
            if idx >= 16 and abs(val) > 1e-4:
                st.session_state.active_joints.add(idx)
                st.session_state[f"input_{idx}"] = float(np.clip(val, -1.0, 1.0))
    st.session_state._trigger_audit = True  # Force audit refresh


def handle_wild_randomize():
    resp = send_command({"command": "wild_randomize"})
    if resp and "joints" in resp:
        for idx, val in enumerate(resp["joints"]):
            if idx >= 16 and abs(val) > 1e-4:
                st.session_state.active_joints.add(idx)
                st.session_state[f"input_{idx}"] = float(np.clip(val, -1.0, 1.0))
    st.session_state._trigger_audit = True  # Force audit refresh


def handle_submit():
    final_packet = [0.0] * 32
    for idx in st.session_state.active_joints:
        final_packet[idx] = float(st.session_state.staging_buffer[idx])
    send_command({"target": final_packet})
    st.session_state._trigger_audit = True  # Refresh audit on submit


def handle_snapshot():
    resp = send_command({"command": "store_snapshot"})
    if resp and resp.get("status") == "snapshot_ok":
        st.session_state._snap_msg = True


# --- Sync Status ---
try:
    send_command({"command": "poll_status"})
except:
    pass

st.title("🔬 Le-Probe: Mechanistic Teleop")
st.divider()

# --- Split-Pane Dashboard ---
col_control, col_audit = st.columns([1.2, 1], gap="large")

# --- Pane 1: Teleop Control ---
with col_control:
    st.subheader("🎮 Teleop Control")
    st.write("")

    # MAIN BODY TOOLBAR: Triggers full rerun (and thus audit refresh)
    c_sel, c_add, c_sub, c_rand, c_wild, c_snap = st.columns(
        [2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    with c_sel:
        sel_name = st.selectbox(
            "Select Joint",
            options=TASK_JOINTS,
            index=None,
            label_visibility="collapsed",
        )
    with c_add:
        if st.button("Add", use_container_width=True):
            if sel_name:
                idx = COMPACT_WIRE_JOINTS.index(sel_name)
                st.session_state.active_joints.add(idx)
                st.session_state[f"input_{idx}"] = 0.0
                st.rerun()
    with c_sub:
        st.button(
            "🚀 Submit",
            type="primary",
            use_container_width=True,
            on_click=handle_submit,
        )
    with c_rand:
        st.button("🎲 Rand", use_container_width=True, on_click=handle_randomize)
    with c_wild:
        st.button("🌀 Wild", use_container_width=True, on_click=handle_wild_randomize)
    with c_snap:
        st.button("📸 Snap", use_container_width=True, on_click=handle_snapshot)
        if st.session_state.get("_snap_msg"):
            st.toast("Snapshot Stored!", icon="📸")
            del st.session_state._snap_msg

    st.write("")

    @st.fragment  # Isolated Sliders: Moving them does NOT trigger audit
    def render_sliders():
        with st.container(height=550):
            if not st.session_state.active_joints:
                st.info("No active joints.")
            else:
                for idx in sorted(list(st.session_state.active_joints)):
                    name = COMPACT_WIRE_JOINTS[idx]
                    col_lbl, col_inp, col_clr = st.columns([3, 6, 1])
                    with col_lbl:
                        st.markdown(f"**[{idx:02}]** {name}")
                    with col_inp:
                        st.slider(
                            name,
                            -1.0,
                            1.0,
                            step=0.01,
                            label_visibility="collapsed",
                            key=f"input_{idx}",
                        )
                        st.session_state.staging_buffer[idx] = st.session_state[
                            f"input_{idx}"
                        ]
                    with col_clr:
                        if st.button("X", key=f"rm_{idx}", use_container_width=True):
                            st.session_state.active_joints.remove(idx)
                            del st.session_state[f"input_{idx}"]
                            st.session_state.staging_buffer[idx] = 0.0
                            st.rerun()

    render_sliders()

# --- Pane 2: Latent Intent Audit ---
with col_audit:
    st.subheader("🔬 Latent Intent Audit")
    st.write("")

    # Audit Logic
    if st.session_state.get("_trigger_audit"):
        acts_data = get_latent_activations(st.session_state.staging_buffer)
        if acts_data and acts_data.get("status") == "ok":
            st.session_state._last_acts = acts_data
        del st.session_state._trigger_audit

    @st.fragment
    def render_audit_pane():
        # Configuration & Trigger Row (Symmetric Button Widths)
        c_id, c_lbl, c_sv, c_aud, c_sid, c_res = st.columns(
            [0.6, 2.9, 1.3, 1.3, 0.6, 1.3]
        )
        with c_id:
            feat_id = st.number_input(
                "ID", min_value=0, max_value=1023, step=1, label_visibility="collapsed"
            )
        with c_lbl:
            n_label = st.text_input(
                "Label Overwrite",
                key="label_input",
                label_visibility="collapsed",
                placeholder="Enter Feature Label...",
            )
        with c_sv:
            if st.button("🏷️ Save", type="primary", use_container_width=True):
                if n_label:
                    update_feature_label(feat_id, n_label)
                    st.rerun()
        with c_aud:
            if st.button("🔍 Audit", use_container_width=True):
                st.session_state._trigger_audit = True
                st.rerun()
        with c_sid:
            s_idx = st.number_input(
                "SnapID", min_value=0, step=1, label_visibility="collapsed"
            )
        with c_res:
            if st.button("⏪ Restore", use_container_width=True):
                handle_restore(s_idx)
                st.rerun()

        stored_acts = st.session_state.get("_last_acts")
        top_features = stored_acts["top_features"] if stored_acts else []

        st.write("")
        with st.container(height=550):
            if top_features:
                import pandas as pd
                import plotly.express as px

                df = pd.DataFrame(top_features, columns=["ID", "Activation", "Label"])
                df["Display"] = df.apply(
                    lambda x: (
                        f"<b>[{x['ID']}]</b> {x['Label']}"
                        if x["Label"]
                        else f"<b>[{x['ID']}]</b> (unlabeled)"
                    ),
                    axis=1,
                )
                neural_heat = [[0, "#0077FF"], [1, "#FF3300"]]
                fig = px.bar(
                    df,
                    x="Activation",
                    y="Display",
                    orientation="h",
                    color="Activation",
                    color_continuous_scale=neural_heat,
                )
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending", "title": ""},
                    xaxis={"title": "Activation Signal"},
                    height=500,
                    margin=dict(l=220, r=20, t=10, b=10),
                    coloraxis_showscale=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig, use_container_width=True)
            elif not stored_acts:
                st.info("Click '🔍 Audit Now' or 'Submit' to inspect model intent.")
            else:
                st.info("No active features found.")

    render_audit_pane()
