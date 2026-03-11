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
    socket = context.socket(zmq.PUB)
    socket.connect("tcp://127.0.0.1:5556")
    return socket

socket = get_zmq_socket()

# --- Initialize Session State ---
if 'active_joints' not in st.session_state:
    st.session_state.active_joints = set()
    
if 'staging_buffer' not in st.session_state:
    st.session_state.staging_buffer = np.full(32, np.nan, dtype=np.float32)

if 'target_buffer' not in st.session_state:
    st.session_state.target_buffer = np.full(32, np.nan, dtype=np.float32)

def send_command():
    payload = {"target": st.session_state.target_buffer.tolist()}
    socket.send(msgpack.packb(payload, use_bin_type=True))

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

st.markdown("### Select Joint:")
col1, col2 = st.columns([3, 1])
with col1:
    selected_joint_name = st.selectbox(
        "Pick a joint...", 
        options=COMPACT_WIRE_JOINTS,
        index=None,
        placeholder="Pick a joint...",
        label_visibility="collapsed"
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
                label_visibility="collapsed"
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

col_sub, col_export, col_clr_all, col_home = st.columns(4)
with col_sub:
    if st.button("Submit Request", type="primary", use_container_width=True):
        st.session_state.target_buffer = np.copy(st.session_state.staging_buffer)
        send_command()
        st.success("Sent!", icon="✅")
        
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
        use_container_width=True
    )
        
with col_clr_all:
    if st.button("Clear", use_container_width=True):
        clear_all()
        st.rerun()
        
with col_home:
    if st.button("Home All", use_container_width=True):
        home_all()
        st.rerun()
