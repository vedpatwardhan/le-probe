import numpy as np
import zmq
import msgpack
import rerun as rr
import mujoco
import os
import datetime
from pathlib import Path
from PIL import Image
from simulation_base import GR1MuJoCoBase
from gr1_config import BASE_DIR, SCENE_PATH


class GR1TeleopServer(GR1MuJoCoBase):
    """
    Reactive Teleoperation Server (REP Socket).
    Dedicated to the Streamlit Dashboard and IK Calibration.
    """

    def __init__(self, scene_path=None, port=5556):
        super().__init__(scene_path or SCENE_PATH, restrict_ik=True)
        self.port = port
        self.is_running = True

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.port}")

        rr.init("gr1_teleop", spawn=False)
        rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
        print(f"🚀 Teleop Server Running on port {self.port}")

        while self.is_running:
            msg = socket.recv()
            data = msgpack.unpackb(msg, raw=False)
            cmd = data.get("command")

            def send_resp(payload):
                payload.update(
                    {
                        "upload_queue": self.recorder.pending_uploads,
                        "total_episodes": self.recorder.total_episodes,
                        "batch_status": self.recorder.episodes_since_sync,
                    }
                )
                socket.send(msgpack.packb(payload))

            if cmd == "reset":
                self.reset_env()
                send_resp(
                    {"status": "reset_ok", "joints": self.get_state_32().tolist()}
                )

            elif cmd == "sync":
                self.recorder.force_sync()
                send_resp({"status": "sync_started"})

            elif cmd == "start_recording":
                self.recorder.start_episode(data.get("task", "Pick up red cube"))
                self.is_recording = True
                send_resp({"status": "recording_started"})

            elif cmd == "stop_recording":
                self.recorder.stop_episode()
                self.is_recording = False
                send_resp({"status": "recording_stopped"})

            elif cmd == "discard_recording":
                self.recorder.discard_episode()
                self.is_recording = False
                send_resp({"status": "recording_discarded"})

            elif cmd == "poll_status":
                send_resp({"status": "status_ok"})

            elif cmd == "ik_pickup":
                phase = data.get("phase", 0)
                offset_cm = data.get("offset_cm", 5)
                self._handle_ik_pickup_logic(phase=phase, offset_cm=offset_cm)
                send_resp(
                    {"status": "ik_pickup_ok", "joints": self.get_state_32().tolist()}
                )

            elif "target" in data:
                action_32 = np.array(data["target"], dtype=np.float32)
                self.process_target_32(action_32)
                self.dispatch_action(action_32, self.last_target_q)
                send_resp({"status": "step_ok"})
            else:
                send_resp({"status": "unknown"})

    def _handle_ik_pickup_logic(self, phase=0, offset_cm=5):
        """Hardened multi-phase IK solver for red cube."""
        print(f"🎯 Executing IK Pickup Phase {phase}...")

        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_pos = self.data.qpos[
            self.model.jnt_qposadr[cube_id] : self.model.jnt_qposadr[cube_id] + 3
        ].copy()

        if phase == 0:
            # Phase 1: Lift (Approach)
            pos_i_h, pos_t_h, pos_w_h = (
                cube_pos + [0, 0.04, 0.07 + offset_cm / 100.0],
                cube_pos + [0.015, -0.035, 0.07 + offset_cm / 100.0],
                cube_pos + [0, 0, 0.15 + offset_cm / 100.0],
            )
            q_reach_h = self.solve_ik(pos_w_h, [0, 1, 0, 0], pos_i_h, pos_t_h)
            self.dispatch_action(self.qpos_to_action_32(q_reach_h), q_reach_h)

        elif phase == 1:
            # Phase 2: Descent
            pos_i_l, pos_t_l, pos_w_l = (
                cube_pos + [0, 0.04, 0.0],
                cube_pos + [0.015, -0.035, 0.0],
                cube_pos + [0, 0, 0.04],
            )
            q_reach_l = self.solve_ik(pos_w_l, [0, 1, 0, 0], pos_i_l, pos_t_l)
            self.dispatch_action(self.qpos_to_action_32(q_reach_l), q_reach_l)

        elif phase == 2:
            # Phase 3: Grasp
            # Re-solve descent to get the base q for grasping
            pos_i_l, pos_t_l, pos_w_l = (
                cube_pos + [0, 0.04, 0.0],
                cube_pos + [0.015, -0.035, 0.0],
                cube_pos + [0, 0, 0.04],
            )
            q_reach_l = self.solve_ik(pos_w_l, [0, 1, 0, 0], pos_i_l, pos_t_l)
            q_grasp = q_reach_l.copy()
            q_grasp[47], q_grasp[48] = -1.1, 1.1
            for g_id in [50, 52, 54, 56]:
                q_grasp[g_id] = -1.1
            self.dispatch_action(self.qpos_to_action_32(q_grasp), q_grasp)

        elif phase == 3:
            # Phase 4: Lift (Retract)
            pos_i_up, pos_t_up, pos_w_up = (
                cube_pos + [0, 0.04, 0.15],
                cube_pos + [0.015, -0.035, 0.15],
                cube_pos + [0, 0, 0.19],
            )
            q_lift = self.solve_ik(pos_w_up, [0, 1, 0, 0], pos_i_up, pos_t_up)
            q_lift[47], q_lift[48] = -1.1, 1.1
            for g_id in [50, 52, 54, 56]:
                q_lift[g_id] = -1.1
            self.dispatch_action(self.qpos_to_action_32(q_lift), q_lift)


if __name__ == "__main__":
    GR1TeleopServer().run()
