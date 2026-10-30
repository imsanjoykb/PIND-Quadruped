import mujoco
import numpy as np
import torch
from typing import Tuple, Optional


class QuadrupedMuJoCoInterface:
    
    def __init__(self, xml_path: Optional[str] = None):
        """
        Initialize MuJoCo interface.
        
        Args:
            xml_path: Path to MJCF XML file. If None, creates procedural model.
        """
        if xml_path is not None:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            # Fallback procedural MJCF for an 18-DOF floating-base quadruped
            self.model = self._create_procedural_quadruped_model()
            
        self.data = mujoco.MjData(self.model)
        self.nv = self.model.nv  # 18 DOFs (6 base + 12 joints)
        self.nq = self.model.nq  # 19 coordinates (7 base with quaternion + 12 joints)
        self.nu = self.model.nu  # 12 actuated joints

    def _create_procedural_quadruped_model(self) -> mujoco.MjModel:
        """
        Create a procedural 18-DOF floating-base quadruped model.
        
        Returns:
            MuJoCo model instance
        """
        xml = """
        <mujoco model="procedural_quadruped">
          <compiler angle="radian" coordinate="local"/>
          <option gravity="0 0 -9.81" timestep="0.001"/>
          <worldbody>
            <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1" friction="0.8 0.1 0.1"/>
            <body name="base" pos="0 0 0.35">
              <freejoint name="root"/>
              <geom name="torso" type="box" size="0.2 0.1 0.06" mass="6.0" rgba="0.2 0.3 0.8 1"/>
              <!-- FL Leg -->
              <body name="fl_hip" pos="0.18 0.1 0">
                <joint name="fl_hx" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
                <geom type="capsule" size="0.02 0.05" mass="0.6"/>
                <body name="fl_thigh" pos="0 0.05 0">
                  <joint name="fl_hy" type="hinge" axis="0 1 0" range="-1.0 2.0"/>
                  <geom type="capsule" size="0.02 0.1" mass="0.8"/>
                  <body name="fl_calf" pos="0 0 -0.15">
                    <joint name="fl_kn" type="hinge" axis="0 1 0" range="-2.5 -0.5"/>
                    <geom type="capsule" size="0.015 0.1" mass="0.6"/>
                    <site name="fl_foot" pos="0 0 -0.15" size="0.02"/>
                  </body>
                </body>
              </body>
              <!-- FR Leg -->
              <body name="fr_hip" pos="0.18 -0.1 0">
                <joint name="fr_hx" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
                <geom type="capsule" size="0.02 0.05" mass="0.6"/>
                <body name="fr_thigh" pos="0 -0.05 0">
                  <joint name="fr_hy" type="hinge" axis="0 1 0" range="-1.0 2.0"/>
                  <geom type="capsule" size="0.02 0.1" mass="0.8"/>
                  <body name="fr_calf" pos="0 0 -0.15">
                    <joint name="fr_kn" type="hinge" axis="0 1 0" range="-2.5 -0.5"/>
                    <geom type="capsule" size="0.015 0.1" mass="0.6"/>
                    <site name="fr_foot" pos="0 0 -0.15" size="0.02"/>
                  </body>
                </body>
              </body>
              <!-- RL Leg -->
              <body name="rl_hip" pos="-0.18 0.1 0">
                <joint name="rl_hx" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
                <geom type="capsule" size="0.02 0.05" mass="0.6"/>
                <body name="rl_thigh" pos="0 0.05 0">
                  <joint name="rl_hy" type="hinge" axis="0 1 0" range="-1.0 2.0"/>
                  <geom type="capsule" size="0.02 0.1" mass="0.8"/>
                  <body name="rl_calf" pos="0 0 -0.15">
                    <joint name="rl_kn" type="hinge" axis="0 1 0" range="-2.5 -0.5"/>
                    <geom type="capsule" size="0.015 0.1" mass="0.6"/>
                    <site name="rl_foot" pos="0 0 -0.15" size="0.02"/>
                  </body>
                </body>
              </body>
              <!-- RR Leg -->
              <body name="rr_hip" pos="-0.18 -0.1 0">
                <joint name="rr_hx" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
                <geom type="capsule" size="0.02 0.05" mass="0.6"/>
                <body name="rr_thigh" pos="0 -0.05 0">
                  <joint name="rr_hy" type="hinge" axis="0 1 0" range="-1.0 2.0"/>
                  <geom type="capsule" size="0.02 0.1" mass="0.8"/>
                  <body name="rr_calf" pos="0 0 -0.15">
                    <joint name="rr_kn" type="hinge" axis="0 1 0" range="-2.5 -0.5"/>
                    <geom type="capsule" size="0.015 0.1" mass="0.6"/>
                    <site name="rr_foot" pos="0 0 -0.15" size="0.02"/>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
          <actuator>
            <motor name="fl_hx_act" joint="fl_hx" gear="1"/>
            <motor name="fl_hy_act" joint="fl_hy" gear="1"/>
            <motor name="fl_kn_act" joint="fl_kn" gear="1"/>
            <motor name="fr_hx_act" joint="fr_hx" gear="1"/>
            <motor name="fr_hy_act" joint="fr_hy" gear="1"/>
            <motor name="fr_kn_act" joint="fr_kn" gear="1"/>
            <motor name="rl_hx_act" joint="rl_hx" gear="1"/>
            <motor name="rl_hy_act" joint="rl_hy" gear="1"/>
            <motor name="rl_kn_act" joint="rl_kn" gear="1"/>
            <motor name="rr_hx_act" joint="rr_hx" gear="1"/>
            <motor name="rr_hy_act" joint="rr_hy" gear="1"/>
            <motor name="rr_kn_act" joint="rr_kn" gear="1"/>
          </actuator>
        </mujoco>
        """
        return mujoco.MjModel.from_xml_string(xml)

    def compute_nominal_dynamics(
        self, q: np.ndarray, v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute analytical M(q), nonlinear bias h(q, v), and foot contact Jacobians.
        
        Args:
            q: Generalized positions (nq,)
            v: Generalized velocities (nv,)
        
        Returns:
            M: Mass matrix (nv, nv)
            h: Nonlinear bias forces C(q, v)v + g(q) (nv,)
            p_feet: Foot positions (4, 3)
            J_feet: Foot Jacobians (4, 3, nv)
        """
        self.data.qpos[:] = q
        self.data.qvel[:] = v
        mujoco.mj_forward(self.model, self.data)

        # 1. Mass matrix M(q)
        M = np.zeros((self.nv, self.nv), dtype=np.float64)
        mujoco.mj_fullM(self.model, M, self.data.qM)

        # 2. Coriolis, centrifugal, and gravity wrench h(q, v)
        # Using Inverse Dynamics (RNEA) with zero acceleration
        self.data.qacc[:] = 0.0
        mujoco.mj_inverse(self.model, self.data)
        h = self.data.qfrc_inverse.copy()

        # 3. Foot positions and Jacobians
        foot_site_names = ["fl_foot", "fr_foot", "rl_foot", "rr_foot"]
        p_feet = []
        J_feet = []
        
        for name in foot_site_names:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            p = self.data.site_xpos[site_id].copy()
            p_feet.append(p)
            
            jacp = np.zeros((3, self.nv), dtype=np.float64)
            jacr = np.zeros((3, self.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
            J_feet.append(jacp)

        return M, h, np.array(p_feet), np.array(J_feet)
