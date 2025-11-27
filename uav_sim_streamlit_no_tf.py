"""
Streamlit web-app: UAV flight control demo WITHOUT TensorFlow
- Использует простую физику БПЛА (x,y,vx,vy,psi)
- EKF (Extended Kalman Filter)
- Predictor: numpy polyfit / линейная регрессия по времени (без TF)
- Blending controller: смешение PID и NN-предсказания
- 3D matplotlib + 2D time plot
Run:
    pip install streamlit numpy matplotlib scipy
    streamlit run uav_sim_streamlit_no_tf.py
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import block_diag
import time

st.set_page_config(layout="wide", page_title="UAV EKF+Polyfit Demo")
st.title("UAV: EKF + Polyfit prediction + Blended PID controller — demo (no TF)")

# ----------------------- Utilities and physics -----------------------
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

class SimpleUAV:
    """Simple planar UAV dynamics with yaw. State: [x,y,vx,vy,psi]
    Control: acceleration in body frame (ax_b, ay_b) and yaw_rate r
    """
    def __init__(self, dt=0.1):
        self.dt = dt

    def step(self, state, control):
        x, y, vx, vy, psi = state
        ax_b, ay_b, r = control
        ax = ax_b * np.cos(psi) - ay_b * np.sin(psi)
        ay = ax_b * np.sin(psi) + ay_b * np.cos(psi)
        vx_new = vx + ax * self.dt
        vy_new = vy + ay * self.dt
        x_new = x + vx_new * self.dt
        y_new = y + vy_new * self.dt
        psi_new = wrap_angle(psi + r * self.dt)
        return np.array([x_new, y_new, vx_new, vy_new, psi_new])

# ----------------------- EKF -----------------------
def ekf_predict(x, P, u, Q, dt):
    psi = x[4]
    ax_b, ay_b, r = u
    ax = ax_b * np.cos(psi) - ay_b * np.sin(psi)
    ay = ax_b * np.sin(psi) + ay_b * np.cos(psi)
    F = np.eye(5)
    F[0,2] = dt
    F[1,3] = dt
    dax_dpsi = -ax_b * np.sin(psi) - ay_b * np.cos(psi)
    day_dpsi = ax_b * np.cos(psi) - ay_b * np.sin(psi)
    F[2,4] = dax_dpsi * dt
    F[3,4] = day_dpsi * dt
    x_pred = x.copy()
    x_pred[0] = x[0] + (x[2] + ax * dt) * dt
    x_pred[1] = x[1] + (x[3] + ay * dt) * dt
    x_pred[2] = x[2] + ax * dt
    x_pred[3] = x[3] + ay * dt
    x_pred[4] = wrap_angle(x[4] + r * dt)
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, z, R):
    H = np.zeros((3,5))
    H[0,0] = 1
    H[1,1] = 1
    H[2,4] = 1
    y = z - H @ x_pred
    y[2] = wrap_angle(y[2])
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    x_upd[4] = wrap_angle(x_upd[4])
    P_upd = (np.eye(5) - K @ H) @ P_pred
    return x_upd, P_upd

# ----------------------- Predictor (no TF) -----------------------
class PolyfitPredictor:
    """
    Predict future positions using polynomial fit on recent window.
    Fits poly of degree deg separately to x(t) and y(t).
    If fit fails or insufficient data -> linear extrapolation by velocity.
    """
    def __init__(self, seq_len=10, pred_steps=5, deg=1, dt=0.1):
        self.seq_len = seq_len
        self.pred_steps = pred_steps
        self.deg = deg
        self.dt = dt

    def predict(self, seq):
        # seq: array of shape (seq_len, 5) -> [x,y,vx,vy,psi]
        n = len(seq)
        if n == 0:
            return None
        times = np.arange(-n+1, 1) * self.dt  # relative times ending at t=0
        xs = seq[:,0]
        ys = seq[:,1]
        try:
            # if not enough points for deg, fallback to deg = min(...)
            deg = min(self.deg, max(1, n-1))
            coef_x = np.polyfit(times, xs, deg)
            coef_y = np.polyfit(times, ys, deg)
            future = []
            for i in range(1, self.pred_steps+1):
                t = i * self.dt
                px = np.polyval(coef_x, t)
                py = np.polyval(coef_y, t)
                future.append([px, py])
            return np.array(future)
        except Exception:
            # fallback: linear extrapolation using last velocity
            last = seq[-1]
            x,y,vx,vy = last[0], last[1], last[2], last[3]
            preds = []
            for i in range(1, self.pred_steps+1):
                preds.append([x + vx * self.dt * i, y + vy * self.dt * i])
            return np.array(preds)

# ----------------------- PID controller -----------------------
class PID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.1, dt=0.1):
        self.kp = kp; self.ki = ki; self.kd = kd; self.dt = dt
        self.integral = np.zeros(2)
        self.prev_err = np.zeros(2)
    def reset(self):
        self.integral.fill(0); self.prev_err.fill(0)
    def control(self, pos, target):
        err = target - pos
        self.integral += err * self.dt
        deriv = (err - self.prev_err) / self.dt
        self.prev_err = err
        u = self.kp * err + self.ki * self.integral + self.kd * deriv
        return u

# ----------------------- Streamlit UI -----------------------
col1, col2 = st.columns([1,2])
with col1:
    st.header("Simulation params")
    steps = st.slider("Steps to simulate", 50, 1000, 300)
    dt = st.number_input("Delta t (s)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    meas_noise = st.slider("Measurement noise (pos sigma, m)", 0.0, 5.0, 0.5)
    psi_noise = st.slider("Measurement noise (yaw sigma, rad)", 0.0, 0.5, 0.05)
    process_noise = st.slider("Process noise scale", 0.0, 1.0, 0.02)
    alpha = st.slider("Blending alpha (0=PID only,1=NN only)", 0.0, 1.0, 0.4)
    pred_steps = st.slider("Prediction steps", 1, 20, 10)
    seq_len = st.slider("History window (seq_len)", 3, 50, 10)
    deg = st.slider("Poly degree for fit", 1, 3, 1)
    kp = st.number_input("PID kp (pos)", value=2.0)
    kd = st.number_input("PID kd (pos)", value=0.5)
    run_button = st.button("Run simulation")
with col2:
    st.header("Plot")
    plot_area = st.empty()

# ----------------------- Main simulation -----------------------
if run_button:
    np.random.seed(123)
    uav = SimpleUAV(dt=dt)
    x_true = np.array([0.0, 0.0, 1.0, 0.2, 0.0])
    x_est = x_true.copy() + np.array([0.1, -0.1, 0.0, 0.0, 0.01])
    P = np.diag([1,1,1,1,0.1])
    Q = np.diag([process_noise]*5)
    R = np.diag([meas_noise**2, meas_noise**2, psi_noise**2])

    pid = PID(kp=kp, kd=kd, dt=dt)
    predictor = PolyfitPredictor(seq_len=seq_len, pred_steps=pred_steps, deg=deg, dt=dt)

    true_traj = []
    meas_traj = []
    ekf_traj = []
    nn_preds = []
    window = []

    def desired_waypoint(t):
        return np.array([10.0 * np.cos(0.02*t), 8.0 * np.sin(0.015*t)])

    for k in range(steps):
        t = k*dt
        wp = desired_waypoint(k)
        meas = np.array([x_true[0] + np.random.randn()*meas_noise,
                         x_true[1] + np.random.randn()*meas_noise,
                         wrap_angle(x_true[4] + np.random.randn()*psi_noise)])
        # prediction from window
        if len(window) >= 2:
            seq = np.array(window[-min(len(window), seq_len):])
            nn_future = predictor.predict(seq)
        else:
            seq = np.array(window[-min(len(window), seq_len):])
            nn_future = None

        pid_u = pid.control(np.array([x_est[0], x_est[1]]), wp)
        if nn_future is not None:
            nn_target = nn_future[0]
            nn_u = pid.control(np.array([x_est[0], x_est[1]]), nn_target)
            blended_u = (1-alpha)*pid_u + alpha*nn_u
        else:
            blended_u = pid_u

        psi = x_true[4]
        ax_b = blended_u[0] * np.cos(psi) + blended_u[1] * np.sin(psi)
        ay_b = -blended_u[0] * np.sin(psi) + blended_u[1] * np.cos(psi)
        desired_yaw = np.arctan2(wp[1]-x_est[1], wp[0]-x_est[0])
        r = 2.0 * wrap_angle(desired_yaw - x_est[4])
        control = np.array([ax_b, ay_b, r])

        x_true = uav.step(x_true, control)
        true_traj.append(x_true.copy())
        meas_traj.append(meas.copy())

        x_pred, P = ekf_predict(x_est, P, control, Q, dt)
        x_est, P = ekf_update(x_pred, P, meas, R)
        ekf_traj.append(x_est.copy())

        window.append(x_est.copy())
        if len(window) > 500:
            window.pop(0)

        if len(window) >= 2:
            nn_preds.append(predictor.predict(np.array(window[-min(len(window), seq_len):])))
        else:
            nn_preds.append(None)

    true_traj = np.array(true_traj)
    meas_traj = np.array(meas_traj)
    ekf_traj = np.array(ekf_traj)

    # Plot
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(121, projection='3d')
    T = np.arange(len(true_traj))*dt
    ax.plot(true_traj[:,0], true_traj[:,1], T, linestyle='--', color='k', label='True (dashed)')
    ax.scatter(meas_traj[:,0], meas_traj[:,1], T, c='r', s=6, label='Measurements')
    ax.plot(ekf_traj[:,0], ekf_traj[:,1], T, color='b', linewidth=2, label='EKF')
    # plot some NN preds
    step = max(1, len(nn_preds)//12)
    for i, p in enumerate(nn_preds[::step]):
        if p is not None:
            base_idx = i * step
            base_t = base_idx*dt
            ax.plot(p[:,0], p[:,1], base_t + np.arange(1,1+pred_steps)*dt, color='g', alpha=0.6)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Time (s)')
    ax.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(T, true_traj[:,0], label='x true')
    ax2.plot(T, ekf_traj[:,0], label='x ekf')
    ax2.plot(T, meas_traj[:,0], '.', label='x meas', markersize=3)
    ax2.set_xlabel('Time (s)')
    ax2.legend()

    plot_area.pyplot(fig)
    st.markdown("**Legend:** black dashed = true trajectory, red dots = noisy measurements, blue = EKF estimate, green = polyfit predictions (segments)")
    st.success('Simulation finished')
else:
    st.info('Configure parameters at left and press Run simulation')
