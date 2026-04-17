import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped, PoseArray
from std_msgs.msg import Float64

# ---------------- PID Controller ----------------
class PID:
    def __init__(self, kp, ki, kd, integral_limit=10):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
        self.integral_limit = integral_limit

    def compute(self, error, dt):
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

# ---------------- Research-grade 6DOF AUV with Level9 ROS features ----------------
class FullAUV6DOF:
    def __init__(self, waypoints=None, dt=0.1):
        # Physical parameters
        self.m = 100.0
        self.Ix, self.Iy, self.Iz = 10.0, 20.0, 30.0
        self.W = 981.0
        self.Buoy = 981.0
        self.xg, self.zg = 0.0, 0.0

        # Added mass
        self.Xudot, self.Yvdot, self.Zwdot = -20.0, -30.0, -40.0
        self.Kpdot, self.Mqdot, self.Nrdot = -5.0, -10.0, -15.0

        # Linear & quadratic damping
        self.Xu, self.Yv, self.Zw = -10.0, -15.0, -20.0
        self.Kp, self.Mq, self.Nr = -5.0, -8.0, -12.0
        self.Xuu, self.Yvv, self.Zww = -20.0, -30.0, -40.0

        # Thruster allocation matrix (6DOF -> 4 thrusters)
        l = 0.5
        self.B = np.array([
            [1, 1, 0, 0],     # X
            [0, 0, 1, 1],     # Y
            [0, 0, 0, 0],     # Z
            [0, 0, l, -l],    # K
            [l, -l, 0, 0],    # M
            [0, 0, l, l]      # N
        ])

        # State: [x,y,z,phi,theta,psi,u,v,w,p,q,r]
        self.state = np.zeros(12)
        self.dt = dt
        self.history = [self.state[:6].copy()]

        # Waypoints
        self.waypoints = np.array(waypoints) if waypoints is not None else np.array([[0,0,0],[50,0,-10]])
        self.current_waypoint_idx = 0
        self.waypoint_threshold = 2.0

        # PID controllers
        self.surge_pid = PID(10.0,1.0,5.0)
        self.yaw_pid   = PID(5.0,0.5,2.0)
        self.pitch_pid = PID(2.0,0.2,1.0)
        self.target_speed = 1.5

        # Battery + ocean current
        self.battery = 1000.0
        self.power_rate = 10.0
        self.current_strength = 0.2

        # Measurement noise
        self.R = np.eye(3) * 0.5**2

        # History for ROS visualization
        self.history_noisy = []

    # ---------------- Mass & Dynamics ----------------
    def MRB(self): return np.diag([self.m,self.m,self.m,self.Ix,self.Iy,self.Iz])
    def MA(self): return -np.diag([self.Xudot,self.Yvdot,self.Zwdot,self.Kpdot,self.Mqdot,self.Nrdot])
    def M(self): return self.MRB() + self.MA()

    def CRB(self, nu):
        u,v,w,p,q,r = nu
        S = np.array([[0,-r,q],[r,0,-p],[-q,p,0]])
        CRB_upper = self.m * S
        CRB_lower = np.zeros((3,3))
        return np.block([[CRB_upper, CRB_lower],[CRB_lower, CRB_upper]])

    def CA(self, nu):
        u,v,w,p,q,r = nu
        Xu,Yv,Zw,Kp,Mq,Nr = self.Xudot,self.Yvdot,self.Zwdot,self.Kpdot,self.Mqdot,self.Nrdot
        CA = np.zeros((6,6))
        CA[0,4] = Zw*w; CA[0,5] = -Yv*v
        CA[1,3] = -Zw*w; CA[1,5] = Xu*u
        CA[2,3] = Yv*v; CA[2,4] = -Xu*u
        CA[3,1] = -Zw*w; CA[3,2] = Yv*v
        CA[4,0] = Zw*w; CA[4,2] = -Xu*u
        CA[5,0] = -Yv*v; CA[5,1] = Xu*u
        return -0.5*(CA + CA.T)

    def C(self, nu): return self.CRB(nu)+self.CA(nu)

    def D(self, nu):
        u,v,w,p,q,r = nu
        lin = np.diag([self.Xu,self.Yv,self.Zw,self.Kp,self.Mq,self.Nr])
        quad = np.diag([self.Xuu*abs(u),self.Yvv*abs(v),self.Zww*abs(w),0,0,0])
        return lin+quad

    def g(self, eta):
        phi,theta,_ = eta[3:6]
        g_vec = np.zeros(6)
        g_vec[2] = (self.W-self.Buoy)
        g_vec[3] = -(self.W*self.zg)*np.sin(theta)
        g_vec[4] = (self.W*self.zg)*np.sin(phi)*np.cos(theta)
        return g_vec

    def J(self, eta):
        phi,theta,psi = eta[3:6]
        R = np.array([
            [np.cos(psi)*np.cos(theta), -np.sin(psi)*np.cos(phi)+np.cos(psi)*np.sin(theta)*np.sin(phi), np.sin(psi)*np.sin(phi)+np.cos(psi)*np.cos(phi)*np.sin(theta)],
            [np.sin(psi)*np.cos(theta), np.cos(psi)*np.cos(phi)+np.sin(psi)*np.sin(theta)*np.sin(phi), -np.cos(psi)*np.sin(phi)+np.sin(psi)*np.cos(phi)*np.sin(theta)],
            [-np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)]
        ])
        T = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])
        return np.block([[R,np.zeros((3,3))],[np.zeros((3,3)),T]])

    # ---------------- Ocean currents ----------------
    def get_current(self, position, t):
        cx = self.current_strength * np.sin(0.1*position[1]+0.05*t)
        cy = self.current_strength * np.cos(0.1*position[0]+0.05*t)
        cz = 0.05 * self.current_strength
        return np.array([cx,cy,cz])

    # ---------------- 6DOF dynamics ----------------
    def dynamics(self,t,state,tau):
        eta,nu = state[:6], state[6:]
        M_inv = np.linalg.inv(self.M())
        eta_dot = self.J(eta) @ nu
        nu_dot = M_inv @ (tau - self.C(nu)@nu - self.D(nu)@nu - self.g(eta))
        return np.concatenate((eta_dot,nu_dot))

    # ---------------- Waypoint navigation ----------------
    def get_target(self):
        if self.current_waypoint_idx >= len(self.waypoints):
            return self.state[:3], 0.0, self.state[2]
        target_pos = self.waypoints[self.current_waypoint_idx]
        dx,dy = target_pos[0]-self.state[0], target_pos[1]-self.state[1]
        target_yaw = np.arctan2(dy,dx)
        return target_pos,target_yaw,target_pos[2]

    # ---------------- PID + Thruster control ----------------
    def control(self):
        target_pos,target_yaw,target_depth = self.get_target()
        surge_error = self.target_speed - self.state[6]
        delta_surge = self.surge_pid.compute(surge_error,self.dt)
        yaw_error = target_yaw - self.state[5]
        delta_yaw = self.yaw_pid.compute(yaw_error,self.dt)
        dist = np.linalg.norm(target_pos-self.state[:3])
        pitch_error = np.arcsin((target_depth-self.state[2])/dist)-self.state[4] if dist>1e-6 else 0.0
        delta_pitch = self.pitch_pid.compute(pitch_error,self.dt)

        desired_tau = np.array([delta_surge,0,0,0,delta_pitch,delta_yaw])
        B_pinv = np.linalg.pinv(self.B)
        thruster_inputs = B_pinv @ desired_tau
        return np.clip(thruster_inputs,-100,100)

    # ---------------- Step simulation ----------------
    def step(self, t):
        tau = self.control()
        sol = solve_ivp(lambda tt,yy: self.dynamics(tt,yy,tau), [0,self.dt], self.state, method='RK45', t_eval=[self.dt])
        self.state = sol.y[:,-1]
        self.histo


