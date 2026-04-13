# viz_server.py
import zmq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class TrajectoryVisualizer:
    def __init__(self, horizon=16):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.horizon = horizon
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, trajectory):
        try:
            self.ax.cla()
            xs, ys = trajectory[:, 0], trajectory[:, 1]
            num_points = len(xs)
            points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            line_alphas = np.linspace(1.0, 0.1, len(segments))

            line_colors = np.zeros((len(segments), 4))
            line_colors[:, 0] = 1.0  # Red
            line_colors[:, 3] = line_alphas

            lc = LineCollection(segments, colors=line_colors, linestyle='--', linewidth=2, zorder=3)
            self.ax.add_collection(lc)

            p_alphas = np.linspace(1.0, 0.3, num_points)
            p_colors = np.zeros((num_points, 4))
            p_colors[:, 0] = 0.0  # Blue 느낌을 위해 R=0
            p_colors[:, 2] = 1.0  # Blue
            p_colors[:, 3] = p_alphas

            # scatter의 c 파라미터에 RGBA 배열을 직접 전달
            self.ax.scatter(xs, ys, c=p_colors, s=10, edgecolors='none', zorder=4)

            # 3. 현재 로봇 위치 (기준점)
            self.ax.scatter(0, 0, color='black', marker='s', s=40, label='Robot (Now)', zorder=5)

            self.ax.set_aspect('equal')
            self.ax.set_xlim([-0.075, 0.075])  # 범위를 조금 넓혔습니다. 필요시 조정하세요.
            self.ax.set_ylim([-0.075, 0.075])
            self.ax.set_xlabel("X [m]")
            self.ax.set_ylabel("Y [m]")
            self.ax.set_title(f"Live AI Trajectory")
            self.ax.grid(True, linestyle=':', alpha=0.3)

            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"Viz Error: {e}")

def run_viz_server():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)  # 데이터를 받는 포트
    socket.bind("tcp://*:5555")

    viz = TrajectoryVisualizer()
    print("Visualization Server Ready. Waiting for data...")

    while True:
        try:
            # 비차단 모드로 데이터 확인 (없으면 통과)
            if socket.poll(timeout=10):
                data = socket.recv_pyobj()  # numpy array 수신
                viz.update(data)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run_viz_server()
