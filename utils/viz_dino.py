import zmq
import numpy as np
import matplotlib.pyplot as plt

class QuadDinoVisualizer:
    def __init__(self, cmap='jet'):
        plt.ion()
        # 2행 2열의 그리드 생성 (Row 1: 원본 이미지, Row 2: 히트맵)
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 10))

        # 더미 데이터 (240x320 크기 가정)
        dummy_img = np.zeros((240, 320, 3))
        dummy_hm = np.zeros((240, 320))

        # --- [1행] 원본 이미지 ---
        self.im_img1 = self.axs[0, 0].imshow(dummy_img)
        self.axs[0, 0].set_title("Room 1 : Original Image")
        self.axs[0, 0].axis('off')

        self.im_img2 = self.axs[0, 1].imshow(dummy_img)
        self.axs[0, 1].set_title("Room 2 : Original Image")
        self.axs[0, 1].axis('off')

        # --- [2행] 히트맵 ---
        self.im_hm1 = self.axs[1, 0].imshow(dummy_hm, cmap=cmap, interpolation='bilinear')
        # self.fig.colorbar(self.im_hm1, ax=self.axs[1, 0], fraction=0.046, pad=0.04)
        self.axs[1, 0].set_title("Room 1 : DINO Heatmap")
        self.axs[1, 0].axis('off')

        self.im_hm2 = self.axs[1, 1].imshow(dummy_hm, cmap=cmap, interpolation='bilinear')
        # self.fig.colorbar(self.im_hm2, ax=self.axs[1, 1], fraction=0.046, pad=0.04)
        self.axs[1, 1].set_title("Room 2 : DINO Heatmap")
        self.axs[1, 1].axis('off')

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _process_image(self, img):
        """[B, C, H, W] 텐서를 Matplotlib용 [H, W, C]로 변환 및 정규화"""
        img = np.asarray(img)

        # 1. 배치 차원 제거 [1, C, H, W] -> [C, H, W]
        if img.ndim == 4:
            img = img.squeeze(0)

        # 2. 채널 차원 이동 [C, H, W] -> [H, W, C]
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        # 3. 값 정규화 (Matplotlib은 float일 경우 0~1 범위를 요구함)
        if img.dtype != np.uint8:
            if img.max() > 2.0:  # 0~255 스케일인 경우
                img = img / 255.0
            else:  # 이미 0~1 이거나 정규화된 음수 값이 있는 경우 대비 안전한 클리핑
                img = np.clip(img, 0, 1)

        return img

    def _process_heatmap(self, heatmap):
        """[B, H, W] 텐서를 [H, W]로 변환"""
        h = np.asarray(heatmap)
        if h.ndim == 3:
            if h.shape[0] == 1:
                h = h.squeeze(0)
            elif h.shape[-1] == 1:
                h = h.squeeze(-1)
        return h

    def update(self, img1, img2, hm1, hm2):
        try:
            # 1. 데이터 차원 정리 및 전처리
            p_img1 = self._process_image(img1)
            p_img2 = self._process_image(img2)
            p_hm1 = self._process_heatmap(hm1)
            p_hm2 = self._process_heatmap(hm2)

            # 2. 원본 이미지 업데이트
            self.im_img1.set_data(p_img1)
            self.im_img2.set_data(p_img2)

            # 3. 히트맵 데이터 및 컬러맵 스케일 업데이트
            self.im_hm1.set_data(p_hm1)
            if p_hm1.min() != p_hm1.max():
                self.im_hm1.set_clim(vmin=p_hm1.min(), vmax=p_hm1.max())

            self.im_hm2.set_data(p_hm2)
            if p_hm2.min() != p_hm2.max():
                self.im_hm2.set_clim(vmin=p_hm2.min(), vmax=p_hm2.max())

            # 화면 갱신
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        except Exception as e:
            print(f"Visualization Error: {e}")


def main():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.CONFLATE, 1)  # 딜레이 방지를 위해 최신 데이터만 유지
    socket.bind("tcp://*:5558")

    viz = QuadDinoVisualizer(cmap='jet')

    while True:
        try:
            if socket.poll(timeout=10):
                # 4개의 요소를 담은 튜플 수신 (img1, img2, hm1, hm2)
                data = socket.recv_pyobj()

                if isinstance(data, tuple) and len(data) == 4:
                    viz.update(data[0], data[1], data[2], data[3])
                else:
                    pass

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Server Error: {e}")


if __name__ == "__main__":
    main()