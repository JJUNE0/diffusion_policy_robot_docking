import torch
import torch.nn.functional as F
import cv2
import numpy as np
from transformers import AutoModel
import os
import matplotlib.pyplot as plt

class DinoBatchDetector:
    def __init__(self, device, master_vector_path="master_vector.pt"):
        self.device = device
        # DINOv3 백본 로드
        self.model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m").to(device)
        self.model.eval()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, master_vector_path)

        if not os.path.exists(full_path):
            full_path = os.path.abspath(master_vector_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Master vector not found. Checked: {full_path}")

        print(f"Loading master vector from: {full_path}")
        self.master_vector = torch.load(full_path, map_location=device, weights_only=True)

        # 검증용 색상 범위 (HSV)
        self.lower_green = np.array([35, 75, 60])
        self.upper_green = np.array([85, 255, 255])
        self.lower_white = np.array([0, 0, 190])
        self.upper_white = np.array([180, 55, 255])

        # 정규화 파라미터
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def get_heatmap(self, img_tensor):
        """
        img_tensor: [B, 3, 240, 320]
        Returns:
            patches: [B, 196, 768] (DINO Raw Features) 💡 추가됨
            final_sim_maps: [B, 240, 320] Tensor
            has_objects: [B] Tensor (1.0 for detected, 0.0 for not)
        """
        b, c, h, w = img_tensor.shape

        # 1. DINO 특징 추출
        x = F.interpolate(img_tensor, size=(224, 224), mode='bicubic', align_corners=False)
        x = (x - self.mean) / self.std
        outputs = self.model(x)

        # [B, 196, 768] - 나중에 모델에서 재사용할 원본 패치 특징
        raw_patches = outputs.last_hidden_state[:, 5:, :]

        # 유사도 계산용 정규화 패치
        patches_norm = F.normalize(raw_patches, p=2, dim=-1)

        # 2. 유사도 맵(Heatmap) 계산
        side_size = int(np.sqrt(patches_norm.shape[1]))
        sim_maps = torch.matmul(patches_norm, self.master_vector.reshape(-1, 1)).squeeze(-1)
        sim_maps = sim_maps.view(b, side_size, side_size)

        # [B, 1, 14, 14] -> [B, 240, 320] 리사이징
        sim_maps_resized = F.interpolate(sim_maps.unsqueeze(1), size=(h, w), mode='bilinear').squeeze(1)

        # 3. 개별 이미지 검증 (OpenCV)
        sim_maps_np = sim_maps_resized.cpu().numpy()
        images_np = (img_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

        final_batch_sim = []
        final_batch_has_obj = []

        for i in range(b):
            refined_sim, has_obj = self._verify_single(images_np[i], sim_maps_np[i], h, w)
            final_batch_sim.append(refined_sim)
            final_batch_has_obj.append(float(has_obj))

        # 4. 결과 반환 (특징 맵 포함)
        return raw_patches, \
            torch.from_numpy(np.stack(final_batch_sim)).to(self.device), \
            torch.tensor(final_batch_has_obj).to(self.device)

    def _verify_single(self, img_rgb, s_map, h, w):
        original_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        s_map_smoothed = cv2.GaussianBlur(s_map, (0, 0), sigmaX=5)

        temp_sim_map = s_map_smoothed.copy()
        is_detected = False
        best_loc = (0, 0)
        max_val_found = 0.0

        for _ in range(5):
            _, cur_val, _, cur_loc = cv2.minMaxLoc(temp_sim_map)
            r_s = 40
            x_s, y_s = max(0, cur_loc[0] - r_s // 2), max(0, cur_loc[1] - r_s // 2)
            x_e, y_e = min(w, cur_loc[0] + r_s // 2), min(h, cur_loc[1] + r_s // 2)
            roi_hsv = cv2.cvtColor(original_bgr[y_s:y_e, x_s:x_e], cv2.COLOR_BGR2HSV)

            mask_g = cv2.inRange(roi_hsv, self.lower_green, self.upper_green)
            mask_w = cv2.inRange(roi_hsv, self.lower_white, self.upper_white)
            refined_mask = cv2.bitwise_and(mask_g, cv2.bitwise_not(mask_w))
            g_ratio = np.sum(refined_mask > 0) / (roi_hsv.shape[0] * roi_hsv.shape[1] + 1e-6)

            if cur_val > 0.45 and g_ratio > 0.10:
                is_detected = True
                best_loc = cur_loc
                max_val_found = cur_val
                break
            cv2.circle(temp_sim_map, cur_loc, 60, 0, -1)

        if is_detected:
            y_range, x_range = np.ogrid[:h, :w]
            dist_sq = (x_range - best_loc[0]) ** 2 + (y_range - best_loc[1]) ** 2
            focus_mask = np.exp(-dist_sq / (2 * 150 ** 2))
            refined_sim = np.clip((s_map_smoothed * focus_mask - 0.4) / (max_val_found - 0.4 + 1e-8), 0, 1)
        else:
            refined_sim = np.zeros((h, w), dtype=np.float32)

        return refined_sim, is_detected


def main():
    # 1. 환경 설정 및 디렉토리 생성
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)

    # 마스터 벡터 경로 설정 (실제 경로에 맞춰 수정 필요)
    master_vector_path = "master_vector.pt"

    try:
        detector = DinoBatchDetector(device=device, master_vector_path=master_vector_path)
    except FileNotFoundError as e:
        print(f"❌ 에러: {e}")
        return

    # 2. npy 데이터 로드
    input_path = "img1.npy"
    if not os.path.exists(input_path):
        print(f"❌ 에러: {input_path} 파일이 없습니다.")
        return

    # npy 로드 (보통 [B, 3, H, W] 또는 [3, H, W] 형태)
    img_array = np.load(input_path)
    img_tensor = torch.from_numpy(img_array).to(device).float()

    # 0~255 범위라면 0~1로 정규화 (DINO 입력용)
    if img_tensor.max() > 1.0:
        img_tensor /= 255.0

    if img_tensor.ndim == 3:  # 단일 이미지인 경우 배치 차원 추가
        img_tensor = img_tensor.unsqueeze(0)

    # 3. 추론 실행
    print(f"🚀 추론 시작... (입력 크기: {img_tensor.shape})")
    raw_patches, sim_maps, has_objs = detector.get_heatmap(img_tensor)

    # 4. 결과 시각화 및 저장
    # 시각화를 위해 CPU 넘파이 배열로 변환 [B, H, W, 3]
    images_np = (img_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    sim_maps_np = sim_maps.cpu().numpy()
    has_objs_np = has_objs.cpu().numpy()

    print(f"📊 결과 저장 중: '{output_dir}' 폴더를 확인하세요.")

    for i in range(len(images_np)):
        res_img = images_np[i].copy()
        is_detected = has_objs_np[i] > 0.5
        status = "DET" if is_detected else "NEG"

        # 검출된 경우에만 히트맵 합성 (OpenCV 사용)
        if is_detected:
            # sim_map을 0~255로 변환 후 컬러맵 적용
            heatmap_color = cv2.applyColorMap((sim_maps_np[i] * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # OpenCV(BGR) -> Matplotlib/일반(RGB) 변환
            heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            # 유사도가 어느 정도 있는 영역(0.1 이상)만 원본과 합성
            mask = sim_maps_np[i] > 0.1
            res_img[mask] = cv2.addWeighted(images_np[i][mask], 0.5, heatmap_rgb[mask], 0.5, 0)

        # Matplotlib을 이용한 결과 저장
        plt.figure(figsize=(8, 6))
        plt.imshow(res_img)
        plt.title(f"Frame {i:03d} - Status: {status} (Score: {has_objs_np[i]:.2f})")
        plt.axis('off')

        save_name = os.path.join(output_dir, f"result_{i:03d}_{status}.png")
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()  # 메모리 해제

    print(f"✅ 완료! 총 {len(images_np)}개의 이미지를 처리했습니다.")


if __name__ == "__main__":
    main()