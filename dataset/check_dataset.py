import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def extract_image_timestamps(img_dir):
    """하위 폴더 포함 모든 jpg, png 파일에서 타임스탬프 추출"""
    img_files = glob.glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True)
    img_files += glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True)
    
    ts_list = []
    for f in img_files:
        try:
            ts_str = os.path.basename(f).split('_')[-1].replace('.jpg', '').replace('.png', '')
            ts_list.append(float(ts_str))
        except: pass
    return np.array(sorted(ts_list))

def calc_sync_stats(img_ts_norm, enc_ts_norm):
    if len(img_ts_norm) < 2 or len(enc_ts_norm) < 2:
        return 0, 0, 0, 0, 0
    hz = 1.0 / np.mean(np.diff(img_ts_norm))
    jitter_ms = np.std(np.diff(img_ts_norm)) * 1000
    delays = []
    for i_ts in img_ts_norm:
        idx = np.searchsorted(enc_ts_norm, i_ts, side="left")
        if idx == 0: nearest_ts = enc_ts_norm[0]
        elif idx == len(enc_ts_norm): nearest_ts = enc_ts_norm[-1]
        else:
            left, right = enc_ts_norm[idx-1], enc_ts_norm[idx]
            nearest_ts = left if (i_ts - left) < (right - i_ts) else right
        delays.append(i_ts - nearest_ts)
    delays_ms = np.array(delays) * 1000
    return hz, jitter_ms, np.mean(delays_ms), np.max(np.abs(delays_ms)), np.std(delays_ms)

def generate_ultimate_report(
    base_dir=".",
    output_pdf="total_timeline_report.pdf",
    bad_txt_path="bad_episodes_list.txt",
):
    print("==================================================")
    print(" 🕵️‍♂️ Ultimate Time Sync Checker & Bad Data Finder")
    print("==================================================\n")

    # 💡 [핵심] 하위 폴더까지 싹 다 뒤져서 episode로 시작하는 폴더를 모두 찾음
    search_path = os.path.join(base_dir, "**", "episode_*")
    episode_dirs = sorted(glob.glob(search_path, recursive=True))
    
    if not episode_dirs:
        print(f"❌ '{search_path}' 경로에서 에피소드를 찾을 수 없습니다.")
        return

    bad_episodes = [] # 불량 폴더 경로를 저장할 리스트
    
    print(f"총 {len(episode_dirs)}개의 에피소드 분석을 시작합니다...\n")

    with PdfPages(output_pdf) as pdf:
        for i, ep_dir in enumerate(episode_dirs):
            ep_full_path = os.path.abspath(ep_dir) # 💡 절대 경로 추출
            ep_name = os.path.basename(ep_dir)
            parent_name = os.path.basename(os.path.dirname(ep_full_path)) # 부모 폴더명 (예: set001)
            
            print(f"[{i+1}/{len(episode_dirs)}] 분석 중: {parent_name}/{ep_name} ...", end=" ")

            enc_csv = os.path.join(ep_dir, "encoder.csv")
            room1_dir = os.path.join(ep_dir, "image", "room1")
            room2_dir = os.path.join(ep_dir, "image", "room2")

            try:
                df_enc = pd.read_csv(enc_csv, names=['ts', 'vx', 'wz'], header=None)
                df_enc['ts'] = pd.to_numeric(df_enc['ts'], errors='coerce')
                enc_ts = df_enc.dropna(subset=['ts'])['ts'].values
            except:
                print("⚠️ 인코더 로드 실패 (스킵)")
                bad_episodes.append(f"[데이터 파손] {ep_full_path}")
                continue

            img1_ts = extract_image_timestamps(room1_dir)
            img2_ts = extract_image_timestamps(room2_dir)

            if len(enc_ts) < 2 or (len(img1_ts) == 0 and len(img2_ts) == 0):
                print("⚠️ 이미지 데이터 부족 (스킵)")
                bad_episodes.append(f"[이미지 누락] {ep_full_path}")
                continue

            # 0초 정규화
            all_starts = [enc_ts[0]]
            if len(img1_ts) > 0: all_starts.append(img1_ts[0])
            if len(img2_ts) > 0: all_starts.append(img2_ts[0])
            global_start = min(all_starts)

            enc_norm = enc_ts - global_start
            img1_norm = img1_ts - global_start if len(img1_ts) > 0 else np.array([])
            img2_norm = img2_ts - global_start if len(img2_ts) > 0 else np.array([])

            enc_hz = 1.0 / np.mean(np.diff(enc_norm)) if len(enc_norm) > 1 else 0
            hz1, jit1, mean1, max1, _ = calc_sync_stats(img1_norm, enc_norm)
            hz2, jit2, mean2, max2, _ = calc_sync_stats(img2_norm, enc_norm)

            # 🚨 불량 데이터 판별 로직 (Delay 50ms 초과, Jitter 20ms 초과, 또는 FPS가 10 미만인 경우)
            is_danger = abs(mean1)>50 or abs(mean2)>50 or jit1>20 or jit2>20 or hz1<10 or hz2<10
            if is_danger:
                bad_episodes.append(f"[싱크/지연 불량] {ep_full_path}")

            # ================= 시각화 =================
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            
            # 💡 [핵심] 제목 아래에 절대 경로(Absolute Path) 추가
            title_str = f"[{parent_name}/{ep_name}] Dual Camera Sync Report\n"
            title_str += f"Path: {ep_full_path}"
            fig.suptitle(title_str, fontsize=15, fontweight='bold', color='darkred' if is_danger else 'black')

            stats_text = (
                f"📊 [System Statistics]\n"
                f"Encoder : {enc_hz:.1f} Hz\n"
                f"----------------------------------------\n"
                f"[Room 1 (Top)]\n"
                f"FPS: {hz1:.1f} Hz | Jitter: {jit1:.1f} ms\n"
                f"Delay -> Mean: {mean1:+.1f} ms | Max: {max1:.1f} ms\n"
                f"----------------------------------------\n"
                f"[Room 2 (Bottom)]\n"
                f"FPS: {hz2:.1f} Hz | Jitter: {jit2:.1f} ms\n"
                f"Delay -> Mean: {mean2:+.1f} ms | Max: {max2:.1f} ms"
            )
            
            box_color = 'lightcoral' if is_danger else 'lightgreen'
            fig.text(0.77, 0.82, stats_text, fontsize=10, family='monospace',
                     bbox=dict(facecolor=box_color, alpha=0.6, edgecolor='black', boxstyle='round,pad=0.5'))

            y_ticks = [0.4, 1.4, 2.4]
            y_labels = ['Encoder\n(Velocity)', 'Room 2\n(Bottom Cam)', 'Room 1\n(Top Cam)']

            # [Plot 1] Macro View
            ax1.vlines(enc_norm, 0.0, 0.8, color='dodgerblue', alpha=0.5, linewidth=1)
            if len(img2_norm) > 0: ax1.vlines(img2_norm, 1.0, 1.8, color='mediumseagreen', alpha=0.8, linewidth=1.5)
            if len(img1_norm) > 0: ax1.vlines(img1_norm, 2.0, 2.8, color='darkorange', alpha=0.8, linewidth=1.5)
            ax1.set_yticks(y_ticks); ax1.set_yticklabels(y_labels, fontsize=11, fontweight='bold')
            ax1.set_title("Macro View: Full Episode Timeline")
            ax1.grid(axis='x', linestyle='--', alpha=0.7)

            total_duration = max([enc_norm[-1] if len(enc_norm) else 0, img1_norm[-1] if len(img1_norm) else 0, img2_norm[-1] if len(img2_norm) else 0])
            zoom_duration = 2.0
            zoom_start = max(0, (total_duration / 2.0) - (zoom_duration / 2.0))
            ax1.axvspan(zoom_start, zoom_start + zoom_duration, color='gray', alpha=0.2)

            # [Plot 2] Micro View
            ax2.vlines(enc_norm, 0.0, 0.8, color='dodgerblue', alpha=0.7, linewidth=2)
            if len(img2_norm) > 0: ax2.vlines(img2_norm, 1.0, 1.8, color='mediumseagreen', alpha=1.0, linewidth=3)
            if len(img1_norm) > 0: ax2.vlines(img1_norm, 2.0, 2.8, color='darkorange', alpha=1.0, linewidth=3)

            for img_norm, base_y, color in [(img2_norm, 1.0, 'green'), (img1_norm, 2.0, 'red')]:
                for i_ts in img_norm:
                    if i_ts < zoom_start or i_ts > zoom_start + zoom_duration: continue
                    idx = np.searchsorted(enc_norm, i_ts, side="left")
                    if idx == 0: nearest_enc = enc_norm[0]
                    elif idx == len(enc_norm): nearest_enc = enc_norm[-1]
                    else:
                        left, right = enc_norm[idx-1], enc_norm[idx]
                        nearest_enc = left if (i_ts - left) < (right - i_ts) else right
                    ax2.plot([i_ts, nearest_enc], [base_y, 0.8], color=color, linestyle=':', alpha=0.6)

            ax2.set_yticks(y_ticks); ax2.set_yticklabels(y_labels, fontsize=11, fontweight='bold')
            ax2.set_xlim(zoom_start, zoom_start + zoom_duration)
            ax2.set_title(f"Micro View: {zoom_start:.1f}s ~ {zoom_start + zoom_duration:.1f}s")
            ax2.set_xlabel("Time (Seconds)"); ax2.grid(axis='x', linestyle='--', alpha=0.7)

            plt.tight_layout(rect=[0, 0.03, 0.75, 0.90]) # 제목 공간 확보를 위해 0.90으로 수정
            pdf.savefig(fig)
            plt.close(fig)
            print("완료!")

    # ================= 결과 리포트 출력 및 저장 =================
    print("\n==================================================")
    print(f" 🎉 모든 폴더 탐색 완료! 종합 PDF 확인: {output_pdf}")
    print("==================================================")
    
    with open(bad_txt_path, "w") as f:
        if bad_episodes:
            print(f"🚨 총 {len(bad_episodes)}개의 불량 에피소드(사용 불가 데이터)가 발견되었습니다!")
            f.write("=== 삭제 또는 제외 권장 에피소드 리스트 ===\n")
            for bad_ep in bad_episodes:
                print(f"  - {bad_ep}")
                f.write(f"{bad_ep}\n")
            print(f"\n📂 불량 리스트가 텍스트 파일로 저장되었습니다: {bad_txt_path}")
            print("💡 Zarr 생성 스크립트에서 이 파일의 경로를 읽어 자동으로 스킵하도록 코딩하시면 편리합니다.")
        else:
            print("✨ 축하합니다! 모든 데이터가 아주 깨끗합니다.")
            f.write("불량 에피소드 없음.\n")

if __name__ == "__main__":
    import sys

    ROOT_DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
    out_pdf = sys.argv[2] if len(sys.argv) > 2 else "total_timeline_report.pdf"
    out_bad = sys.argv[3] if len(sys.argv) > 3 else "bad_episodes_list.txt"
    generate_ultimate_report(ROOT_DATA_DIR, output_pdf=out_pdf, bad_txt_path=out_bad)