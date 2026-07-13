"""
iq_scatter.py
สร้าง IQ scatter image จาก raw CSI data
รองรับหลาย feature types:
  - phase_norm: unit circle normalized (invariant ต่อ amplitude)
  - dphase:     differential phase (CFO removed)
  - conj:       conjugate product (CFO removed)
  - raw:        raw IQ diff
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMG_SIZE  = 30
N_FRAMES  = 20
input_dir = "data/"

# ── PARSE CSI ─────────────────────────────────────────────────────────────────
def parse_csi_raw(csi_str):
    """parse CSI string → I, Q arrays (45 subcarriers)"""
    vals  = list(map(float, csi_str.strip().split()))
    I_all = vals[0::2]
    Q_all = vals[1::2]
    # trim 45 subcarriers (21 neg + 24 pos)
    I_neg = I_all[33+5 : 33+5+21]
    I_pos = I_all[1:25]
    Q_neg = Q_all[33+5 : 33+5+21]
    Q_pos = Q_all[1:25]
    return np.array(I_neg + I_pos), np.array(Q_neg + Q_pos)  # (45,)

# ── LOAD FRAMES ───────────────────────────────────────────────────────────────
def load_frames(pt, start=0, n=N_FRAMES, dir=input_dir):
    """โหลด n frames จาก CSV → I1, Q1, I2, Q2 stacks"""
    path_lt = os.path.join(dir, f"{pt}_lt_01.csv")
    path_rt = os.path.join(dir, f"{pt}_rt_01.csv")

    df_lt = pd.read_csv(path_lt, header=0)
    df_rt = pd.read_csv(path_rt, header=0)

    I1l, Q1l, I2l, Q2l = [], [], [], []
    for i in range(start, min(start+n, len(df_lt), len(df_rt))):
        try:
            I1, Q1 = parse_csi_raw(df_lt['CSI_VALUES'].iloc[i])
            I2, Q2 = parse_csi_raw(df_rt['CSI_VALUES'].iloc[i])
            I1l.append(I1); Q1l.append(Q1)
            I2l.append(I2); Q2l.append(Q2)
        except:
            pass

    return (np.array(I1l), np.array(Q1l),
            np.array(I2l), np.array(Q2l))   # each (n, 45)

# ── FEATURE FUNCTIONS ─────────────────────────────────────────────────────────

def make_phase_norm_image(I1s, Q1s, I2s, Q2s, size=IMG_SIZE):
    """
    Phase normalized (unit circle)
    normalize I/Q ให้อยู่บน unit circle ก่อน plot
    invariant ต่อ amplitude → ลด effect ของระยะห่าง
    """
    img   = np.zeros((size, size), dtype=np.float32)
    a1    = np.sqrt(I1s**2 + Q1s**2) + 1e-6
    a2    = np.sqrt(I2s**2 + Q2s**2) + 1e-6
    I_dn  = I1s/a1 - I2s/a2   # normalized diff I
    Q_dn  = Q1s/a1 - Q2s/a2   # normalized diff Q
    scale = (size - 1) / 2.0

    for fi in range(I_dn.shape[0]):
        for si in range(I_dn.shape[1]):
            px = int((I_dn[fi,si] + 1.0) * scale)
            py = int((Q_dn[fi,si] + 1.0) * scale)
            img[np.clip(py,0,size-1), np.clip(px,0,size-1)] += 1.0

    return img / (img.max() + 1e-6)

def make_dphase_image(I1s, Q1s, I2s, Q2s, size=IMG_SIZE):
    """
    Differential phase (CFO REMOVED)
    phase(H1/H2)[k+1] - phase(H1/H2)[k]
    CFO เท่ากันทุก subcarrier → diff หักออกหมด
    """
    img = np.zeros((size, size), dtype=np.float32)

    # phase ของ H1/H2 ต่อ subcarrier ต่อ frame
    phase   = np.arctan2(Q1s*I2s - I1s*Q2s,
                         I1s*I2s + Q1s*Q2s)   # (n, 45)
    d_phase = np.diff(phase, axis=1)           # (n, 44)
    d_norm  = (d_phase + np.pi) / (2*np.pi)   # normalize 0..1

    scale_x = (size - 1) / 43.0
    for fi in range(d_norm.shape[0]):
        for si in range(d_norm.shape[1]):
            px = int(si * scale_x)
            py = int(d_norm[fi,si] * (size-1))
            img[np.clip(py,0,size-1), np.clip(px,0,size-1)] += 1.0

    return img / (img.max() + 1e-6)

def make_conj_image(I1s, Q1s, I2s, Q2s, size=IMG_SIZE):
    """
    Conjugate product H1 * conj(H2) normalized (CFO REMOVED)
    Real = I1*I2 + Q1*Q2
    Imag = Q1*I2 - I1*Q2
    CFO cancel: e^(jφ) × e^(-jφ) = 1
    """
    img   = np.zeros((size, size), dtype=np.float32)
    real  = I1s*I2s + Q1s*Q2s
    imag  = Q1s*I2s - I1s*Q2s
    amp   = np.sqrt(real**2 + imag**2) + 1e-6
    rn    = real / amp   # cos(θ1-θ2)
    im_n  = imag / amp   # sin(θ1-θ2)
    scale = (size - 1) / 2.0

    for fi in range(rn.shape[0]):
        for si in range(rn.shape[1]):
            px = int((rn[fi,si]   + 1.0) * scale)
            py = int((im_n[fi,si] + 1.0) * scale)
            img[np.clip(py,0,size-1), np.clip(px,0,size-1)] += 1.0

    return img / (img.max() + 1e-6)

def make_raw_image(I1s, Q1s, I2s, Q2s, size=IMG_SIZE):
    """
    Raw IQ diff (I1-I2, Q1-Q2)
    ไม่ normalize — เห็น amplitude ด้วย
    """
    img   = np.zeros((size, size), dtype=np.float32)
    I_dr  = I1s - I2s
    Q_dr  = Q1s - Q2s
    scale = (size - 1) / 256.0

    for fi in range(I_dr.shape[0]):
        for si in range(I_dr.shape[1]):
            px = int((I_dr[fi,si] + 128) * scale)
            py = int((Q_dr[fi,si] + 128) * scale)
            img[np.clip(py,0,size-1), np.clip(px,0,size-1)] += 1.0

    return img / (img.max() + 1e-6)

def make_amplitude_image(I1s, Q1s, I2s, Q2s, size=IMG_SIZE):
    """
    Amplitude ratio per subcarrier × frame
    x = subcarrier index, y = frame index, value = ratio
    """
    amp1  = np.sqrt(I1s**2 + Q1s**2)
    amp2  = np.sqrt(I2s**2 + Q2s**2) + 1e-6
    ratio = amp1 / amp2   # (n_frames, 45)

    # resize เป็น size×size
    from PIL import Image
    ratio_norm = ratio / (ratio.max() + 1e-6)
    img_pil    = Image.fromarray((ratio_norm * 255).astype(np.uint8))
    img_resized = img_pil.resize((size, size), Image.BILINEAR)
    return np.array(img_resized, dtype=np.float32) / 255.0

# ── COMPUTE ALL FEATURES ──────────────────────────────────────────────────────
def compute_all(pt, start=0, n=N_FRAMES, dir=input_dir):
    """คำนวณ feature ทุกชนิดสำหรับ 1 sample"""
    I1s, Q1s, I2s, Q2s = load_frames(pt, start, n, dir)
    return {
        'phase_norm': make_phase_norm_image(I1s, Q1s, I2s, Q2s),
        'dphase':     make_dphase_image(I1s, Q1s, I2s, Q2s),
        'conj':       make_conj_image(I1s, Q1s, I2s, Q2s),
        'raw':        make_raw_image(I1s, Q1s, I2s, Q2s),
        'amplitude':  make_amplitude_image(I1s, Q1s, I2s, Q2s),
    }

# ── FLATTEN สำหรับ ML ─────────────────────────────────────────────────────────
def get_feature_vector(pt, feature_type='conj',
                       start=0, n=N_FRAMES, dir=input_dir):
    """
    คืน flat vector สำหรับใช้กับ KNN/RF
    feature_type: 'phase_norm', 'dphase', 'conj', 'raw', 'amplitude'
    """
    I1s, Q1s, I2s, Q2s = load_frames(pt, start, n, dir)
    funcs = {
        'phase_norm': make_phase_norm_image,
        'dphase':     make_dphase_image,
        'conj':       make_conj_image,
        'raw':        make_raw_image,
        'amplitude':  make_amplitude_image,
    }
    img = funcs[feature_type](I1s, Q1s, I2s, Q2s)
    return img.flatten()  # 900 values

# ── VISUALIZE ─────────────────────────────────────────────────────────────────
def visualize_point(pt, start=0, n=N_FRAMES,
                    dir=input_dir, save=None):
    """วาด IQ scatter ทุกชนิดสำหรับ 1 จุด"""
    features = compute_all(pt, start, n, dir)

    cmaps = {
        'phase_norm': 'Blues',
        'dphase':     'RdYlGn',
        'conj':       'plasma',
        'raw':        'hot',
        'amplitude':  'inferno',
    }
    titles = {
        'phase_norm': 'Phase Norm\n(unit circle)',
        'dphase':     'Diff Phase\n(CFO removed)',
        'conj':       'Conjugate Product\n(CFO removed)',
        'raw':        'Raw IQ Diff',
        'amplitude':  'Amplitude Ratio\n(subcarrier×frame)',
    }

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f"IQ Scatter — Point: {pt}  "
                 f"({n} frames, {IMG_SIZE}×{IMG_SIZE})",
                 fontsize=13, fontweight='bold')

    for ax, (name, img) in zip(axes, features.items()):
        ax.imshow(img, cmap=cmaps[name], aspect='equal',
                  interpolation='nearest')
        ax.set_title(titles[name], fontsize=10)
        ax.axis('off')
        # colorbar
        plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"Saved: {save}")
    else:
        plt.show()
    plt.close()

def visualize_compare(points, feature_type='conj',
                      dir=input_dir, save=None):
    """เปรียบเทียบ IQ scatter ของหลายจุดในชนิดเดียวกัน"""
    n_pts  = len(points)
    fig, axes = plt.subplots(1, n_pts,
                             figsize=(4*n_pts, 4))
    if n_pts == 1:
        axes = [axes]

    fig.suptitle(f"IQ Scatter Comparison — {feature_type}",
                 fontsize=13, fontweight='bold')

    for ax, pt in zip(axes, points):
        try:
            vec = get_feature_vector(pt, feature_type, dir=dir)
            img = vec.reshape(IMG_SIZE, IMG_SIZE)
            ax.imshow(img, cmap='plasma', aspect='equal',
                      interpolation='nearest')
            ax.set_title(f"{pt}", fontsize=10)
            ax.axis('off')
        except Exception as e:
            ax.set_title(f"{pt}\nError: {e}", fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"Saved: {save}")
    else:
        plt.show()
    plt.close()

def visualize_all_points_all_features(points,
                                      dir=input_dir,
                                      save=None):
    """grid: rows=points, cols=feature types"""
    feature_types = ['phase_norm','dphase','conj','raw','amplitude']
    cmaps = ['Blues','RdYlGn','plasma','hot','inferno']

    n_pts  = len(points)
    n_feat = len(feature_types)
    fig    = plt.figure(figsize=(4*n_feat, 4*n_pts))
    gs     = gridspec.GridSpec(n_pts, n_feat, figure=fig,
                               hspace=0.4, wspace=0.3)

    for r, pt in enumerate(points):
        try:
            features = compute_all(pt, dir=dir)
        except Exception as e:
            print(f"Skip {pt}: {e}")
            continue

        for c, (ft, cmap) in enumerate(zip(feature_types, cmaps)):
            ax  = fig.add_subplot(gs[r, c])
            img = features[ft]
            ax.imshow(img, cmap=cmap, aspect='equal',
                      interpolation='nearest')
            if r == 0:
                ax.set_title(ft, fontsize=9, fontweight='bold')
            if c == 0:
                ax.set_ylabel(pt, fontsize=9, rotation=0,
                              labelpad=60, va='center')
            ax.axis('off')

    fig.suptitle("IQ Scatter — All Points × All Features",
                 fontsize=14, fontweight='bold', y=1.01)
    if save:
        plt.savefig(save, dpi=120, bbox_inches='tight')
        print(f"Saved: {save}")
    else:
        plt.show()
    plt.close()

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    points = ["162_134", "1762_864", "162_864", "1762_134"]

    # ── ทดสอบ 1 จุด ──────────────────────────────────────────────
    print("Generating single point visualization...")
    try:
        visualize_point("162_134",
                        save="iq_scatter_162_134.png")
    except Exception as e:
        print(f"Error: {e}")

    # ── เปรียบเทียบหลายจุด (conj) ────────────────────────────────
    print("Generating comparison visualization...")
    try:
        visualize_compare(points,
                          feature_type='conj',
                          save="iq_scatter_compare_conj.png")
    except Exception as e:
        print(f"Error: {e}")

    # ── grid ทุกจุด × ทุก feature ─────────────────────────────────
    print("Generating full grid visualization...")
    try:
        visualize_all_points_all_features(
            points,
            save="iq_scatter_grid.png")
    except Exception as e:
        print(f"Error: {e}")

    print("\nDone!")
    print("Files: iq_scatter_162_134.png")
    print("       iq_scatter_compare_conj.png")
    print("       iq_scatter_grid.png")