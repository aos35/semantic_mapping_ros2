#!/usr/bin/env python3
"""Evalúa la trayectoria de rko_lio (output/trajectory.csv) frente a la
trayectoria de referencia del Oxford Spires Dataset (formato TUM).

    python3 tools/eval_trajectory.py --est output/trajectory.csv \
        --gt ../ground_truth/hba-tum.txt --out output/ate.png

Métrica: ATE (Absolute Trajectory Error) tras alinear con Umeyama en SE(3)
(rotación + traslación, sin escala, porque el LiDAR mide en metros).
Nota: nuestra trayectoria es la del frame del LiDAR ('pandar') y la de
referencia la del frame 'base' (0,124 m por debajo según la calibración);
la diferencia es casi constante y la absorbe la alineación.
"""
import argparse
import json

import numpy as np


def load_est(path):
    a = np.loadtxt(path, delimiter=',', skiprows=1)
    return a[:, 0], a[:, 1:4]


def load_tum(path):
    a = np.loadtxt(path, comments='#')
    return a[:, 0], a[:, 1:4]


def associate(t_gt, t_est, max_dt):
    idx = np.searchsorted(t_est, t_gt)
    idx = np.clip(idx, 1, len(t_est) - 1)
    left, right = t_est[idx - 1], t_est[idx]
    idx = np.where(np.abs(t_gt - left) < np.abs(t_gt - right), idx - 1, idx)
    ok = np.abs(t_est[idx] - t_gt) <= max_dt
    return np.nonzero(ok)[0], idx[ok]


def umeyama(src, dst):
    """R, t que minimizan ||dst - (R src + t)||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--est', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--max-dt', type=float, default=0.05)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    t_est, p_est = load_est(a.est)
    t_gt, p_gt = load_tum(a.gt)
    i_gt, i_est = associate(t_gt, t_est, a.max_dt)
    G, E = p_gt[i_gt], p_est[i_est]
    R, t = umeyama(E, G)
    E_al = E @ R.T + t
    err = np.linalg.norm(E_al - G, axis=1)
    err_xy = np.linalg.norm((E_al - G)[:, :2], axis=1)
    err_z = np.abs((E_al - G)[:, 2])
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(G, axis=0), axis=1))])
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    res = {
        'poses_asociadas': int(len(err)),
        'poses_referencia': int(len(t_gt)),
        'longitud_recorrido_m': float(dist[-1]),
        'ate_rmse_m': float(np.sqrt((err ** 2).mean())),
        'ate_media_m': float(err.mean()),
        'ate_mediana_m': float(np.median(err)),
        'ate_max_m': float(err.max()),
        'ate_rmse_xy_m': float(np.sqrt((err_xy ** 2).mean())),
        'ate_rmse_z_m': float(np.sqrt((err_z ** 2).mean())),
        'error_final_m': float(err[-1]),
        'error_rel_recorrido_pct': float(100 * err.mean() / dist[-1]),
        'inclinacion_odom_vs_referencia_deg': float(tilt),
    }
    print(json.dumps(res, indent=2, ensure_ascii=False))

    if a.out:
        import cv2
        H, W = 700, 1400
        img = np.full((H, W, 3), 255, np.uint8)
        # planta (izquierda)
        allp = np.vstack([G[:, :2], E_al[:, :2]])
        lo, hi = allp.min(0) - 3, allp.max(0) + 3
        s = min((W / 2 - 40) / (hi - lo)[0], (H - 80) / (hi - lo)[1])
        def px(p):
            q = (p - lo) * s
            return np.stack([20 + q[:, 0], H - 40 - q[:, 1]], 1).astype(np.int32)
        cv2.polylines(img, [px(G[:, :2])], False, (60, 60, 60), 2)
        cv2.polylines(img, [px(E_al[:, :2])], False, (40, 40, 220), 2)
        cv2.putText(img, 'Planta: referencia (gris) vs rko_lio (rojo)', (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        # error vs distancia (derecha)
        x0, x1, y0, y1 = W // 2 + 60, W - 30, H - 60, 60
        emax = max(err.max() * 1.1, 0.1)
        pts = np.stack([x0 + dist / dist[-1] * (x1 - x0), y0 - err / emax * (y0 - y1)], 1).astype(np.int32)
        cv2.line(img, (x0, y0), (x1, y0), (0, 0, 0), 1)
        cv2.line(img, (x0, y0), (x0, y1), (0, 0, 0), 1)
        cv2.polylines(img, [pts], False, (40, 40, 220), 2)
        cv2.putText(img, f'ATE (m) vs distancia recorrida | RMSE = {res["ate_rmse_m"]:.2f} m',
                    (x0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, f'{emax:.1f}', (x0 - 45, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, '0', (x0 - 20, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, f'{dist[-1]:.0f} m', (x1 - 50, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(a.out, img)
        print('figura:', a.out)


if __name__ == '__main__':
    main()
