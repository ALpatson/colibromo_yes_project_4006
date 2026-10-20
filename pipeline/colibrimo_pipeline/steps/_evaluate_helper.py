"""Helper run in the GPU environment: score a splat PLY on held-out frames.

Renders the exported splat (what the viewer gets) at each held-out frame with
gsplat's rasterizer and compares it with the real frame, loaded and undistorted
by gsplat's own COLMAP loader, so every trainer is scored on identical images.
Metrics as in gsplat v1.5.3's simple_trainer.py eval (torchmetrics PSNR and
SSIM with data_range 1, LPIPS AlexNet with normalize=True).

Usage: python _evaluate_helper.py <job.json>
job: examples_dir, data_dir (colmap/), ply, matrix_4x4 (COLMAP -> PLY
coordinates), test_frames, save_frames, out_dir, antialiased
Prints one JSON line with the results.
"""

import json
import sys
import time


def main() -> None:
    job = json.load(open(sys.argv[1], encoding="utf-8"))
    sys.path.insert(0, job["examples_dir"])

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from datasets.colmap import Dataset, Parser  # gsplat examples package
    from gsplat import rasterization
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    from colibrimo_pipeline.utils.cameras import scene_camtoworld
    from colibrimo_pipeline.utils.ply import read_splat_ply

    device = "cuda"
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
    lpips = lpips.to(device)

    # COLMAP's own coordinates (normalize=False); the PLY's are matrix @ COLMAP.
    parser = Parser(data_dir=job["data_dir"], factor=1, normalize=False, test_every=8)
    dataset = Dataset(parser, split="train")
    index = {name: i for i, name in enumerate(parser.image_names)}
    missing = [n for n in job["test_frames"] if n not in index]
    if missing:
        raise SystemExit(f"Held-out frames not in the COLMAP model: {missing[:5]}")
    dataset.indices = np.array([index[n] for n in job["test_frames"]])
    matrix = np.asarray(job["matrix_4x4"], dtype=np.float64)

    raw = read_splat_ply(job["ply"])
    tensor = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    quats = tensor(raw["quats"])
    splat = {
        "means": tensor(raw["means"]),
        "quats": quats / quats.norm(dim=-1, keepdim=True),
        "scales": torch.exp(tensor(raw["log_scales"])),
        "opacities": torch.sigmoid(tensor(raw["opacity_logits"])),
        "colors": tensor(raw["sh"]),
    }
    mode = "antialiased" if job.get("antialiased") else "classic"
    save = set(job["save_frames"])
    per_frame, saved, render_s = [], [], 0.0
    for item, name in enumerate(job["test_frames"]):
        data = dataset[item]
        c2w = scene_camtoworld(data["camtoworld"].numpy().astype(np.float64), matrix)
        viewmat = torch.from_numpy(np.linalg.inv(c2w)).float().to(device)[None]
        K = data["K"].float().to(device)[None]
        pixels = data["image"].float().to(device)[None] / 255.0  # [1, H, W, 3]
        height, width = pixels.shape[1:3]

        torch.cuda.synchronize()
        tic = time.time()
        with torch.no_grad():
            colors, _, _ = rasterization(
                **splat,
                viewmats=viewmat,
                Ks=K,
                width=width,
                height=height,
                sh_degree=raw["sh_degree"],
                near_plane=0.01,
                far_plane=1e10,
                rasterize_mode=mode,
            )
        torch.cuda.synchronize()
        render_s += time.time() - tic
        colors = torch.clamp(colors, 0.0, 1.0)

        p, c = pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
        with torch.no_grad():
            per_frame.append(
                {
                    "frame": name,
                    "psnr": float(psnr(c, p)),
                    "ssim": float(ssim(c, p)),
                    "lpips": float(lpips(c, p)),
                }
            )
        if name in save:
            canvas = torch.cat([pixels, colors], dim=2)[0].cpu().numpy()
            out = f"{job['out_dir']}/{name.rsplit('.', 1)[0]}_real_vs_render.png"
            imageio.imwrite(out, (canvas * 255).round().astype(np.uint8))
            saved.append(out)

    mean = lambda key: float(np.mean([f[key] for f in per_frame]))  # noqa: E731
    print(
        json.dumps(
            {
                "psnr": round(mean("psnr"), 3),
                "ssim": round(mean("ssim"), 4),
                "lpips": round(mean("lpips"), 4),
                "num_test_frames": len(per_frame),
                "num_splats": raw["count"],
                "sh_degree": raw["sh_degree"],
                "rasterize_mode": mode,
                "render_ms_per_frame": round(1000 * render_s / len(per_frame), 1),
                "image_size": [int(width), int(height)],
                "per_frame": per_frame,
                "saved_images": saved,
            }
        )
    )


if __name__ == "__main__":
    main()
