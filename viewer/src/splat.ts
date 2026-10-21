import * as THREE from "three";
import { SplatMesh } from "@sparkjsdev/spark";

export type LoadStage = "download" | "decode";
export type ProgressFn = (stage: LoadStage, loaded: number, total: number) => void;

/**
 * Download a file with byte-level progress.
 * We fetch ourselves (instead of SplatMesh's `url` option) so the progress bar is
 * reliable and so the same bytes could be reused later (before/after switching).
 */
export async function fetchWithProgress(url: string, onProgress: ProgressFn): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Could not download ${url} (HTTP ${res.status})`);

  const total = Number(res.headers.get("Content-Length")) || 0;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress("download", loaded, total);
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.byteLength;
  }
  return bytes;
}

/** Download a .ply (or .spz/.splat/...) and turn it into an initialized SplatMesh. */
export async function loadSplat(url: string, onProgress: ProgressFn): Promise<SplatMesh> {
  const bytes = await fetchWithProgress(url, onProgress);
  onProgress("decode", 0, 0);

  const mesh = new SplatMesh({
    fileBytes: bytes,
    fileName: url.split("/").pop(),
  });
  await mesh.initialized;
  return mesh;
}

/** Object-space centres of the splats that are not nearly transparent, as packed xyz triples. */
export function opaqueCenters(mesh: SplatMesh, minOpacity = 0.2): Float32Array {
  const out: number[] = [];
  mesh.forEachSplat((_i, center, _scales, _quat, opacity) => {
    if (opacity >= minOpacity) out.push(center.x, center.y, center.z);
  });
  return Float32Array.from(out);
}

/** p-th percentile (0..1) of values (the array is sorted in place). */
export function percentile(values: Float32Array, p: number): number {
  values.sort();
  return values[Math.min(values.length - 1, Math.floor(p * values.length))] ?? 0;
}

/**
 * Robust bounds of points after applying `matrix`: per-axis percentiles. Unlike
 * the full bounding box, this is not blown up by the stray "floaters" that every
 * capture has.
 */
export function robustBounds(
  centers: Float32Array,
  matrix: THREE.Matrix4,
  lowPct = 0.05,
  highPct = 0.95,
): THREE.Box3 {
  const n = centers.length / 3;
  const axes = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
  const v = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    v.fromArray(centers, i * 3).applyMatrix4(matrix);
    axes[0][i] = v.x;
    axes[1][i] = v.y;
    axes[2][i] = v.z;
  }
  const [x, y, z] = axes.map((a) => [percentile(a, lowPct), percentile(a, highPct)]);
  return new THREE.Box3(new THREE.Vector3(x[0], y[0], z[0]), new THREE.Vector3(x[1], y[1], z[1]));
}
