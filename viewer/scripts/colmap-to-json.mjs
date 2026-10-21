// Convert a COLMAP sparse model (images.bin) into public/colmap/cameras.json,
// a small file with each camera's position and orientation that the viewer loads.
//
// Usage:  npm run colmap -- <path to sparse/0> [output json]
// Example: npm run colmap -- ../data/runs/myRoom/colmap/sparse/0
//
// No COLMAP install needed: images.bin is read directly.
// Format: https://colmap.github.io/format.html#binary-file-format

/* global process, console -- Node.js script */

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, dirname, resolve } from "node:path";

const modelDir = process.argv[2];
const outFile = resolve(process.argv[3] ?? "public/colmap/cameras.json");
if (!modelDir) {
  console.error("Usage: npm run colmap -- <path to sparse/0> [output json]");
  process.exit(1);
}

const buf = readFileSync(join(modelDir, "images.bin"));
let o = 0;
const u32 = () => {
  const v = buf.readUInt32LE(o);
  o += 4;
  return v;
};
const u64 = () => {
  const v = Number(buf.readBigUInt64LE(o));
  o += 8;
  return v;
};
const f64 = () => {
  const v = buf.readDoubleLE(o);
  o += 8;
  return v;
};

// World-from-camera rotation (R^T) from COLMAP's camera-from-world quaternion.
function rotationMatrix(qw, qx, qy, qz) {
  return [
    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
    [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
    [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
  ];
}

const round = (v) => Math.round(v * 1e6) / 1e6;
const cameras = [];
const count = u64();
for (let i = 0; i < count; i++) {
  u32(); // image_id
  const [qw, qx, qy, qz] = [f64(), f64(), f64(), f64()];
  const t = [f64(), f64(), f64()];
  u32(); // camera_id
  let end = o;
  while (buf[end] !== 0) end++;
  const name = buf.toString("utf8", o, end);
  o = end + 1;
  const numPoints2D = u64();
  o += numPoints2D * 24; // x, y (double) + point3D_id (int64)

  const R = rotationMatrix(qw, qx, qy, qz); // camera-from-world
  // Camera centre C = -R^T t. Camera axes in world space are the rows of R:
  // COLMAP cameras look along +z with +y pointing down in the image.
  const position = [0, 1, 2].map((j) => -(R[0][j] * t[0] + R[1][j] * t[1] + R[2][j] * t[2]));
  const up = R[1].map((v) => -v);
  const forward = R[2];

  cameras.push({
    name,
    group: name.split(/[_\-.]/)[0], // "bottom", "middle", "up"
    position: position.map(round),
    up: up.map(round),
    forward: forward.map(round),
  });
}

cameras.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
mkdirSync(dirname(outFile), { recursive: true });
writeFileSync(outFile, JSON.stringify({ source: resolve(modelDir), cameras }, null, 1));

const groups = {};
for (const c of cameras) groups[c.group] = (groups[c.group] ?? 0) + 1;
console.log(`Wrote ${cameras.length} cameras to ${outFile}`, groups);
