import * as THREE from "three";
import { publicUrl } from "./config";

/**
 * One camera pose, in the splat's own coordinates ("raw"; the room calibration
 * is applied on top). `group` is the source video ("bottom" | "middle" | "up").
 */
export interface ColmapCamera {
  name: string;
  group: string;
  position: [number, number, number];
  up: [number, number, number];
  forward: [number, number, number];
}

type Vec3 = [number, number, number];

/** Camera entry of the pipeline's cameras.json ("video") or the old converter's JSON ("group"). */
interface CameraJson {
  name: string;
  video?: string;
  group?: string;
  position: Vec3;
  up: Vec3;
  forward: Vec3;
}

export const GROUP_COLORS: Record<string, number> = {
  bottom: 0xff9a3c,
  middle: 0x3cff6e,
  up: 0x3caeff,
};

/** Video name from a frame name: "bottom_00012.jpg" -> "bottom". */
const videoOfFrame = (name: string) => name.replace(/_\d+\.\w+$/, "");

/**
 * Load camera poses from:
 * - the pipeline's cameras.json (export step; already in scene.ply's coordinates),
 * - the old `npm run colmap` JSON (same fields, "group" instead of "video"),
 * - or, for old runs without cameras.json, COLMAP's images.bin itself
 *   (a path ending in .bin, or a sparse model folder ending in "/").
 * Returns [] (and logs why) if nothing could be loaded.
 */
export async function loadCameras(path: string): Promise<ColmapCamera[]> {
  try {
    const binary = path.endsWith(".bin") || path.endsWith("/");
    const url = publicUrl(path.endsWith("/") ? `${path}images.bin` : path);
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    if (binary) return parseImagesBin(await res.arrayBuffer());

    const cameras = ((await res.json()) as { cameras: CameraJson[] }).cameras;
    return cameras.map((c) => ({
      name: c.name,
      group: c.video ?? c.group ?? videoOfFrame(c.name),
      position: c.position,
      up: c.up,
      forward: c.forward,
    }));
  } catch (err) {
    console.warn(`Camera poses (${path}) not loaded:`, err);
    return [];
  }
}

/**
 * Read camera poses straight from COLMAP's images.bin (format:
 * https://colmap.github.io/format.html#binary-file-format). COLMAP stores
 * camera-from-world (q, t): the centre is -Rᵀt, and the rows of R are the
 * camera's x (right), y (down), z (forward) axes in world coordinates.
 * Note: the poses are in COLMAP's coordinates, which match the splat only for
 * trainers that keep them (Brush); for gsplat use the pipeline's cameras.json.
 */
export function parseImagesBin(buffer: ArrayBuffer): ColmapCamera[] {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  const count = Number(view.getBigUint64(0, true));
  const cameras: ColmapCamera[] = [];
  let o = 8;
  for (let i = 0; i < count; i++) {
    o += 4; // image_id
    const [qw, qx, qy, qz, tx, ty, tz] = Array.from({ length: 7 }, (_, k) =>
      view.getFloat64(o + 8 * k, true),
    );
    o += 56 + 4; // qvec + tvec, camera_id
    let end = o;
    while (bytes[end] !== 0) end++;
    const name = new TextDecoder().decode(bytes.subarray(o, end));
    o = end + 1;
    const numPoints2D = Number(view.getBigUint64(o, true));
    o += 8 + 24 * numPoints2D;

    const r = new THREE.Matrix3().setFromMatrix4(
      new THREE.Matrix4().makeRotationFromQuaternion(new THREE.Quaternion(qx, qy, qz, qw)),
    );
    const e = r.elements; // column-major: row i of R = (e[i], e[i+3], e[i+6])
    const row = (k: number) => new THREE.Vector3(e[k], e[k + 3], e[k + 6]);
    const t = new THREE.Vector3(tx, ty, tz);
    const centre = t.applyMatrix3(r.clone().transpose()).negate();
    cameras.push({
      name,
      group: videoOfFrame(name),
      position: centre.toArray() as Vec3,
      up: row(1).negate().toArray() as Vec3,
      forward: row(2).toArray() as Vec3,
    });
  }
  return cameras;
}

/** Average "up" of the cameras (raw coordinates). The videos were filmed upright, so this is the room's up. */
export function averageUp(cameras: ColmapCamera[]): THREE.Vector3 {
  const up = new THREE.Vector3();
  for (const c of cameras) up.add(new THREE.Vector3().fromArray(c.up));
  return up.normalize();
}

/**
 * Small cones at each camera position, pointing where the camera looked,
 * coloured by group. Lives in raw coordinates: add it to Room.root.
 */
export function createCameraMarkers(cameras: ColmapCamera[], size: number): THREE.Group {
  const group = new THREE.Group();
  group.name = "colmap-cameras";

  // Cone tip along +Z (= camera forward after orienting).
  const geometry = new THREE.ConeGeometry(size * 0.5, size * 1.5, 8).rotateX(Math.PI / 2);
  for (const [name, color] of Object.entries(GROUP_COLORS)) {
    const cams = cameras.filter((c) => c.group === name);
    if (!cams.length) continue;
    const mesh = new THREE.InstancedMesh(
      geometry,
      new THREE.MeshBasicMaterial({ color }),
      cams.length,
    );
    const m = new THREE.Matrix4();
    const pos = new THREE.Vector3();
    const fwd = new THREE.Vector3();
    const up = new THREE.Vector3();
    const x = new THREE.Vector3();
    cams.forEach((c, i) => {
      pos.fromArray(c.position);
      fwd.fromArray(c.forward);
      up.fromArray(c.up);
      x.crossVectors(up, fwd);
      m.makeBasis(x, up, fwd).setPosition(pos);
      mesh.setMatrixAt(i, m);
    });
    group.add(mesh);
  }
  return group;
}
