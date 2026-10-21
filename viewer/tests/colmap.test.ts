import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { averageUp, parseImagesBin } from "../src/colmap";

/** COLMAP images.bin (https://colmap.github.io/format.html) for cameras given as
 *  world-from-camera rotation + centre; one 2D point each. */
function imagesBin(
  cameras: { name: string; worldFromCam: THREE.Matrix4; centre: THREE.Vector3 }[],
): ArrayBuffer {
  const bytes: Uint8Array[] = [];
  const u64 = (v: number) => {
    const b = new Uint8Array(8);
    new DataView(b.buffer).setBigUint64(0, BigInt(v), true);
    return b;
  };
  bytes.push(u64(cameras.length));
  cameras.forEach((cam, i) => {
    // COLMAP stores camera-from-world: R = (world-from-camera)^T, t = -R C.
    const r = cam.worldFromCam.clone().transpose();
    const q = new THREE.Quaternion().setFromRotationMatrix(r);
    const t = cam.centre.clone().applyMatrix4(r).negate();
    const head = new DataView(new ArrayBuffer(4 + 56 + 4));
    head.setUint32(0, i + 1, true);
    [q.w, q.x, q.y, q.z, t.x, t.y, t.z].forEach((v, k) => head.setFloat64(4 + 8 * k, v, true));
    head.setUint32(60, 1, true);
    bytes.push(new Uint8Array(head.buffer));
    bytes.push(new TextEncoder().encode(cam.name + "\0"));
    bytes.push(u64(1));
    const point = new DataView(new ArrayBuffer(24));
    point.setFloat64(0, 10, true);
    point.setFloat64(8, 20, true);
    point.setBigInt64(16, -1n, true);
    bytes.push(new Uint8Array(point.buffer));
  });
  const total = bytes.reduce((n, b) => n + b.length, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const b of bytes) {
    out.set(b, o);
    o += b.length;
  }
  return out.buffer;
}

// OpenCV axes: x right, y down, z forward. A camera looking along world +X, upright (+Z up).
const lookX = new THREE.Matrix4().makeBasis(
  new THREE.Vector3(0, -1, 0), // right
  new THREE.Vector3(0, 0, -1), // down
  new THREE.Vector3(1, 0, 0), // forward
);

describe("parseImagesBin", () => {
  it("reads names, video groups, centres and axes", () => {
    const tilted = new THREE.Matrix4().makeRotationZ(0.3).multiply(lookX);
    const cams = parseImagesBin(
      imagesBin([
        { name: "middle_00000.jpg", worldFromCam: lookX, centre: new THREE.Vector3(1, 2, 3) },
        { name: "up_00007.jpg", worldFromCam: tilted, centre: new THREE.Vector3(-1, 0, 0.5) },
      ]),
    );
    expect(cams.map((c) => [c.name, c.group])).toEqual([
      ["middle_00000.jpg", "middle"],
      ["up_00007.jpg", "up"],
    ]);
    cams[0].position.forEach((v, i) => expect(v).toBeCloseTo([1, 2, 3][i], 9));
    cams[0].up.forEach((v, i) => expect(v).toBeCloseTo([0, 0, 1][i], 9));
    cams[0].forward.forEach((v, i) => expect(v).toBeCloseTo([1, 0, 0][i], 9));
    cams[1].position.forEach((v, i) => expect(v).toBeCloseTo([-1, 0, 0.5][i], 9));
  });
});

describe("averageUp", () => {
  it("averages the cameras' up directions (the room's up)", () => {
    const up = averageUp([
      { name: "a", group: "middle", position: [0, 0, 0], up: [0.1, 0, 1], forward: [1, 0, 0] },
      { name: "b", group: "middle", position: [0, 0, 0], up: [-0.1, 0, 1], forward: [0, 1, 0] },
    ]);
    expect(up.x).toBeCloseTo(0, 9);
    expect(up.z).toBeCloseTo(1, 9);
  });
});
