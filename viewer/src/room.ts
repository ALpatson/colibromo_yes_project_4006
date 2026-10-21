import * as THREE from "three";
import type { SplatMesh } from "@sparkjsdev/spark";
import type { Calibration } from "./config";
import { averageUp, type ColmapCamera } from "./colmap";
import { opaqueCenters, robustBounds } from "./splat";

/** Wrap an angle in degrees into -180..180. */
const wrapDeg = (deg: number) => ((((deg + 180) % 360) + 360) % 360) - 180;

/**
 * The room: a group holding the splat (and debug markers) in raw COLMAP
 * coordinates. The calibration is the group's transform, so everything outside
 * the group works in a levelled world with y up. Units stay COLMAP units:
 * eye height, speed and walk area are all derived from the camera poses, so
 * no real-world scale is needed.
 *
 * The helpers edit the calibration arrays in place, so UI bound to them stays in sync.
 */
export class Room {
  readonly root = new THREE.Group();
  /** Raw (COLMAP) centres of the opaque splats, cached for framing. */
  readonly rawCenters: Float32Array;
  calibration: Calibration = { rotationDeg: [0, 0, 0], scale: 1, offset: [0, 0, 0] };

  constructor(readonly splat: SplatMesh) {
    this.root.name = "room";
    this.root.add(splat);
    this.rawCenters = opaqueCenters(splat);
  }

  /** Use a manual calibration (from config.json). */
  setCalibration(calibration: Calibration) {
    this.calibration = structuredClone(calibration);
    this.applyCalibration();
  }

  /** Push `calibration` into the group transform. Call after editing it. */
  applyCalibration() {
    const c = this.calibration;
    const toRad = THREE.MathUtils.degToRad;
    this.root.rotation.set(
      toRad(c.rotationDeg[0]),
      toRad(c.rotationDeg[1]),
      toRad(c.rotationDeg[2]),
      "YXZ",
    );
    this.root.scale.setScalar(c.scale);
    this.root.position.fromArray(c.offset);
    this.root.updateMatrixWorld(true);
  }

  /** World-space robust bounds of the room (ignores floaters). */
  bounds(lowPct = 0.05, highPct = 0.95): THREE.Box3 {
    return robustBounds(this.rawCenters, this.root.matrixWorld, lowPct, highPct);
  }

  /** Raw COLMAP point → world. */
  toWorld(raw: THREE.Vector3 | [number, number, number]): THREE.Vector3 {
    const v = Array.isArray(raw) ? new THREE.Vector3().fromArray(raw) : raw.clone();
    return v.applyMatrix4(this.root.matrixWorld);
  }

  /**
   * Automatic calibration: level the room so the cameras' average "up" (the
   * videos were filmed upright) points to +Y, and centre the cameras at the origin.
   */
  autoCalibrate(cameras: ColmapCamera[], flip: boolean) {
    this.calibration = { rotationDeg: [0, 0, 0], scale: 1, offset: [0, 0, 0] };
    this.levelTo(averageUp(cameras));
    if (flip) this.flip();

    const centre = new THREE.Vector3();
    for (const c of cameras) centre.add(this.toWorld(c.position));
    centre.divideScalar(cameras.length);
    this.calibration.offset = [-centre.x, -centre.y, -centre.z];
    this.applyCalibration();
  }

  /** Rotate so that `rawUp` points to world +Y; keeps the current turn (Y angle). */
  levelTo(rawUp: THREE.Vector3) {
    const q = new THREE.Quaternion().setFromUnitVectors(
      rawUp.clone().normalize(),
      new THREE.Vector3(0, 1, 0),
    );
    const e = new THREE.Euler().setFromQuaternion(q, "YXZ");
    // With YXZ order the Y turn is applied last, around world up, so it can be
    // replaced freely without un-levelling the room.
    const toDeg = THREE.MathUtils.radToDeg;
    let x = toDeg(e.x);
    let z = toDeg(e.z);
    // (x, z) and (180 - x, z + 180) level the room the same way (they differ by a
    // half turn around Y). Prefer small Z so a flipped room reads as "Tilt X = 180".
    if (Math.abs(z) > 90) {
      x = wrapDeg(180 - x);
      z = wrapDeg(z + 180);
    }
    this.calibration.rotationDeg[0] = x;
    this.calibration.rotationDeg[2] = z;
    this.applyCalibration();
  }

  /** Turn the room upside down (around X). */
  flip() {
    const r = this.calibration.rotationDeg;
    r[0] = wrapDeg(r[0] + 180);
    this.applyCalibration();
  }
}
