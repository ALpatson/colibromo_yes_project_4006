import * as THREE from "three";
import type { ColmapCamera } from "./colmap";
import type { PlayerSettings } from "./config";
import type { Room } from "./room";

/**
 * Where the user may walk, derived from the COLMAP cameras (the places the
 * room was filmed from, so where the splat looks good):
 * - eye height: average height of the eye group (middle_* frames),
 * - walk area: convex outline of the eye-group camera positions seen from above,
 *   shrunk automatically to keep a distance from the walls,
 * - room size: the larger side of that outline (used for speed and margins),
 * - start: a middle camera looking into the room, turned toward the centre.
 *
 * Positions are 2D on the floor plan: Vector2(x, z).
 */
export class WalkArea {
  eyeY = 0;
  /** Estimated floor height (for markers on the floor). */
  floorY = -1;
  roomSize = 1;
  margin = 0;
  centre = new THREE.Vector2();
  /** Direction of the room's long axis on the floor plan (radians), from the outline's shape. */
  axisAngle = 0;
  /** Convex outline of the camera positions, counter-clockwise. */
  hull: THREE.Vector2[] = [];
  /** Outward unit normal of each hull edge (edge i goes from hull[i] to hull[i + 1]). */
  private normals: THREE.Vector2[] = [];
  start = { position: new THREE.Vector3(), yaw: 0 };

  constructor(
    private readonly room: Room,
    private readonly cameras: ColmapCamera[],
    private settings: PlayerSettings,
  ) {
    this.update(settings);
  }

  /** Recompute everything (after the room transform or the settings changed). */
  update(settings: PlayerSettings = this.settings) {
    this.settings = settings;
    const world = this.cameras.map((c) => ({ cam: c, pos: this.room.toWorld(c.position) }));
    if (!world.length) return this.fallbackFromSplat();

    const eyeCams = world.filter((w) => w.cam.group === settings.eyeGroup);
    const eyeSet = eyeCams.length ? eyeCams : world;

    // Footprint of the eye-group cameras: that is where the filming person's
    // eyes actually were. (The "up" cameras, held overhead, lean close to the walls.)
    this.hull = convexHull(eyeSet.map((w) => new THREE.Vector2(w.pos.x, w.pos.z)));
    this.normals = this.hull.map((a, i) => {
      const b = this.hull[(i + 1) % this.hull.length];
      return new THREE.Vector2(b.y - a.y, -(b.x - a.x)).normalize();
    });

    const box = new THREE.Box2().setFromPoints(this.hull);
    const size = box.getSize(new THREE.Vector2());
    this.roomSize = Math.max(size.x, size.y, 1e-3);
    this.centre = this.hull
      .reduce((s, p) => s.add(p), new THREE.Vector2())
      .divideScalar(this.hull.length);
    this.axisAngle = principalAngle(this.hull, this.centre);
    this.eyeY =
      eyeSet.reduce((s, w) => s + w.pos.y, 0) / eyeSet.length + settings.eyeOffset * this.roomSize;
    this.margin =
      settings.walkMargin !== null
        ? settings.walkMargin * this.roomSize
        : this.autoMargin(settings.wallDistance * this.roomSize);
    this.floorY = this.estimateFloor();

    this.chooseStart(eyeSet);
  }

  /**
   * Where a ray from the centre in direction `angle` (radians, on the floor
   * plan) leaves the walk area. Exact for the convex area: the nearest of the
   * edge lines (shifted by the margin) that the ray runs into.
   */
  edgePoint(angle: number, margin = this.margin): THREE.Vector2 {
    const d = new THREE.Vector2(Math.cos(angle), Math.sin(angle));
    let tMin = Infinity;
    this.hull.forEach((a, i) => {
      const n = this.normals[i];
      const facing = n.dot(d);
      if (facing <= 1e-9) return;
      // Solve (centre + t·d − a)·n = margin for t.
      const t = (margin - ((this.centre.x - a.x) * n.x + (this.centre.y - a.y) * n.y)) / facing;
      tMin = Math.min(tMin, t);
    });
    return this.centre.clone().addScaledVector(d, Math.max(0, tMin));
  }

  /** Is this floor-plan point inside the walk area? */
  contains(p: THREE.Vector2): boolean {
    return this.hull.every((a, i) => {
      const n = this.normals[i];
      return (p.x - a.x) * n.x + (p.y - a.y) * n.y <= this.margin + 1e-6;
    });
  }

  /**
   * Floor height: a low percentile of the splats below eye level inside the
   * walk area (where the floor is visible and filmed). Only used to place markers.
   */
  private estimateFloor(): number {
    const ys: number[] = [];
    const v = new THREE.Vector3();
    const p = new THREE.Vector2();
    const m = this.room.root.matrixWorld;
    const c = this.room.rawCenters;
    for (let i = 0; i < c.length; i += 3) {
      v.fromArray(c, i).applyMatrix4(m);
      if (v.y < this.eyeY && this.contains(p.set(v.x, v.z))) ys.push(v.y);
    }
    const lowest = this.eyeY - 0.6 * this.roomSize;
    const highest = this.eyeY - 0.1 * this.roomSize;
    if (ys.length < 50) return (lowest + highest) / 2;
    ys.sort((a, b) => a - b);
    return THREE.MathUtils.clamp(ys[Math.floor(ys.length * 0.05)], lowest, highest);
  }

  /**
   * People usually film a room walking along the walls and facing inward, so the
   * camera footprint touches the walls. Find the largest margin (usually an
   * inset, i.e. negative) for which every edge of the walk area stays at least
   * `minDist` away from the splats at eye height (walls, tall furniture).
   */
  private autoMargin(minDist: number): number {
    // Floor-plan positions of the splats in a thin slab around eye height.
    const slab: number[] = [];
    const v = new THREE.Vector3();
    const m = this.room.root.matrixWorld;
    const c = this.room.rawCenters;
    const halfThickness = 0.05 * this.roomSize;
    for (let i = 0; i < c.length; i += 3) {
      v.fromArray(c, i).applyMatrix4(m);
      if (Math.abs(v.y - this.eyeY) < halfThickness) slab.push(v.x, v.z);
    }
    if (slab.length < 20) return 0;

    // Distance to the 5th nearest slab point, so single floaters don't count.
    const K = 5;
    const wallDist = (p: THREE.Vector2) => {
      const nearest: number[] = [];
      for (let i = 0; i < slab.length; i += 2) {
        const d2 = (slab[i] - p.x) ** 2 + (slab[i + 1] - p.y) ** 2;
        if (nearest.length < K || d2 < nearest[K - 1]) {
          nearest.push(d2);
          nearest.sort((a, b) => a - b);
          if (nearest.length > K) nearest.pop();
        }
      }
      return Math.sqrt(nearest[nearest.length - 1]);
    };
    // Smallest wall distance along the edge of the walk area (sampled in 32 directions).
    const edgeClearance = (margin: number) => {
      let min = Infinity;
      for (let k = 0; k < 48; k++)
        min = Math.min(min, wallDist(this.edgePoint((k / 48) * Math.PI * 2, margin)));
      return min;
    };

    // Clearance grows as the margin shrinks: binary search the largest margin that is enough.
    // Never inset more than 90 % of the distance from the centre to the nearest edge.
    const inradius = Math.min(
      ...this.hull.map(
        (a, i) =>
          -((this.centre.x - a.x) * this.normals[i].x + (this.centre.y - a.y) * this.normals[i].y),
      ),
    );
    let lo = -0.9 * inradius;
    let hi = 0.05 * this.roomSize;
    if (edgeClearance(hi) >= minDist) return hi;
    for (let i = 0; i < 14; i++) {
      const mid = (lo + hi) / 2;
      if (edgeClearance(mid) >= minDist) lo = mid;
      else hi = mid;
    }
    return lo;
  }

  /**
   * Start at an eye-group camera that is away from the centre and was itself
   * looking into the room (so the splat is good from there), facing the centre.
   */
  private chooseStart(eyeSet: { cam: ColmapCamera; pos: THREE.Vector3 }[]) {
    const maxDist = Math.max(
      ...eyeSet.map((w) => this.centre.distanceTo(new THREE.Vector2(w.pos.x, w.pos.z))),
    );
    const rot = new THREE.Matrix3().setFromMatrix4(this.room.root.matrixWorld);
    let best = eyeSet[0];
    let bestScore = -Infinity;
    for (const w of eyeSet) {
      const toCentre = this.centre.clone().sub(new THREE.Vector2(w.pos.x, w.pos.z));
      const dist = toCentre.length();
      const fwd = new THREE.Vector3().fromArray(w.cam.forward).applyMatrix3(rot);
      const fwd2 = new THREE.Vector2(fwd.x, fwd.z).normalize();
      const facing = fwd2.dot(toCentre.normalize()); // 1 = looking at the centre
      const score = facing + (dist > 0.4 * maxDist ? 0.5 : 0) + 0.3 * (dist / maxDist);
      if (score > bestScore) {
        bestScore = score;
        best = w;
      }
    }
    const p = new THREE.Vector2(best.pos.x, best.pos.z);
    this.clamp2(p);
    this.start.position.set(p.x, this.eyeY, p.y);
    this.start.yaw = yawToward(p, this.centre);
  }

  /** Without camera poses: use the splat's dense bounds (worse, but usable). */
  private fallbackFromSplat() {
    const b = this.room.bounds(0.1, 0.9);
    this.hull = [
      new THREE.Vector2(b.min.x, b.min.z),
      new THREE.Vector2(b.max.x, b.min.z),
      new THREE.Vector2(b.max.x, b.max.z),
      new THREE.Vector2(b.min.x, b.max.z),
    ];
    this.normals = [
      new THREE.Vector2(0, -1),
      new THREE.Vector2(1, 0),
      new THREE.Vector2(0, 1),
      new THREE.Vector2(-1, 0),
    ];
    const size = b.getSize(new THREE.Vector3());
    this.roomSize = Math.max(size.x, size.z);
    this.margin = 0;
    this.centre.set((b.min.x + b.max.x) / 2, (b.min.z + b.max.z) / 2);
    this.eyeY = b.getCenter(new THREE.Vector3()).y + this.settings.eyeOffset * this.roomSize;
    this.floorY = b.min.y;
    this.start.position.set(this.centre.x, this.eyeY, this.centre.y);
    this.start.yaw = 0;
  }

  /** Push a floor-plan point back inside the walk area (slides along the edges). */
  clamp2(p: THREE.Vector2): THREE.Vector2 {
    // The area is convex, so a few rounds of pushing out of each violated
    // half-plane converge (also in the corners).
    for (let iter = 0; iter < 4; iter++) {
      let moved = false;
      for (let i = 0; i < this.hull.length; i++) {
        const n = this.normals[i];
        const a = this.hull[i];
        const out = (p.x - a.x) * n.x + (p.y - a.y) * n.y - this.margin;
        if (out > 0) {
          p.x -= n.x * out;
          p.y -= n.y * out;
          moved = true;
        }
      }
      if (!moved) break;
    }
    return p;
  }

  /** Clamp a 3D position: x/z into the walk area, y to eye height. */
  clamp(position: THREE.Vector3) {
    const p = this.clamp2(new THREE.Vector2(position.x, position.z));
    position.set(p.x, this.eyeY, p.y);
  }

  /** The walkable outline (hull + margin) as a line loop at height y, for debugging. */
  createDebugOutline(y: number): THREE.LineLoop {
    const pts = this.hull.map((v, i) => {
      // Miter offset: move each corner out along both neighbouring edge normals.
      const n1 = this.normals[(i - 1 + this.hull.length) % this.hull.length];
      const n2 = this.normals[i];
      const miter = n1
        .clone()
        .add(n2)
        .divideScalar(1 + n1.dot(n2));
      return new THREE.Vector3(v.x + miter.x * this.margin, y, v.y + miter.y * this.margin);
    });
    return new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0xffe14d, depthTest: false }),
    );
  }
}

/** Yaw (radians, three.js convention: 0 looks along -Z) to look from `from` toward `to` on the floor plan. */
export function yawToward(from: THREE.Vector2, to: THREE.Vector2): number {
  return Math.atan2(-(to.x - from.x), -(to.y - from.y));
}

/**
 * Main axis of a polygon's outline (principal component of its boundary,
 * sampled evenly along the edges so long walls count more than corners).
 */
function principalAngle(poly: THREE.Vector2[], centre: THREE.Vector2): number {
  let sxx = 0;
  let szz = 0;
  let sxz = 0;
  poly.forEach((a, i) => {
    const b = poly[(i + 1) % poly.length];
    const len = a.distanceTo(b);
    const steps = 8;
    for (let s = 0; s < steps; s++) {
      const p = a
        .clone()
        .lerp(b, (s + 0.5) / steps)
        .sub(centre);
      const w = len / steps;
      sxx += w * p.x * p.x;
      szz += w * p.y * p.y;
      sxz += w * p.x * p.y;
    }
  });
  return 0.5 * Math.atan2(2 * sxz, sxx - szz);
}

/** Convex hull (Andrew's monotone chain), counter-clockwise. */
function convexHull(points: THREE.Vector2[]): THREE.Vector2[] {
  const pts = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  if (pts.length < 3) return pts;
  const cross = (o: THREE.Vector2, a: THREE.Vector2, b: THREE.Vector2) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower: THREE.Vector2[] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
      lower.pop();
    lower.push(p);
  }
  const upper: THREE.Vector2[] = [];
  for (const p of [...pts].reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
      upper.pop();
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}
