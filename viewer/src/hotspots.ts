import * as THREE from "three";
import type { Hotspot } from "./config";
import type { Room } from "./room";
import { yawToward, type WalkArea } from "./walkArea";

/** A hotspot resolved into the current world: floor-plan position (at eye height) and view yaw. */
export interface HotspotPose {
  position: THREE.Vector3;
  yaw: number | null;
}

/**
 * Tour hotspots: rings on the floor that the user taps to glide to.
 * The list itself lives in config.hotspots (raw COLMAP coordinates).
 */
export class Hotspots {
  readonly group = new THREE.Group();
  /** Index of the hotspot the user is standing on (-1 = none). */
  current = -1;
  /** Index of the hotspot under the mouse (-1 = none). */
  hovered = -1;
  private markers: THREE.Group[] = [];

  constructor(
    public list: Hotspot[],
    private readonly room: Room,
    private area: WalkArea,
  ) {
    this.group.name = "hotspots";
    this.rebuild();
  }

  setArea(area: WalkArea) {
    this.area = area;
    this.rebuild();
  }

  /** World pose of hotspot i: position at eye height inside the walk area, and arrival yaw. */
  pose(i: number): HotspotPose {
    const h = this.list[i];
    const position = this.room.toWorld(h.position);
    position.y = this.area.eyeY;
    this.area.clamp(position);
    let yaw: number | null = null;
    if (h.look) {
      const dir = new THREE.Vector3()
        .fromArray(h.look)
        .transformDirection(this.room.root.matrixWorld);
      if (Math.hypot(dir.x, dir.z) > 1e-3) yaw = Math.atan2(-dir.x, -dir.z);
    }
    return { position, yaw };
  }

  /** Add a hotspot at a world position with a view yaw. Returns its index. */
  add(name: string, position: THREE.Vector3, yaw: number): number {
    const toRaw = this.room.root.matrixWorld.clone().invert();
    const raw = position.clone().applyMatrix4(toRaw);
    const lookWorld = new THREE.Vector3(-Math.sin(yaw), 0, -Math.cos(yaw));
    const lookRaw = lookWorld.transformDirection(toRaw);
    this.list.push({ name, position: round3(raw.toArray()), look: round3(lookRaw.toArray()) });
    this.rebuild();
    return this.list.length - 1;
  }

  remove(i: number) {
    this.list.splice(i, 1);
    this.rebuild();
  }

  /** Index of the hotspot nearest to a world position (floor plan). */
  nearest(position: THREE.Vector3): number {
    let best = -1;
    let bestD = Infinity;
    this.list.forEach((_, i) => {
      const p = this.pose(i).position;
      const d = Math.hypot(p.x - position.x, p.z - position.z);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    return best;
  }

  /**
   * A starter tour when the config has none: one spot toward each corner of
   * the walk area (following the room's main axis), each looking across the
   * room through the centre, and the centre itself, looking along the room.
   */
  generate(): Hotspot[] {
    this.list.length = 0;
    const a = this.area;
    for (let k = 0; k < 4; k++) {
      const angle = a.axisAngle + Math.PI / 4 + (k * Math.PI) / 2;
      const p = a.centre.clone().lerp(a.edgePoint(angle), 0.75);
      this.add(`Corner ${k + 1}`, new THREE.Vector3(p.x, a.eyeY, p.y), yawToward(p, a.centre));
    }
    // Centre: look along the long axis, toward the farther end of the walk area.
    const ends = [a.axisAngle, a.axisAngle + Math.PI].map((ang) => a.edgePoint(ang));
    const far = ends[0].distanceTo(a.centre) >= ends[1].distanceTo(a.centre) ? ends[0] : ends[1];
    this.add("Centre", new THREE.Vector3(a.centre.x, a.eyeY, a.centre.y), yawToward(a.centre, far));
    return this.list;
  }

  /** Recreate the floor markers (after the list, the room or the walk area changed). */
  rebuild() {
    for (const m of this.markers) {
      m.traverse((o) => {
        if (o instanceof THREE.Mesh) {
          o.geometry.dispose();
          (o.material as THREE.Material).dispose();
        }
      });
    }
    this.group.clear();
    this.markers = this.list.map((_, i) => {
      const marker = createMarker(this.area.roomSize * 0.035);
      const p = this.pose(i).position;
      // Slightly above the estimated floor so the floor splats don't cover it.
      marker.position.set(p.x, this.area.floorY + this.area.roomSize * 0.01, p.z);
      this.group.add(marker);
      return marker;
    });
  }

  /**
   * The hotspot whose marker is within `radiusPx` of a screen point, or -1.
   * Picking in screen space keeps small markers easy to hit on a phone.
   */
  pick(x: number, y: number, camera: THREE.Camera, radiusPx: number): number {
    let best = -1;
    let bestD = radiusPx;
    const v = new THREE.Vector3();
    const forward = camera.getWorldDirection(new THREE.Vector3());
    this.markers.forEach((m, i) => {
      if (i === this.current || !m.visible) return;
      if (v.copy(m.position).sub(camera.position).dot(forward) <= 0) return; // behind the camera
      v.copy(m.position).project(camera);
      const sx = ((v.x + 1) / 2) * window.innerWidth;
      const sy = ((1 - v.y) / 2) * window.innerHeight;
      const d = Math.hypot(sx - x, sy - y);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    return best;
  }

  /** Per frame: hide the marker under the user, highlight the hovered one, gentle pulse. */
  update(viewer: THREE.Vector3, time: number) {
    const near = this.area.roomSize * 0.06;
    this.current = -1;
    this.markers.forEach((m, i) => {
      const here = Math.hypot(m.position.x - viewer.x, m.position.z - viewer.z) < near;
      if (here) this.current = i;
      m.visible = !here;
      const pulse = 1 + 0.06 * Math.sin(time * 3 + i);
      m.scale.setScalar(i === this.hovered ? 1.25 : pulse);
    });
  }
}

/** Floor marker: a white ring with a faint filled disc, lying flat. */
function createMarker(radius: number): THREE.Group {
  const g = new THREE.Group();
  const common = { transparent: true, depthWrite: true, side: THREE.DoubleSide };
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(radius * 0.78, radius, 48),
    new THREE.MeshBasicMaterial({ color: 0xffffff, opacity: 0.95, ...common }),
  );
  const disc = new THREE.Mesh(
    new THREE.CircleGeometry(radius * 0.78, 48),
    new THREE.MeshBasicMaterial({ color: 0xffffff, opacity: 0.25, ...common }),
  );
  for (const mesh of [ring, disc]) {
    mesh.rotation.x = -Math.PI / 2;
    // Draw before the splats: splats in front (walls) then cover the marker,
    // and its depth hides the floor splats right under it.
    mesh.renderOrder = -1;
    g.add(mesh);
  }
  return g;
}

const round3 = (v: number[]) => v.map((x) => Math.round(x * 1e4) / 1e4) as [number, number, number];
