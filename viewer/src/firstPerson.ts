import * as THREE from "three";
import type { PlayerSettings } from "./config";
import type { WalkArea } from "./walkArea";
import { Joystick } from "./joystick";

const LOCKED_SENSITIVITY = 0.0022; // radians per pixel (mouse locked)
const DRAG_SENSITIVITY = 0.005; // radians per pixel (drag to look, mouse or touch)
const MAX_PITCH = THREE.MathUtils.degToRad(85);
/** Default gaze: slightly down, so the hotspot rings on the floor nearby are in view. */
export const REST_PITCH = THREE.MathUtils.degToRad(-11);

/**
 * First-person walking at a fixed eye height inside the walk area.
 * Desktop: drag with the mouse to look (cursor stays visible), WASD / arrows to
 *          walk, Shift to run. L toggles a game-style locked mouse (Esc to leave).
 * Touch:   drag to look, joystick (bottom left) to walk.
 */
export class FirstPersonControls {
  enabled = true;
  /** false in tour mode: no walking (keys / joystick), only looking around and gliding to hotspots. */
  walkEnabled = true;
  /** Show a pointing-hand cursor (e.g. over a hotspot). */
  hover = false;
  yaw = 0;
  pitch = 0;
  readonly position = new THREE.Vector3();
  /** Called when the pointer is locked / unlocked (desktop). */
  onLockChange?: (locked: boolean) => void;
  /** A click / tap that was not a drag, at screen coordinates (screen centre when the mouse is locked). */
  onTap?: (x: number, y: number) => void;

  private readonly keys = new Set<string>();
  private readonly velocity = new THREE.Vector3();
  private readonly joystick: Joystick;
  private lookPointer: {
    id: number;
    x: number;
    y: number;
    startX: number;
    startY: number;
    time: number;
  } | null = null;
  private glide: {
    from: THREE.Vector3;
    to: THREE.Vector3;
    fromYaw: number;
    toYaw: number | null;
    fromPitch: number;
    /** Turn the view during the glide (switched off as soon as the user looks around). */
    turn: boolean;
    t: number;
    duration: number;
  } | null = null;

  constructor(
    private readonly camera: THREE.PerspectiveCamera,
    private readonly dom: HTMLElement,
    private area: WalkArea,
    private settings: PlayerSettings,
  ) {
    this.joystick = new Joystick();
    this.joystick.setVisible(isTouchDevice());

    // --- optional locked mouse (L key; Esc leaves, handled by the browser) ---
    document.addEventListener("pointerlockchange", () => {
      this.updateCursor();
      this.onLockChange?.(this.locked);
    });
    document.addEventListener("mousemove", (e) => {
      if (!this.enabled || !this.locked) return;
      this.look(e.movementX * LOCKED_SENSITIVITY, e.movementY * LOCKED_SENSITIVITY);
    });
    document.addEventListener("mousedown", (e) => {
      if (this.enabled && this.locked && e.button === 0)
        this.onTap?.(window.innerWidth / 2, window.innerHeight / 2);
    });

    // --- keyboard ---
    window.addEventListener("keydown", (e) => {
      if (e.target instanceof HTMLInputElement) return;
      this.keys.add(e.code);
      if (e.code === "KeyL" && this.enabled && !isTouchDevice()) {
        if (this.locked) document.exitPointerLock();
        else dom.requestPointerLock();
      }
    });
    window.addEventListener("keyup", (e) => this.keys.delete(e.code));
    window.addEventListener("blur", () => this.keys.clear());

    // --- drag to look (mouse left button, or a finger anywhere except the joystick) ---
    dom.addEventListener("pointerdown", (e) => {
      if (!this.enabled || this.locked || this.lookPointer) return;
      if (e.pointerType === "mouse" && e.button !== 0) return;
      const { clientX: x, clientY: y } = e;
      this.lookPointer = { id: e.pointerId, x, y, startX: x, startY: y, time: performance.now() };
      dom.setPointerCapture(e.pointerId); // keep dragging even outside the window
      this.updateCursor();
    });
    dom.addEventListener("pointermove", (e) => {
      if (!this.lookPointer || e.pointerId !== this.lookPointer.id) return;
      this.look(
        (e.clientX - this.lookPointer.x) * DRAG_SENSITIVITY,
        (e.clientY - this.lookPointer.y) * DRAG_SENSITIVITY,
      );
      this.lookPointer.x = e.clientX;
      this.lookPointer.y = e.clientY;
    });
    const endLook = (e: PointerEvent) => {
      const p = this.lookPointer;
      if (p?.id !== e.pointerId) return;
      this.lookPointer = null;
      this.updateCursor();
      // Short and (almost) still: a tap, not a drag.
      const moved = Math.hypot(e.clientX - p.startX, e.clientY - p.startY);
      if (e.type === "pointerup" && moved < 8 && performance.now() - p.time < 500)
        this.onTap?.(e.clientX, e.clientY);
    };
    dom.addEventListener("pointerup", endLook);
    dom.addEventListener("pointercancel", endLook);
    this.updateCursor();
  }

  /** Grab cursor while drag-to-look is available. (OrbitControls also writes the cursor.) */
  updateCursor() {
    let cursor = "grab";
    if (!this.enabled || this.locked) cursor = "";
    else if (this.lookPointer) cursor = "grabbing";
    else if (this.hover) cursor = "pointer";
    this.dom.style.cursor = cursor;
  }

  setHover(hover: boolean) {
    if (hover === this.hover) return;
    this.hover = hover;
    this.updateCursor();
  }

  setWalkEnabled(walk: boolean) {
    this.walkEnabled = walk;
    this.joystick.setVisible(this.enabled && walk && isTouchDevice());
  }

  get gliding() {
    return this.glide !== null;
  }

  /**
   * Glide smoothly to a position (clamped into the walk area), optionally
   * turning to `yaw` on the way. Walking input cancels it.
   */
  glideTo(position: THREE.Vector3, yaw: number | null = null) {
    const to = position.clone();
    this.area.clamp(to);
    const dist = to.distanceTo(this.position);
    const speed = this.settings.speed * this.area.roomSize * 1.5;
    this.glide = {
      from: this.position.clone(),
      to,
      fromYaw: this.yaw,
      // Turn the short way round.
      toYaw: yaw === null ? null : this.yaw + wrapAngle(yaw - this.yaw),
      fromPitch: this.pitch,
      turn: !this.locked, // with a locked mouse the user steers the view
      t: 0,
      duration: THREE.MathUtils.clamp(dist / speed, 0.6, 2.5),
    };
    this.velocity.set(0, 0, 0);
  }

  get locked() {
    return document.pointerLockElement === this.dom;
  }

  /** Swap in a recomputed walk area / settings (keeps the user's position, clamped). */
  setArea(area: WalkArea, settings: PlayerSettings) {
    this.area = area;
    this.settings = settings;
    this.area.clamp(this.position);
  }

  /** Jump to a position and view direction. */
  teleport(position: THREE.Vector3, yaw: number, pitch = REST_PITCH) {
    this.position.copy(position);
    this.area.clamp(this.position);
    this.yaw = yaw;
    this.pitch = pitch;
    this.velocity.set(0, 0, 0);
    this.apply();
  }

  setEnabled(enabled: boolean) {
    this.enabled = enabled;
    this.joystick.setVisible(enabled && this.walkEnabled && isTouchDevice());
    if (!enabled && this.locked) document.exitPointerLock();
    if (!enabled) this.lookPointer = null;
    this.updateCursor();
    if (enabled) this.apply();
  }

  update(dt: number) {
    if (!this.enabled) return;
    dt = Math.min(dt, 0.1); // avoid jumps after a stall

    // Input in the camera's local frame: x = right, y = forward.
    const k = this.keys;
    let inX =
      (k.has("KeyD") || k.has("ArrowRight") ? 1 : 0) -
      (k.has("KeyA") || k.has("ArrowLeft") ? 1 : 0);
    let inY =
      (k.has("KeyW") || k.has("ArrowUp") ? 1 : 0) - (k.has("KeyS") || k.has("ArrowDown") ? 1 : 0);
    const len = Math.hypot(inX, inY);
    if (len > 1) {
      inX /= len;
      inY /= len;
    }
    inX += this.joystick.value.x;
    inY += this.joystick.value.y;
    if (!this.walkEnabled) inX = inY = 0;

    if (this.glide) {
      if (inX !== 0 || inY !== 0)
        this.glide = null; // walking takes over
      else return this.updateGlide(dt);
    }

    const run = k.has("ShiftLeft") || k.has("ShiftRight") ? this.settings.runFactor : 1;
    const speed = this.settings.speed * this.area.roomSize * run;
    const sin = Math.sin(this.yaw);
    const cos = Math.cos(this.yaw);
    // forward = (-sin, -cos), right = (cos, -sin) on the floor plan
    const target = new THREE.Vector3(
      inX * cos - inY * sin,
      0,
      -inX * sin - inY * cos,
    ).multiplyScalar(speed);

    // Smooth start / stop.
    this.velocity.lerp(target, 1 - Math.exp(-dt * 10));
    this.position.addScaledVector(this.velocity, dt);
    this.area.clamp(this.position);
    this.apply();
  }

  private updateGlide(dt: number) {
    const g = this.glide!;
    g.t = Math.min(1, g.t + dt / g.duration);
    const e = g.t * g.t * (3 - 2 * g.t); // smoothstep ease in/out
    this.position.lerpVectors(g.from, g.to, e);
    // Turn toward the hotspot's view and level the gaze (until the user looks around).
    if (g.turn) {
      if (g.toYaw !== null) this.yaw = THREE.MathUtils.lerp(g.fromYaw, g.toYaw, e);
      this.pitch = THREE.MathUtils.lerp(g.fromPitch, REST_PITCH, e);
    }
    if (g.t >= 1) this.glide = null;
    this.apply();
  }

  private look(dYaw: number, dPitch: number) {
    // Looking around during a glide takes over the view direction.
    if (this.glide) this.glide.turn = false;
    this.yaw -= dYaw;
    this.pitch = THREE.MathUtils.clamp(this.pitch - dPitch, -MAX_PITCH, MAX_PITCH);
  }

  private apply() {
    this.camera.position.copy(this.position);
    this.camera.rotation.set(this.pitch, this.yaw, 0, "YXZ");
  }
}

/** Wrap an angle in radians into -π..π. */
function wrapAngle(a: number): number {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

export function isTouchDevice(): boolean {
  return window.matchMedia("(pointer: coarse)").matches;
}
