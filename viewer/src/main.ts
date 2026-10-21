import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SparkRenderer, isMobile } from "@sparkjsdev/spark";
import { loadConfig, publicUrl } from "./config";
import { loadSplat } from "./splat";
import { loadingOverlay } from "./loadingOverlay";
import { Room } from "./room";
import { loadCameras } from "./colmap";
import { WalkArea } from "./walkArea";
import { FirstPersonControls, isTouchDevice } from "./firstPerson";
import { createDebugPanel } from "./debugPanel";
import { Hotspots } from "./hotspots";
import { createTourBar, type Mode } from "./tourBar";
import { runSnapshot, snapshotParams, sparkOptions } from "./snapshot";

/** ?snap=<frame>: render one frame's exact view and download it (quality check). */
const snap = snapshotParams();

// ---------- renderer / scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: false });
// Splat rendering is fill-rate heavy: cap the pixel ratio, lower on phones.
renderer.setPixelRatio(Math.min(window.devicePixelRatio, isMobile() ? 1.5 : 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000);
// three's default outputColorSpace is sRGB. Spark then writes the splat colours
// unchanged (no linear conversion), which is how the reference 3DGS renderer and
// Brush display them. Do not add tone mapping: it would shift the colours.
renderer.toneMapping = THREE.NoToneMapping;
document.getElementById("app")!.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 100);

const spark = new SparkRenderer({ renderer, ...sparkOptions(snap) });
scene.add(spark);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------- HUD (fps + debug info) ----------
const hud = document.getElementById("hud")!;
const hint = document.getElementById("hint")!;
const crosshair = document.getElementById("crosshair")!;

/** Which GPU the browser renders with (laptops may pick the integrated one). */
function gpuName(): string {
  const gl = renderer.getContext();
  const info = gl.getExtension("WEBGL_debug_renderer_info");
  const name = String(
    info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
  );
  // "ANGLE (NVIDIA, NVIDIA T500 Direct3D11 ...)" → "NVIDIA T500"
  return name
    .replace(/^ANGLE \([^,]+, /, "")
    .replace(/ \(0x[0-9a-f]+\)| Direct3D.*|,.*/gi, "")
    .trim();
}
const gpuInfo = () => {
  const size = renderer.getDrawingBufferSize(new THREE.Vector2());
  return `${gpuName()} · ${size.x}×${size.y}`;
};

let frames = 0;
let lastFpsTime = performance.now();
let hudInfo = "";

function updateHud(now: number) {
  frames++;
  if (now - lastFpsTime >= 500) {
    const fps = (frames * 1000) / (now - lastFpsTime);
    frames = 0;
    lastFpsTime = now;
    hud.textContent = `${fps.toFixed(0)} fps\n${hudInfo}`;
  }
}

// Set once loading is done.
let update: (dt: number) => void = () => {};

// ---------- load ----------
async function start() {
  const config = await loadConfig();
  loadingOverlay.show();

  let splat;
  try {
    splat = await loadSplat(publicUrl(config.splats.before), loadingOverlay.progress);
  } catch (err) {
    console.error(err);
    loadingOverlay.error(String(err instanceof Error ? err.message : err));
    return;
  }
  const cameras = config.cameras ? await loadCameras(config.cameras) : [];
  if (config.cameras && !cameras.length) {
    console.warn("No camera poses: the room is not levelled automatically.");
  }

  // Level the room automatically from the camera poses (unless a manual
  // calibration was saved from the debug panel).
  const room = new Room(splat);
  if (config.calibration) room.setCalibration(config.calibration);
  else if (cameras.length) room.autoCalibrate(cameras, config.flip);
  else
    room.setCalibration({
      rotationDeg: [config.flip ? 0 : 180, 0, 0],
      scale: 1,
      offset: [0, 0, 0],
    });
  scene.add(room.root);

  if (snap) {
    loadingOverlay.hide();
    hint.classList.add("hidden");
    if (!config.cameras?.endsWith(".json")) {
      loadingOverlay.error("Snapshot mode needs the run's cameras.json.");
      return;
    }
    try {
      await runSnapshot(snap, {
        renderer,
        scene,
        camera,
        room,
        splat,
        spark,
        camerasPath: config.cameras,
        gpuName,
      });
    } catch (err) {
      loadingOverlay.error(String(err instanceof Error ? err.message : err));
    }
    return;
  }

  // Eye height, walk area, speed and start position all come from the cameras.
  const area = new WalkArea(room, cameras, config.player);
  camera.near = area.roomSize * 0.002;
  camera.far = area.roomSize * 20;
  camera.updateProjectionMatrix();

  const fp = new FirstPersonControls(camera, renderer.domElement, area, config.player);
  fp.teleport(area.start.position, area.start.yaw);

  // Debug orbit view (from the panel): look at the room and walk area from outside.
  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.enableDamping = true;
  orbit.enabled = false;
  fp.updateCursor(); // OrbitControls reset it to "auto"
  function setOrbit(on: boolean) {
    fp.setEnabled(!on);
    orbit.enabled = on;
    if (on) {
      const c = new THREE.Vector3(area.centre.x, area.eyeY, area.centre.y);
      orbit.target.copy(c);
      camera.position.copy(c).add(new THREE.Vector3(0, area.roomSize * 0.9, area.roomSize * 0.9));
      orbit.update();
    }
    updateHint();
  }

  // ---------- hotspots + tour ----------
  // No hotspots in the config yet: generate a starter set (kept if you Save).
  const hotspots = new Hotspots(config.hotspots ?? [], room, area);
  if (!config.hotspots) config.hotspots = hotspots.generate();
  scene.add(hotspots.group);

  let mode: Mode = config.startMode;
  /** Hotspot the tour is at / heading to (for ◀ ▶). */
  let tourIndex = -1;

  function goToHotspot(i: number) {
    if (i < 0 || i >= hotspots.list.length) return;
    tourIndex = i;
    const pose = hotspots.pose(i);
    fp.glideTo(pose.position, pose.yaw);
    updateTourLabel();
  }

  function step(delta: number) {
    const n = hotspots.list.length;
    if (!n) return;
    const from = tourIndex >= 0 ? tourIndex : hotspots.nearest(fp.position) - delta;
    goToHotspot((((from + delta) % n) + n) % n);
  }

  function setMode(m: Mode) {
    mode = m;
    fp.setWalkEnabled(m === "walk");
    tourBar.setMode(m);
    // Tour mode always stands on a hotspot.
    if (m === "tour" && hotspots.current < 0) goToHotspot(hotspots.nearest(fp.position));
    updateHint();
  }

  function updateTourLabel() {
    const i = hotspots.current >= 0 ? hotspots.current : fp.gliding ? tourIndex : -1;
    tourBar.setLabel(i >= 0 ? `${i + 1} / ${hotspots.list.length} · ${hotspots.list[i].name}` : "");
  }

  function onHotspotsChanged() {
    hotspots.rebuild();
    tourIndex = -1;
    tourBar.setHasHotspots(hotspots.list.length > 0);
    panel.refreshHotspots();
    updateTourLabel();
  }

  function addHotspot() {
    const i = hotspots.add(`Spot ${hotspots.list.length + 1}`, fp.position, fp.yaw);
    onHotspotsChanged();
    flash(`Added "${hotspots.list[i].name}". Save it in the debug panel (C).`);
  }

  const tourBar = createTourBar({ onMode: setMode, onPrev: () => step(-1), onNext: () => step(1) });
  tourBar.setHasHotspots(hotspots.list.length > 0);

  // Click / tap a marker to glide there.
  fp.onTap = (x, y) => {
    const i = hotspots.pick(x, y, camera, isTouchDevice() ? 60 : 40);
    if (i >= 0) goToHotspot(i);
  };
  // Mouse hover: highlight the marker and show a pointing hand.
  renderer.domElement.addEventListener("pointermove", (e) => {
    if (e.pointerType !== "mouse" || e.buttons || fp.locked || !fp.enabled) return;
    hotspots.hovered = hotspots.pick(e.clientX, e.clientY, camera, 40);
    fp.setHover(hotspots.hovered >= 0);
  });

  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement || !fp.enabled) return;
    if (e.code === "KeyH") addHotspot();
    // In tour mode the arrow keys step through the viewpoints.
    if (mode === "tour" && (e.code === "ArrowRight" || e.code === "ArrowLeft"))
      step(e.code === "ArrowRight" ? 1 : -1);
  });

  const panel = createDebugPanel({
    scene,
    room,
    area,
    config,
    cameras,
    setOrbit,
    hotspots,
    addHotspot,
    goToHotspot,
    onHotspotsChanged,
    onRoomChanged(restart) {
      area.update(config.player);
      fp.setArea(area, config.player);
      hotspots.setArea(area);
      if (restart) fp.teleport(area.start.position, area.start.yaw);
      panel.refresh();
    },
  });

  // ---------- hints ----------
  let flashTimer = 0;
  /** Show a message in the hint bubble for a few seconds. */
  function flash(text: string) {
    hint.textContent = text;
    hint.classList.remove("hidden");
    clearTimeout(flashTimer);
    flashTimer = window.setTimeout(updateHint, 3500);
  }

  let touchHintShown = true;
  function updateHint() {
    let text = "";
    if (orbit.enabled) text = "Orbit view: drag to rotate, scroll to zoom";
    else if (isTouchDevice()) {
      if (touchHintShown)
        text =
          mode === "tour"
            ? "Drag to look around · tap a circle or ▶"
            : "Drag to look around · joystick to walk · tap a circle to jump";
    } else if (fp.locked) text = "Mouse locked · Esc to free the mouse";
    else if (mode === "tour") text = "Drag to look around · click a circle or ◀ ▶";
    else text = "Drag to look · WASD to walk · click a circle to jump · L: lock mouse";
    hint.textContent = text;
    hint.classList.toggle("hidden", !text);
    crosshair.classList.toggle("hidden", !fp.locked);
  }
  fp.onLockChange = updateHint;
  setMode(mode);
  if (isTouchDevice()) {
    setTimeout(() => {
      touchHintShown = false;
      updateHint();
    }, 6000);
  }

  // Debug key: K cycles the spherical-harmonics level 3 → 0 → 1 → 2 → 3.
  const shInfo = () =>
    `${gpuInfo()}\n${splat.packedSplats?.numSplats.toLocaleString() ?? "?"} splats · SH ${splat.maxSh} (K)\nC: debug panel`;
  hudInfo = shInfo();
  window.addEventListener("keydown", (e) => {
    if (e.code === "KeyK") {
      splat.maxSh = (splat.maxSh + 1) % 4;
      splat.updateGenerator();
      hudInfo = shInfo();
    }
  });

  let lastCurrent = -2;
  update = (dt) => {
    if (orbit.enabled) orbit.update();
    else fp.update(dt);
    hotspots.update(fp.position, performance.now() / 1000);
    if (hotspots.current !== lastCurrent) {
      lastCurrent = hotspots.current;
      if (hotspots.current >= 0) tourIndex = hotspots.current;
      updateTourLabel();
    }
  };

  hud.classList.remove("hidden");
  loadingOverlay.hide();

  // Handy in the browser console while debugging.
  Object.assign(window, { THREE, scene, camera, room, area, fp, spark, config, cameras, hotspots });
}

const timer = new THREE.Timer();
renderer.setAnimationLoop((time) => {
  timer.update(time);
  update(timer.getDelta());
  renderer.render(scene, camera);
  updateHud(time);
});

start();
