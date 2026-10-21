import * as THREE from "three";
import GUI from "lil-gui";
import type { Room } from "./room";
import { createCameraMarkers, type ColmapCamera } from "./colmap";
import { saveConfig, type SceneConfig } from "./config";
import type { WalkArea } from "./walkArea";
import type { Hotspots } from "./hotspots";

interface PanelDeps {
  scene: THREE.Scene;
  room: Room;
  area: WalkArea;
  config: SceneConfig;
  cameras: ColmapCamera[];
  /**
   * Re-run the automatic setup after the room transform or player settings
   * changed. `restart` puts the user back at the start position.
   */
  onRoomChanged: (restart: boolean) => void;
  /** Switch between first-person and the debug orbit view. */
  setOrbit: (orbit: boolean) => void;
  hotspots: Hotspots;
  /** Add the current view as a hotspot (same as the H key). */
  addHotspot: () => void;
  goToHotspot: (i: number) => void;
  /** Call after the hotspot list changed. */
  onHotspotsChanged: () => void;
}

/**
 * Debug panel (toggle with C). Everything is automatic; this panel is for the
 * author: flip the room if the automatic "up" is wrong, look at the walk area
 * and the COLMAP cameras, and (Advanced) tweak the settings by hand.
 */
export function createDebugPanel(deps: PanelDeps) {
  const { scene, room, area, config, cameras, onRoomChanged, setOrbit, hotspots } = deps;
  // ---------- debug visuals ----------
  const rawSize = room.bounds().getSize(new THREE.Vector3()).length() / room.calibration.scale;
  const markers = createCameraMarkers(cameras, rawSize * 0.008);
  room.root.add(markers);

  let outline: THREE.LineLoop | null = null;
  function rebuildOutline() {
    if (outline) {
      scene.remove(outline);
      outline.geometry.dispose();
    }
    outline = area.createDebugOutline(area.floorY);
    outline.renderOrder = 999;
    scene.add(outline);
    updateVisibility();
  }

  // ---------- panel ----------
  const gui = new GUI({ title: "Debug (C)", width: 290 });
  const ui = {
    showArea: true,
    showCameras: cameras.length > 0,
    orbit: false,
    info: "",
    status: "",
  };

  gui.add({ flip }, "flip").name("↕ Flip upside down");
  gui
    .add(ui, "showArea")
    .name("Show walk area")
    .onChange(() => updateVisibility());
  if (cameras.length)
    gui
      .add(ui, "showCameras")
      .name("Show COLMAP cameras")
      .onChange(() => updateVisibility());
  gui
    .add(ui, "orbit")
    .name("Orbit view (look from outside)")
    .onChange((v: boolean) => setOrbit(v));
  gui.add(ui, "info").name("Walk area").disable().listen();
  gui.add({ save }, "save").name("💾 Save config.json");
  gui.add(ui, "status").name("").disable().listen();

  // --- Hotspots (tour viewpoints) ---
  const hs = gui.addFolder("Hotspots");
  hs.add({ add: deps.addHotspot }, "add").name("＋ Add current view (H)");
  hs.add(
    {
      regenerate: () => {
        hotspots.generate();
        deps.onHotspotsChanged();
      },
    },
    "regenerate",
  ).name("Regenerate automatic set");
  let list: GUI | null = null;
  function rebuildHotspotList() {
    list?.destroy();
    list = hs.addFolder(`List (${hotspots.list.length})`);
    hotspots.list.forEach((h, i) => {
      const row = list!.addFolder(`${i + 1}. ${h.name}`).close();
      row
        .add(h, "name")
        .name("Name")
        .onFinishChange(() => deps.onHotspotsChanged());
      row.add({ go: () => deps.goToHotspot(i) }, "go").name("Go there");
      row
        .add(
          {
            up: () => {
              if (i === 0) return;
              [hotspots.list[i - 1], hotspots.list[i]] = [hotspots.list[i], hotspots.list[i - 1]];
              deps.onHotspotsChanged();
            },
          },
          "up",
        )
        .name("Move up in tour order");
      row
        .add(
          {
            remove: () => {
              hotspots.remove(i);
              deps.onHotspotsChanged();
            },
          },
          "remove",
        )
        .name("✕ Remove");
    });
  }

  // --- Advanced: manual overrides, for debugging only ---
  const adv = gui.addFolder("Advanced").close();
  const cal = () => room.calibration;
  // Proxy so the sliders always edit the current calibration object.
  const rot = {
    get tiltX() {
      return cal().rotationDeg[0];
    },
    set tiltX(v: number) {
      setRotation(0, v);
    },
    get tiltZ() {
      return cal().rotationDeg[2];
    },
    set tiltZ(v: number) {
      setRotation(2, v);
    },
    get turnY() {
      return cal().rotationDeg[1];
    },
    set turnY(v: number) {
      setRotation(1, v);
    },
  };
  adv.add(rot, "tiltX", -180, 180, 0.1).name("Tilt X°").listen();
  adv.add(rot, "tiltZ", -180, 180, 0.1).name("Tilt Z°").listen();
  adv.add(rot, "turnY", -180, 180, 0.1).name("Turn Y°").listen();
  adv.add({ reset: resetToAuto }, "reset").name("Reset to automatic levelling");
  const p = config.player;
  adv
    .add(p, "eyeOffset", -0.3, 0.3, 0.005)
    .name("Eye offset (× room)")
    .onChange(() => onRoomChanged(false));
  adv.add(p, "speed", 0.05, 1, 0.01).name("Speed (rooms / s)");
  adv.add(p, "runFactor", 1, 5, 0.1).name("Shift run factor");
  adv
    .add(p, "wallDistance", 0, 0.3, 0.005)
    .name("Wall distance (× room)")
    .onChange(() => onRoomChanged(false));

  // ---------- actions ----------
  function flip() {
    if (config.calibration) {
      room.flip(); // manual mode: flip the manual rotation
    } else {
      config.flip = !config.flip;
      room.autoCalibrate(cameras, config.flip);
    }
    onRoomChanged(true);
  }

  function setRotation(axis: number, deg: number) {
    room.calibration.rotationDeg[axis] = deg;
    room.applyCalibration();
    config.calibration = room.calibration; // now a manual calibration
    onRoomChanged(false);
  }

  function resetToAuto() {
    delete config.calibration;
    room.autoCalibrate(cameras, config.flip);
    onRoomChanged(true);
  }

  async function save() {
    const result = await saveConfig(config);
    ui.status = result === "saved" ? "Saved public/config.json" : "Downloaded: copy into public/";
    setTimeout(() => (ui.status = ""), 4000);
  }

  // ---------- show / hide ----------
  let open = false;

  function updateVisibility() {
    if (outline) outline.visible = open && ui.showArea;
    markers.visible = open && ui.showCameras;
  }

  function toggle(show = !open) {
    open = show;
    gui.show(show);
    if (!show && ui.orbit) {
      ui.orbit = false;
      setOrbit(false);
    }
    updateVisibility();
  }

  /** Call after the walk area was recomputed. */
  function refresh() {
    const margin = (area.margin / area.roomSize).toFixed(3);
    ui.info = `size ${area.roomSize.toFixed(2)} · margin ${margin}${config.calibration ? " · manual" : ""}`;
    rebuildOutline();
  }

  refresh();
  rebuildHotspotList();
  toggle(false);
  window.addEventListener("keydown", (e) => {
    if (e.code === "KeyC" && !(e.target instanceof HTMLInputElement)) toggle();
  });

  return {
    toggle,
    refresh,
    refreshHotspots: rebuildHotspotList,
    get open() {
      return open;
    },
  };
}
