// Scene configuration, loaded from public/config.json at startup.
// Paths inside the config are relative to the site root (the public/ folder).

export interface Calibration {
  /** Euler angles in degrees, applied in "YXZ" order: X/Z level the room, Y turns it. */
  rotationDeg: [number, number, number];
  /** Uniform scale applied to the raw COLMAP coordinates. */
  scale: number;
  /** Translation applied after rotation and scale. */
  offset: [number, number, number];
}

export interface PlayerSettings {
  /** Camera group whose average height is the eye height ("middle" = middle_* frames). */
  eyeGroup: string;
  /** Extra eye height, as a fraction of the room size (0 = exactly the eye group's height). */
  eyeOffset: number;
  /** Walking speed in room sizes per second (Shift multiplies it by `runFactor`). */
  speed: number;
  runFactor: number;
  /**
   * Minimum distance to the walls (splats at eye height), as a fraction of the
   * room size. The walk area is shrunk automatically until it is respected.
   */
  wallDistance: number;
  /**
   * Manual margin around the camera footprint (fraction of the room size,
   * negative = inset). null = automatic, from `wallDistance`.
   */
  walkMargin: number | null;
}

/** A tour viewpoint. Stored in raw COLMAP coordinates, so it survives flipping / recalibrating. */
export interface Hotspot {
  name: string;
  /** Raw COLMAP position (only the floor-plan position matters; the eye height is automatic). */
  position: [number, number, number];
  /** Raw COLMAP view direction to turn to on arrival (optional). */
  look?: [number, number, number];
}

export interface SceneConfig {
  splats: {
    before: string;
    after?: string;
  };
  /**
   * A pipeline run folder (inside public/), e.g. "runs/myRoom2": uses its
   * scene.ply and cameras.json. Can also be given in the URL: ?run=runs/myRoom2
   */
  run?: string;
  /**
   * Camera poses, needed for the automatic setup. One of:
   * - the pipeline's cameras.json (export step), or the old `npm run colmap` JSON;
   * - for old runs without cameras.json: a COLMAP images.bin file, or a
   *   sparse model folder ending in "/" (its images.bin is read).
   */
  cameras?: string;
  /** Old name of `cameras` (still read). */
  colmap?: string;
  /** Turn the automatically levelled room upside down (if the automatic "up" is wrong). */
  flip: boolean;
  /**
   * Manual calibration (debug). When absent, the room is levelled automatically
   * from the camera poses on every load, which is the normal case.
   */
  calibration?: Calibration;
  player: PlayerSettings;
  /** Tour viewpoints. When absent, a starter set is generated automatically (and saved with Save). */
  hotspots?: Hotspot[];
  /** Mode on load: "walk" (free walking + hotspots) or "tour" (hotspots only, look around). */
  startMode: "walk" | "tour";
}

export const DEFAULT_PLAYER: PlayerSettings = {
  eyeGroup: "middle",
  eyeOffset: 0,
  speed: 0.3,
  runFactor: 2.5,
  wallDistance: 0.15,
  walkMargin: null,
};

export const DEFAULT_CONFIG: SceneConfig = {
  splats: { before: "splats/scene.ply" },
  cameras: "colmap/cameras.json",
  flip: false,
  player: DEFAULT_PLAYER,
  startMode: "walk",
};

/** Resolve a public/ path against Vite's base URL, so it works in dev and in a sub-folder build. */
export function publicUrl(path: string): string {
  return import.meta.env.BASE_URL + path.replace(/^\/+/, "");
}

export async function loadConfig(): Promise<SceneConfig> {
  try {
    // no-store: always pick up the latest edits to config.json
    const res = await fetch(publicUrl("config.json"), { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = (await res.json()) as Partial<SceneConfig>;
    return applyRun({
      ...structuredClone(DEFAULT_CONFIG),
      ...json,
      cameras: json.cameras ?? json.colmap ?? DEFAULT_CONFIG.cameras,
      splats: { ...DEFAULT_CONFIG.splats, ...json.splats },
      player: { ...DEFAULT_PLAYER, ...json.player },
    });
  } catch (err) {
    console.warn("config.json not loaded, using defaults:", err);
    return applyRun(structuredClone(DEFAULT_CONFIG));
  }
}

/**
 * A run folder (from ?run=... in the URL, else config.run) supplies the splat
 * and camera poses. Hotspots and a manual calibration saved for another scene
 * don't fit a different run, so they are dropped when the URL switches runs.
 */
function applyRun(config: SceneConfig): SceneConfig {
  const fromUrl = new URLSearchParams(window.location.search).get("run");
  const run = (fromUrl ?? config.run)?.replace(/\/+$/, "");
  if (!run) return config;
  if (fromUrl && fromUrl.replace(/\/+$/, "") !== config.run?.replace(/\/+$/, "")) {
    delete config.hotspots;
    delete config.calibration;
  }
  config.run = run;
  config.splats = { ...config.splats, before: `${run}/scene.ply` };
  config.cameras = `${run}/cameras.json`;
  return config;
}

/**
 * Save the config. With `npm run dev` on this computer, the dev server writes
 * public/config.json directly. Otherwise (built site, or a phone) the file is
 * downloaded so it can be copied into public/ by hand.
 */
export async function saveConfig(config: SceneConfig): Promise<"saved" | "downloaded"> {
  // Round to 4 decimals so the file stays readable.
  const round = (_key: string, v: unknown) =>
    typeof v === "number" ? Math.round(v * 1e4) / 1e4 : v;
  const text = JSON.stringify(config, round, 2) + "\n";
  if (import.meta.env.DEV) {
    try {
      const res = await fetch("/__save-config", { method: "POST", body: text });
      if (res.ok) return "saved";
      console.warn("Dev save refused:", res.status, await res.text());
    } catch (err) {
      console.warn("Dev save failed:", err);
    }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  a.download = "config.json";
  a.click();
  URL.revokeObjectURL(a.href);
  return "downloaded";
}
