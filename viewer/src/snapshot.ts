import * as THREE from "three";
import type { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { publicUrl } from "./config";
import type { Room } from "./room";

/**
 * Snapshot mode, to measure whether the viewer loses quality: render the splat
 * from exactly where a frame was filmed (pose and field of view from the
 * pipeline's cameras.json), at the frame's size, and download the image plus an
 * info file (GPU, drawing size, SH degree, renderer settings). The pipeline's
 * evaluator saves the same view rendered by gsplat ("*_render_pinhole.png").
 *
 *   ?run=runs/myRoom2&snap=bottom_00000.jpg              (Spark defaults)
 *   &blur=classic   splat blur as in classic 3DGS / gsplat "default" training
 *   &blur=aa        Spark's default (for anti-aliased training)
 *   &sh=0..3        spherical-harmonics degree     &lod=0   no level of detail
 */
export interface SnapshotParams {
  frame: string;
  blur: "default" | "aa" | "classic";
  sh: number | null;
  lod: boolean;
}

export function snapshotParams(): SnapshotParams | null {
  const q = new URLSearchParams(window.location.search);
  const frame = q.get("snap");
  if (!frame) return null;
  const blur = q.get("blur");
  return {
    frame,
    blur: blur === "aa" || blur === "classic" ? blur : "default",
    sh: q.has("sh") ? Number(q.get("sh")) : null,
    lod: q.get("lod") !== "0",
  };
}

/** SparkRenderer options for the snapshot's settings. */
export function sparkOptions(p: SnapshotParams | null) {
  if (!p) return {};
  const blur =
    p.blur === "classic"
      ? { blurAmount: 0, preBlurAmount: 0.3 }
      : p.blur === "aa"
        ? { blurAmount: 0.3, preBlurAmount: 0 }
        : {};
  return { ...blur, ...(p.lod ? {} : { enableLod: false }) };
}

interface CamerasJson {
  intrinsics: Record<string, { model: string; width: number; height: number; params: number[] }>;
  cameras: {
    name: string;
    camera_id: number;
    position: number[];
    up: number[];
    forward: number[];
  }[];
}

function download(blob: Blob, name: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

export async function runSnapshot(
  p: SnapshotParams,
  deps: {
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    room: Room;
    splat: SplatMesh;
    spark: SparkRenderer;
    camerasPath: string;
    gpuName: () => string;
  },
) {
  const { renderer, scene, camera, room, splat, spark } = deps;
  const data = (await (
    await fetch(publicUrl(deps.camerasPath), { cache: "no-store" })
  ).json()) as CamerasJson;
  const cam = data.cameras.find((c) => c.name === p.frame);
  if (!cam) throw new Error(`No camera "${p.frame}" in ${deps.camerasPath}`);
  const intr = data.intrinsics[String(cam.camera_id)];
  const [fx, fy, cx, cy] =
    intr.model.startsWith("SIMPLE_") || intr.model.startsWith("RADIAL")
      ? [intr.params[0], intr.params[0], intr.params[1], intr.params[2]]
      : intr.params.slice(0, 4);
  const { width: w, height: h } = intr;

  // Exactly the frame's size: one canvas pixel per image pixel.
  renderer.setPixelRatio(1);
  renderer.setSize(w, h);
  if (p.sh !== null) {
    splat.maxSh = p.sh;
    splat.updateGenerator();
  }

  // Pinhole camera: vertical field of view from fy, principal point (cx, cy).
  camera.aspect = w / h;
  camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(h / 2 / fy));
  camera.near = 0.01;
  camera.far = 1000;
  camera.updateProjectionMatrix();
  camera.projectionMatrix.elements[0] = (2 * fx) / w;
  camera.projectionMatrix.elements[8] = 1 - (2 * cx) / w;
  camera.projectionMatrix.elements[9] = (2 * cy) / h - 1;
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();

  // OpenCV camera (x right, y down, z forward) -> three.js (x right, y up, looking -z),
  // in the room's levelled world.
  const up = new THREE.Vector3().fromArray(cam.up);
  const forward = new THREE.Vector3().fromArray(cam.forward);
  const right = new THREE.Vector3().crossVectors(up.clone().negate(), forward);
  const raw = new THREE.Matrix4()
    .makeBasis(right, up, forward.clone().negate())
    .setPosition(new THREE.Vector3().fromArray(cam.position));
  const world = room.root.matrixWorld.clone().multiply(raw);
  world.decompose(camera.position, camera.quaternion, new THREE.Vector3());
  camera.updateMatrixWorld(true);
  // Keep the projection we built (three would recompute it on resize).
  camera.updateProjectionMatrix = () => {};

  // Let Spark sort the splats for this view, then capture right after a render.
  const settle = performance.now() + 2500;
  await new Promise<void>((resolve) => {
    renderer.setAnimationLoop(() => {
      renderer.render(scene, camera);
      if (performance.now() < settle) return;
      renderer.setAnimationLoop(null);
      renderer.domElement.toBlob((blob) => {
        const base = `viewer_${p.frame.replace(/\.\w+$/, "")}_blur-${p.blur}_sh${splat.maxSh}${p.lod ? "" : "_nolod"}`;
        if (blob) download(blob, `${base}.png`);
        const info = {
          frame: p.frame,
          gpu: deps.gpuName(),
          drawing_size: renderer.getDrawingBufferSize(new THREE.Vector2()).toArray(),
          sh_degree: splat.maxSh,
          num_splats: splat.packedSplats?.numSplats ?? null,
          spark: {
            blurAmount: (spark as unknown as { blurAmount: number }).blurAmount,
            preBlurAmount: (spark as unknown as { preBlurAmount: number }).preBlurAmount,
            enableLod: (spark as unknown as { enableLod: boolean }).enableLod,
          },
          antialias: renderer.getContextAttributes()?.antialias ?? null,
          user_agent: navigator.userAgent,
        };
        download(
          new Blob([JSON.stringify(info, null, 1)], { type: "application/json" }),
          `${base}.json`,
        );
        document.title = "snapshot saved";
        resolve();
      }, "image/png");
    });
  });
}
