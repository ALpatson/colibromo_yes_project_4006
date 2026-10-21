# viewer/: room splat viewer

First-person web viewer for a Gaussian Splat of a real room (desktop + phone),
prototype for an immersive before/after renovation tour. Started as a personal
prototype before Step 4; Step 4 builds on it.

- Vite + TypeScript, static site, no backend.
- Rendering: [Spark](https://sparkjs.dev) 2.x (`@sparkjsdev/spark`) on three.js.
- Setup is automatic: levelling, eye height, walking speed, walk area and start
  view all come from the COLMAP camera poses. No measuring or real-world scale needed.

## Run it

Requires Node.js 18+ (tested with Node 22). Everything is installed locally in
this folder; nothing global.

```sh
cd viewer
npm install      # once
npm run dev      # dev server, reachable from other devices on the network
npm test         # type check + unit tests (vitest), as CI runs them
```

Vite prints a `Local` address (for this computer) and several `Network`
addresses.

**Room data is not in git.** `public/runs/`, `public/splats/` and `public/colmap/`
are ignored: splats show private interiors and are 60-700 MB. Get a room in with
`cpipe ui` → *Open in viewer* (copies the run's `scene.ply` + `cameras.json` to
`public/runs/<room>/` and opens `?run=runs/<room>`), or copy them by hand (see
"New scan").

### Opening it on a phone

The phone has to reach the computer over the network. On school/office Wi-Fi
this is often blocked (devices can't see each other: `ERR_ADDRESS_UNREACHABLE`).
What works:

- **USB tethering (Android):** plug the phone in, then Settings → USB → *USB tethering*.
  The computer gets a new network address (e.g. `10.253.x.x`); open
  `http://<that address>:5173/` on the phone. No USB debugging needed.
- **Phone hotspot:** connect the computer to the phone's hotspot and use the
  address Vite prints for Wi-Fi.
- On a home network: just use the address Vite prints next to `Wi-Fi`.

Windows Firewall must allow "Node.js JavaScript Runtime" on that network type.

### Performance on laptops with two GPUs

Chrome may render on the integrated GPU (the HUD at the top left shows which
one). For the dedicated one: Windows Settings → System → Display → Graphics →
Google Chrome → *High performance*, then restart Chrome.

### Build

```sh
npm run build    # static site in dist/ (copy it to any static host)
npm run preview  # serve dist/ locally to test it
```

## Controls

| | Desktop | Phone |
| --- | --- | --- |
| Look | drag with the left mouse button (cursor stays visible) | drag anywhere |
| Walk | WASD or arrow keys, Shift to run | joystick, bottom left |
| Jump to a viewpoint | click a circle on the floor | tap a circle |
| Tour | ◀ ▶ (bottom right), or ← → in Tour mode | ◀ ▶ |
| Optional | **L**: game-style locked mouse (move the mouse to look); **Esc** to free it | |

**Walk / Tour** (bottom right): *Walk* is free walking; the circles can still be
clicked. *Tour* is like a hotel/campus 360 tour: no walking, you go from
viewpoint to viewpoint (glide) and look around freely at each one.

Debug keys: **C** opens the debug panel; **H** adds the current view as a hotspot;
**K** cycles the spherical-harmonics level 3 → 0 → 1 → 2 (colour check).

## Hotspots (tour viewpoints)

- Stored in `config.json` → `hotspots`, in raw COLMAP coordinates, so they stay
  correct after a flip or recalibration:
  ```json
  "hotspots": [
    { "name": "Centre", "position": [0.1, -0.2, 0.3], "look": [0.9, 0.0, -0.4] }
  ]
  ```
  `position`: only the floor-plan position matters (the eye height is automatic,
  and the position is kept inside the walk area). `look` (optional): the view
  direction to turn to on arrival.
- If `hotspots` is missing, a starter set is generated: one spot toward each
  corner of the walk area (following the room's main axis), each looking across
  the room through the centre, and the centre itself, looking along the room.
- To make your own: walk to a spot, look in the direction you want to show, press
  **H**. Rename, reorder, remove or regenerate them in the debug panel (C) →
  Hotspots, then **Save config.json**.
- The circles are placed on an estimated floor height (a low percentile of the
  splats inside the walk area). The default gaze is tilted 11° down so nearby
  circles are in view.
- `startMode` in the config: `"walk"` (default) or `"tour"`.

## How the automatic setup works

Everything is computed on load from the camera poses (`cameras`, normally the
pipeline's `cameras.json`; see "New scan" below):

- **Level:** the videos were filmed with the phone upright, so the cameras'
  average "up" direction is the room's up. The room is rotated so it points up.
  If it ever comes out upside down, use **Flip upside down** in the debug panel
  and save.
- **Units:** the scene stays in COLMAP units. Nothing needs metres.
- **Eye height:** the average height of the `middle_*` cameras.
- **Room size:** the larger side of the floor-plan outline of the middle cameras.
  Walking speed is relative to it (`speed` room sizes per second).
- **Walk area:** the outline (convex hull) of the middle camera positions on
  the floor plan. People film a room by walking along the walls facing inward,
  so this outline touches the walls. It is therefore shrunk automatically until
  every edge keeps `wallDistance` × room size from the splats at eye height
  (walls, tall furniture). You slide along its edge, and you stay where the splat
  was filmed from and looks good.
  Limitation: the outline is convex, so in an L-shaped room the missing corner
  would be included.
- **Start:** a middle camera that is away from the centre and was looking into
  the room, turned to face the centre.

## Config (`public/config.json`)

Paths are relative to `public/`.

```json
{
  "splats": {
    "before": "splats/scene.ply"
  },
  "cameras": "colmap/cameras.json",
  "flip": false,
  "player": {
    "eyeGroup": "middle",
    "eyeOffset": 0,
    "speed": 0.3,
    "runFactor": 2.5,
    "wallDistance": 0.15,
    "walkMargin": null
  }
}
```

| Key | Meaning |
| --- | --- |
| `run` | *(optional)* a pipeline run folder inside `public/`, e.g. `runs/myRoom2`: uses its `scene.ply` and `cameras.json` (overrides `splats.before` and `cameras`). Also possible in the URL: `?run=runs/myRoom2` |
| `splats.before` | the splat file |
| `cameras` | camera poses (needed for the automatic setup): the pipeline's `cameras.json`, or the JSON made by `npm run colmap`; for old runs also COLMAP's `images.bin` directly (a `.bin` path, or a sparse folder ending in `/`, e.g. `runs/old/sparse/0/`). `colmap` is the old name of this key and still works |
| `flip` | turn the automatically levelled room upside down |
| `player.eyeGroup` | camera group whose average height is the eye height (`middle` = `middle_*` frames) |
| `player.eyeOffset` | raise/lower the eye, as a fraction of the room size |
| `player.speed` | walking speed in room sizes per second |
| `player.runFactor` | Shift multiplies the speed by this |
| `player.wallDistance` | minimum distance to the walls, as a fraction of the room size |
| `player.walkMargin` | `null` = automatic. A number forces the margin around the camera outline (fraction of room size, negative = inset) |
| `calibration` | *(optional, debug)* a manual rotation/scale/offset that replaces the automatic levelling. Written when you use the Advanced sliders and save |

## Debug panel (C)

- **Flip upside down**: if the automatic up is wrong.
- **Show walk area / COLMAP cameras**: yellow outline (drawn near the floor),
  and cones at each camera (orange bottom, green middle, blue up).
- **Orbit view**: look at the room and the walk area from outside.
- **Save config.json**: on `npm run dev` from this computer it writes
  `public/config.json` directly. Elsewhere (built site, phone) it downloads the file,
  which you copy into `public/` by hand.
- **Advanced** (closed by default, for debugging): manual tilt/turn sliders
  (switches to a manual calibration; *Reset to automatic levelling* undoes it),
  eye offset, speed, run factor, wall distance.

## New scan

**From the team pipeline (`cpipe`)**: a run folder contains `scene.ply` and
`cameras.json` (camera poses already in the splat's coordinates).

1. Copy both files into `public/runs/<name>/` (e.g. `public/runs/myRoom2/`).
2. Open `http://localhost:5173/?run=runs/<name>`, or set `"run": "runs/<name>"` in
   `config.json` to make it the default.
3. If the room is upside down, press C → Flip → Save.

When the URL switches to another run, hotspots and manual calibration saved in
`config.json` for a different scene are ignored (they belong to that scene).

**By hand (Brush + COLMAP without the pipeline):**

1. Copy the splat (.ply from Brush) to `public/splats/` and set `splats.before`.
2. Camera poses, either:
   - convert the COLMAP model to JSON (reads `images.bin` directly, no COLMAP install
     needed): `npm run colmap -- C:\path\to\sparse\0`, which writes
     `public/colmap/cameras.json`; or
   - copy COLMAP's `sparse/0/images.bin` into `public/` and point `cameras` at it
     (the browser reads it).
   Frame names must start with the group: `bottom_*`, `middle_*`, `up_*`.
   Only the group set in `player.eyeGroup` is required. COLMAP's own coordinates
   only match the splat for trainers that keep them (Brush); gsplat runs need the
   pipeline's `cameras.json`.
3. Open the viewer. If the room is upside down, press C → Flip → Save.

## Files

```
public/
  config.json           settings (see above)
  splats/               (not in git) splats added by hand
  colmap/cameras.json   (not in git) camera poses made by `npm run colmap`
  runs/<room>/          (not in git) a pipeline run: scene.ply + cameras.json (?run=runs/<room>)
scripts/
  colmap-to-json.mjs    reads COLMAP images.bin → public/colmap/cameras.json
src/
  main.ts               app entry: renderer, loading, wiring, render loop
  config.ts             config.json types, loading and saving
  splat.ts              download with progress, SplatMesh creation, robust bounds
  room.ts               room group + transform, automatic levelling, flip
  colmap.ts             camera poses: loading (cameras.json or images.bin), average "up", debug markers
  walkArea.ts           eye height, room size, walk area, start position
  firstPerson.ts        first-person controls (mouse/keyboard and touch)
  joystick.ts           on-screen joystick for touch screens
  hotspots.ts           tour viewpoints: floor markers, picking, starter set
  tourBar.ts            Walk / Tour switch and ◀ ▶ buttons
  debugPanel.ts         debug panel (C key)
  loadingOverlay.ts     loading screen / progress bar
  snapshot.ts           snapshot mode (quality check, see below)
  style.css
tests/                  vitest unit tests (camera-pose parsing, snapshot settings)
vite.config.ts          dev server + dev-only "save config.json" endpoint
```

## Snapshot mode (quality check)

Renders the splat from exactly where one frame was filmed (pose and field of
view from `cameras.json`, at the frame's size, one canvas pixel per image pixel)
and downloads the image plus an info file (GPU, drawing size, SH degree, Spark
settings). The pipeline compares it with gsplat's render of the same view
(`bench.compare_viewer`; `docs/benchmarks.md`, "Viewer check").

```
?run=runs/<room>&snap=bottom_00000.jpg        Spark's defaults
...&blur=classic   blur as in classic 3DGS / gsplat "default" training
...&blur=aa        Spark's default (for anti-aliased training)
...&sh=0           no view-dependent colour       ...&lod=0   no level of detail
```

## Colours

The splat is drawn with three's default sRGB output and no tone mapping, so the
colours are the trained values written straight to the screen, which is what
Brush shows. Brush writes the `f_rest_*` SH properties in alphabetical order
(0, 1, 10, 11, …). Spark looks properties up by name, so this order is fine.
A loader that reads them by position would scramble the view-dependent colour
and give a grainy look.

## Roadmap

1. **Load and view.** *(done)*
2. **Orientation.** Automatic levelling from the COLMAP poses, flip button, debug panel. *(done)*
3. **First-person movement.** Mouse/WASD and touch/joystick, eye height and
   walk area from the cameras, start at a middle camera. *(done)*
4. **Tour mode.** Hotspots in config (automatic starter set), H to add one,
   click/tap to glide there, Walk/Tour switch. *(done)*
