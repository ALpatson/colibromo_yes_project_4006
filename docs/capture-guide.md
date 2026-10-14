# Capture guide: how to film a room

> Updated 2026-10-13 with what worked in our first real tests (myRoom2, a kitchen:
> 3 videos, 300 of 300 frames placed by COLMAP, a clean splat with Brush). Still to
> be confirmed on more rooms in Step 7.

## The short version

Film **three videos of the same room, one per height**, walking slowly along the walls
while facing into the room, all in **landscape at 30 fps**. Name them `bottom.mp4`,
`middle.mp4` and `up.mp4` (or label them so in `cpipe ui`), and give all three to one run:

```bash
cpipe run --video bottom.mp4 --video middle.mp4 --video up.mp4 --output data/runs/room
```

## Before filming

- **Lights on**, curtains as you want them in the result, and the same for all passes.
- **Nobody moving in the room** (people, pets, a TV playing, a fan turning): anything that
  moves between frames becomes blurry "ghosts" or breaks pose estimation. Ask people to
  leave or stand behind you.
- **Lock exposure and focus** so brightness doesn't jump while you turn from a window to
  a dark corner:
  - iPhone: in the Camera app, press and hold on the screen until **AE/AF LOCK** appears.
  - Android (Samsung): in Video mode tap the screen, then tap the lock icon (or use Pro
    Video mode with fixed exposure).
- **1080p or 4K, 30 fps.** Keep the same settings (resolution, orientation, lens: no zoom,
  no ultra-wide switching) for all passes, so COLMAP can use one camera model.
- **Film at 30 fps, and on Samsung turn off "High efficiency video"** (Camera settings >
  Advanced video options), so it records H.264 instead of HEVC. 60 fps HEVC gives nothing
  extra for 3D but makes frame extraction several times slower (NewRoom: 7 min 51 s
  for 4 min of 60 fps HEVC, against about 1 min for myRoom2's 30 fps H.264).
- **Same orientation for every pass, landscape recommended.** Don't turn the phone
  between passes or during a pass: mixed orientations need a separate camera model per
  video, which weakens the reconstruction.
- **iPhone HDR is fine:** the pipeline detects HDR (HLG) and tone-maps it to normal colours
  (decision D-012). Turning HDR video off (Settings > Camera > Record Video > HDR Video)
  also works.

## Filming: three heights

| Video | Phone height | Aim | What it captures |
|---|---|---|---|
| `bottom` | about knee to hip | level or slightly up | floor, low furniture, under tables |
| `middle` | **chest / eye height** | level | walls, the main view (the viewer's eye height comes from this one) |
| `up` | arms raised above the head | slightly down | tops of furniture, worktops, the floor from above |

For each video:

- **Walk slowly along the walls, all the way round the room, facing into the room**
  (the camera looks at the opposite side). Don't stand in the middle and spin.
- **Turn slowly** at corners: a slow, steady turn, never a quick pan. Fast motion means
  blurry frames, and blurry frames are dropped or confuse COLMAP.
- **Overlap**: finish where you started, and keep doors, windows and large furniture in
  view for a while; each part of the room should be seen from several places.
- **About 1–1.5 minutes per height** for a normal room (myRoom2: 93 s, 82 s, 57 s).
  The pipeline needs at least 20 s in total and at most 5 min per video.
- Hold the phone **upright and the same way up** (landscape or portrait, but the same
  for all passes): the viewer uses the cameras' "up" direction to level the room.

## Avoid

- **Mirrors, large windows, glossy or glass surfaces**: they show a "different room"
  depending on where you stand and confuse pose estimation.
- **Bare, featureless walls filling the whole frame**: keep some furniture or edges in view.
- **Strong backlight** from a window straight ahead.
- **Moving objects** (see above), and filming through a doorway into another room.

## After filming

Run the pipeline; `poses` reports how many frames COLMAP placed (target ≥ 80%, the run
fails below 70%) and whether all videos ended up in one model
(`steps.poses.info` in `manifest.json`). If it fails, the usual causes are fast motion,
too little overlap, or mirrors/windows: re-film that pass more slowly.
