export type Mode = "walk" | "tour";

interface TourBarCallbacks {
  onMode: (mode: Mode) => void;
  onPrev: () => void;
  onNext: () => void;
}

/**
 * Bottom-right bar: Walk / Tour switch, and ◀ name ▶ to step through the hotspots.
 * Walk: free walking, hotspots can be tapped. Tour: no walking, only hotspots
 * (like a hotel / campus 360 tour), drag to look around.
 */
export function createTourBar({ onMode, onPrev, onNext }: TourBarCallbacks) {
  const bar = document.createElement("div");
  bar.className = "tourbar";
  bar.innerHTML = `
    <div class="tour-nav">
      <button class="tour-prev" aria-label="Previous viewpoint">◀</button>
      <span class="tour-label"></span>
      <button class="tour-next" aria-label="Next viewpoint">▶</button>
    </div>
    <div class="tour-mode" role="group" aria-label="Mode">
      <button data-mode="walk">Walk</button>
      <button data-mode="tour">Tour</button>
    </div>`;
  document.body.appendChild(bar);

  const label = bar.querySelector<HTMLSpanElement>(".tour-label")!;
  const nav = bar.querySelector<HTMLDivElement>(".tour-nav")!;
  const modeButtons = [...bar.querySelectorAll<HTMLButtonElement>("[data-mode]")];

  // Keep taps on the bar from also starting a look-drag on the canvas.
  bar.addEventListener("pointerdown", (e) => e.stopPropagation());
  bar.querySelector(".tour-prev")!.addEventListener("click", onPrev);
  bar.querySelector(".tour-next")!.addEventListener("click", onNext);
  for (const b of modeButtons) b.addEventListener("click", () => onMode(b.dataset.mode as Mode));

  return {
    setMode(mode: Mode) {
      for (const b of modeButtons) b.classList.toggle("active", b.dataset.mode === mode);
    },
    /** "2 / 5 · Kitchen", or empty when not at a hotspot. */
    setLabel(text: string) {
      label.textContent = text || "Viewpoints";
    },
    setHasHotspots(has: boolean) {
      nav.classList.toggle("hidden", !has);
    },
  };
}
