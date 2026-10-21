import type { LoadStage } from "./splat";

const overlay = document.getElementById("loading")!;
const bar = document.getElementById("progress-bar")!;
const detail = document.getElementById("loading-detail")!;
const track = bar.parentElement!;

const mb = (n: number) => (n / 1024 / 1024).toFixed(1);

export const loadingOverlay = {
  show(title = "Loading room…") {
    overlay.querySelector(".loading-title")!.textContent = title;
    overlay.classList.remove("hidden", "fade");
    bar.style.width = "0%";
    detail.classList.remove("error");
  },

  progress(stage: LoadStage, loaded: number, total: number) {
    if (stage === "download") {
      track.classList.toggle("indeterminate", total === 0);
      if (total > 0) {
        const pct = Math.round((loaded / total) * 100);
        bar.style.width = `${pct}%`;
        detail.textContent = `Downloading ${mb(loaded)} / ${mb(total)} MB (${pct}%)`;
      } else {
        detail.textContent = `Downloading ${mb(loaded)} MB`;
      }
    } else {
      track.classList.add("indeterminate");
      detail.textContent = "Decoding splats…";
    }
  },

  error(message: string) {
    track.classList.remove("indeterminate");
    detail.textContent = message;
    detail.classList.add("error");
  },

  hide() {
    overlay.classList.add("fade");
    setTimeout(() => overlay.classList.add("hidden"), 450);
  },
};
