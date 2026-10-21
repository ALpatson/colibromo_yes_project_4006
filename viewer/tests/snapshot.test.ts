import { afterEach, describe, expect, it, vi } from "vitest";
import { snapshotParams, sparkOptions } from "../src/snapshot";

function withUrl(search: string) {
  vi.stubGlobal("window", { location: { search } });
}

afterEach(() => vi.unstubAllGlobals());

describe("snapshot mode", () => {
  it("is off without ?snap", () => {
    withUrl("?run=runs/myRoom");
    expect(snapshotParams()).toBeNull();
    expect(sparkOptions(null)).toEqual({});
  });

  it("reads the frame and the renderer settings to compare", () => {
    withUrl("?run=runs/x&snap=bottom_00000.jpg&blur=classic&sh=0&lod=0");
    const p = snapshotParams()!;
    expect(p).toEqual({ frame: "bottom_00000.jpg", blur: "classic", sh: 0, lod: false });
    // Classic 3DGS / gsplat "default" training: blur without opacity correction.
    expect(sparkOptions(p)).toEqual({ blurAmount: 0, preBlurAmount: 0.3, enableLod: false });
  });

  it("keeps Spark's defaults unless asked", () => {
    withUrl("?snap=up_00001.jpg&blur=nonsense");
    const p = snapshotParams()!;
    expect(p.blur).toBe("default");
    expect(p.sh).toBeNull();
    expect(sparkOptions(p)).toEqual({});
    expect(sparkOptions({ ...p, blur: "aa" })).toEqual({ blurAmount: 0.3, preBlurAmount: 0 });
  });
});
