/**
 * On-screen joystick (bottom left) for touch screens.
 * `value` is the stick deflection: x = right, y = forward, length 0..1.
 */
export class Joystick {
  readonly value = { x: 0, y: 0 };
  readonly element: HTMLDivElement;
  private readonly knob: HTMLDivElement;
  private pointerId: number | null = null;
  private centre = { x: 0, y: 0 };
  private radius = 1;

  constructor(parent: HTMLElement = document.body) {
    this.element = document.createElement("div");
    this.element.className = "joystick";
    this.knob = document.createElement("div");
    this.knob.className = "joystick-knob";
    this.element.appendChild(this.knob);
    parent.appendChild(this.element);

    this.element.addEventListener("pointerdown", (e) => {
      if (this.pointerId !== null) return;
      e.preventDefault();
      e.stopPropagation(); // don't also start a look-drag
      this.pointerId = e.pointerId;
      this.element.setPointerCapture(e.pointerId);
      const rect = this.element.getBoundingClientRect();
      this.centre = { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
      this.radius = rect.width / 2;
      this.move(e);
    });
    this.element.addEventListener("pointermove", (e) => {
      if (e.pointerId === this.pointerId) this.move(e);
    });
    const end = (e: PointerEvent) => {
      if (e.pointerId !== this.pointerId) return;
      this.pointerId = null;
      this.value.x = this.value.y = 0;
      this.knob.style.transform = "";
    };
    this.element.addEventListener("pointerup", end);
    this.element.addEventListener("pointercancel", end);
  }

  setVisible(visible: boolean) {
    this.element.classList.toggle("hidden", !visible);
  }

  private move(e: PointerEvent) {
    let dx = (e.clientX - this.centre.x) / this.radius;
    let dy = (e.clientY - this.centre.y) / this.radius;
    const len = Math.hypot(dx, dy);
    if (len > 1) {
      dx /= len;
      dy /= len;
    }
    this.value.x = dx;
    this.value.y = -dy; // screen up = forward
    this.knob.style.transform = `translate(${dx * this.radius * 0.6}px, ${dy * this.radius * 0.6}px)`;
  }
}
