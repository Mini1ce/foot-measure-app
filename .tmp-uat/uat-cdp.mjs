import fs from "node:fs/promises";

const endpoint = process.env.CDP_ENDPOINT || "http://127.0.0.1:9223";
const [action = "inspect", ...args] = process.argv.slice(2);

const targets = await fetch(`${endpoint}/json/list`).then((response) => response.json());
const target = targets.find((item) => item.type === "page" && item.url.includes("chinda3-uat.iapp.co.th"));
if (!target?.webSocketDebuggerUrl) {
  throw new Error("No Chinda UAT page target found");
}

const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});

let nextId = 1;
const pending = new Map();
socket.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (!message.id) return;
  const waiter = pending.get(message.id);
  if (!waiter) return;
  pending.delete(message.id);
  if (message.error) waiter.reject(new Error(JSON.stringify(message.error)));
  else waiter.resolve(message.result);
});

function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

async function evaluate(expression) {
  const result = await send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
    userGesture: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text || "Evaluation failed");
  }
  return result.result?.value;
}

async function inspect() {
  return evaluate(`(() => {
    const visible = (el) => {
      const rect = el.getBoundingClientRect();
      const style = getComputedStyle(el);
      return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
    };
    const describe = (el) => {
      const rect = el.getBoundingClientRect();
      return {
        tag: el.tagName.toLowerCase(),
        type: el.getAttribute("type"),
        name: el.getAttribute("name"),
        id: el.id || null,
        role: el.getAttribute("role"),
        placeholder: el.getAttribute("placeholder"),
        ariaLabel: el.getAttribute("aria-label"),
        text: (el.innerText || el.textContent || "").trim().replace(/\\s+/g, " ").slice(0, 300),
        value: el.type === "password" ? "<password>" : (el.value ?? null),
        disabled: !!el.disabled,
        required: !!el.required,
        href: el.href || null,
        box: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
      };
    };
    return {
      url: location.href,
      title: document.title,
      viewport: { width: innerWidth, height: innerHeight, dpr: devicePixelRatio },
      bodyText: document.body?.innerText?.replace(/\\n{3,}/g, "\\n\\n").slice(0, 12000) || "",
      elements: [...document.querySelectorAll("input, textarea, select, button, a, [role=button]")].filter(visible).map(describe),
      active: document.activeElement ? describe(document.activeElement) : null,
    };
  })()`);
}

async function findBox(selector) {
  const box = await evaluate(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
  })()`);
  if (!box || box.width <= 0 || box.height <= 0) throw new Error(`Element not visible: ${selector}`);
  return box;
}

async function click(selector) {
  const box = await findBox(selector);
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;
  await send("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
  await send("Input.dispatchMouseEvent", { type: "mousePressed", x, y, button: "left", clickCount: 1 });
  await send("Input.dispatchMouseEvent", { type: "mouseReleased", x, y, button: "left", clickCount: 1 });
}

async function fill(selector, value) {
  await click(selector);
  await send("Input.dispatchKeyEvent", { type: "keyDown", key: "a", code: "KeyA", modifiers: 2 });
  await send("Input.dispatchKeyEvent", { type: "keyUp", key: "a", code: "KeyA", modifiers: 2 });
  await send("Input.dispatchKeyEvent", { type: "keyDown", key: "Backspace", code: "Backspace" });
  await send("Input.dispatchKeyEvent", { type: "keyUp", key: "Backspace", code: "Backspace" });
  await send("Input.insertText", { text: value });
}

async function press(key) {
  const code = key === "Enter" ? "Enter" : key === "Tab" ? "Tab" : key;
  await send("Input.dispatchKeyEvent", { type: "keyDown", key, code });
  await send("Input.dispatchKeyEvent", { type: "keyUp", key, code });
}

async function screenshot(outputPath) {
  const result = await send("Page.captureScreenshot", { format: "png", fromSurface: true, captureBeyondViewport: false });
  await fs.writeFile(outputPath, Buffer.from(result.data, "base64"));
  return outputPath;
}

await send("Page.enable");
await send("Runtime.enable");

let output;
if (action === "inspect") {
  output = await inspect();
} else if (action === "click") {
  await click(args[0]);
  await new Promise((resolve) => setTimeout(resolve, 750));
  output = await inspect();
} else if (action === "fill") {
  await fill(args[0], args[1] ?? "");
  await new Promise((resolve) => setTimeout(resolve, 250));
  output = await inspect();
} else if (action === "press") {
  await press(args[0]);
  await new Promise((resolve) => setTimeout(resolve, 750));
  output = await inspect();
} else if (action === "screenshot") {
  output = { screenshot: await screenshot(args[0]) };
} else {
  throw new Error(`Unknown action: ${action}`);
}

console.log(JSON.stringify(output, null, 2));
socket.close();
