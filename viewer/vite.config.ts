import { defineConfig, type Plugin } from "vite";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";

/**
 * Dev-only endpoint so the calibration panel can write public/config.json.
 * Only requests from this computer are accepted (not from phones on the LAN),
 * and only valid JSON is written. It does not exist in the built site.
 */
function saveConfigPlugin(): Plugin {
  return {
    name: "save-config",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/__save-config", (req, res) => {
        const ip = req.socket.remoteAddress ?? "";
        const local = ip === "::1" || ip === "127.0.0.1" || ip === "::ffff:127.0.0.1";
        if (req.method !== "POST" || !local) {
          res.statusCode = 403;
          res.end("Saving is only allowed from this computer");
          return;
        }
        let body = "";
        req.on("data", (chunk) => (body += chunk));
        req.on("end", () => {
          try {
            JSON.parse(body);
            writeFileSync(resolve(server.config.root, "public/config.json"), body);
            res.end("ok");
          } catch (err) {
            res.statusCode = 400;
            res.end(String(err));
          }
        });
      });
    },
  };
}

export default defineConfig({
  // Relative base so the built site works from any sub-folder of a static host.
  base: "./",
  plugins: [saveConfigPlugin()],
  server: {
    host: true, // listen on the LAN so a phone on the same Wi-Fi can connect
    port: 5173,
    // Don't reload the page when the panel writes config.json.
    watch: { ignored: ["**/public/config.json"] },
  },
  preview: {
    host: true,
    port: 4173,
  },
});
