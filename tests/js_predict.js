// Runs the CORE functions of web/index.html under Node for consistency testing.
// usage: node tests/js_predict.js <model.json> <cases.json>   -> prints JSON results
const fs = require("fs");
const path = require("path");
const html = fs.readFileSync(path.join(__dirname, "..", "web", "index.html"), "utf8");
const core = html.split("// CORE-START")[1].split("// CORE-END")[0];
const api = new Function(core + "\nreturn { makeNet, predictOne, predictFull };")();
const meta = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const model = { with_moisture: api.makeNet(meta.models.with_moisture), no_moisture: api.makeNet(meta.models.no_moisture) };
const cases = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const out = cases.map(c => {
  const r = api.predictFull(model, meta, c);
  return r.errors.length ? { errors: r.errors } : { basis: r.basis, warnings: r.warnings.length };
});
process.stdout.write(JSON.stringify(out));
