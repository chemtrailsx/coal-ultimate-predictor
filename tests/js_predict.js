// Runs the CORE functions of web/index.html under Node for consistency testing.
// usage: node tests/js_predict.js <model.json> <cases.json>   -> prints JSON results
// A case with dir === "u2p" runs ultimate -> proximate, otherwise proximate -> ultimate.
const fs = require("fs");
const path = require("path");
const html = fs.readFileSync(path.join(__dirname, "..", "web", "index.html"), "utf8");
const core = html.split("// CORE-START")[1].split("// CORE-END")[0];
const api = new Function(core + "\nreturn { makeNet, predictOne, predictFull, predictProximate };")();
const meta = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const model = {
  with_moisture: api.makeNet(meta.models.with_moisture), no_moisture: api.makeNet(meta.models.no_moisture),
  rev_vm: api.makeNet(meta.reverse.vm), rev_moist: api.makeNet(meta.reverse.moist),
};
const cases = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const out = cases.map(c => {
  if (c.dir === "u2p") {
    const r = api.predictProximate(model, meta, c);
    return r.errors.length ? { errors: r.errors } : { basis: r.basis, daf: r.daf, src: r.src };
  }
  const r = api.predictFull(model, meta, c);
  return r.errors.length ? { errors: r.errors } : { basis: r.basis, warnings: r.warnings.length };
});
process.stdout.write(JSON.stringify(out));
