const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync(`${__dirname}/theme.js`, "utf8");

for (const saved of [null, "dark", "light", "invalid"]) {
  for (const blocked of [false, true]) {
    let onReady, onChange, stored;
    const select = { addEventListener: (_, callback) => { onChange = callback; } };
    const document = {
      documentElement: { dataset: {} },
      getElementById: () => select,
      addEventListener: (_, callback) => { onReady = callback; }
    };
    vm.runInNewContext(source, {
      document,
      localStorage: {
        getItem() { if (blocked) throw Error("blocked"); return saved; },
        setItem(key, value) { if (blocked) throw Error("blocked"); stored = [key, value]; }
      }
    });
    assert.equal(document.documentElement.dataset.theme, !blocked && saved === "dark" ? "dark" : "light");
    onReady();
    assert.equal(select.value, document.documentElement.dataset.theme);
    for (const value of ["dark", "light"]) {
      select.value = value;
      onChange();
      assert.equal(document.documentElement.dataset.theme, value);
      if (!blocked) assert.deepEqual(stored, ["transformer-theme", value]);
    }
  }
}
console.log("Theme checks passed: defaults, persistence, switching, unavailable storage.");
