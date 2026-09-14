"""The candidate page script (template/app.js) under node with a minimal
DOM: band names and dates come from FITS headers, so they must reach the
page as text, and each lightcurve point is drawn in the colour it carries."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parent.parent / "pyrt_transient" / "template" / "app.js"

RUNNER = r"""
const vm = require('vm');
const fs = require('fs');

class El {
  constructor(tag) { this.tag = tag; this.attrs = {}; this.children = []; this.text = ''; this.html = ''; this.offsetWidth = 600; }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  set textContent(t) { this.text = String(t); this.children = []; }
  get textContent() { return this.text + this.children.map(c => c.textContent).join(''); }
  set innerHTML(h) { this.html = h; }
  get innerHTML() { return this.html; }
  querySelector() { return this.inner || (this.inner = new El('g')); }
}
const nodes = {};
const node = (key) => nodes[key] || (nodes[key] = new El('div'));
const document = {
  getElementById: node, querySelector: node,
  createElement: (tag) => new El(tag), createElementNS: (ns, tag) => new El(tag),
  createTextNode: (t) => ({textContent: String(t)}),
  addEventListener() {},
};
const context = vm.createContext({document, console, setTimeout});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8') + '\n;globalThis.viewer = transientViewer;', context);
const viewer = context.viewer;

viewer.renderInteractiveLightcurve({lightcurve: {points: JSON.parse(process.argv[2])}});
viewer.selectedCandidate = {cutouts: [JSON.parse(process.argv[3])]};
viewer.currentCutoutIndex = 0;
viewer.updateCutoutViewer();

const dump = (n) => ({tag: n.tag, attrs: n.attrs, text: n.text === undefined ? n.textContent : n.text,
                      html: n.html, children: (n.children || []).map(dump)});
process.stdout.write(JSON.stringify({plot: dump(nodes['lightcurve-plot'].inner),
                                     time: dump(nodes['.cutout-time-info'])}));
"""

HOSTILE = '<img src=x onerror="alert(1)">'


def _run(points, cutout):
    out = subprocess.run(["node", "-e", RUNNER, str(APP_JS), json.dumps(points), json.dumps(cutout)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _all(node, tag):
    found = [node] if node.get("tag") == tag else []
    for child in node["children"]:
        found += _all(child, tag)
    return found


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def test_points_use_their_colour_and_band_names_stay_text():
    points = [
        {"time": 0.0, "magnitude": 18.0, "error": 0.1, "filter": "N→Sloan_r", "colour": "#ec7063"},
        {"time": 1.0, "magnitude": 18.2, "error": 0.1, "filter": "N→Sloan_r", "colour": "#ec7063"},
        {"time": 0.5, "magnitude": 17.0, "error": 0.1, "filter": HOSTILE, "colour": 'red" onload="x'},
    ]

    plot = _run(points, {"path": "a.webp", "filename": "a", "date": "2026-09-11", "filter": "Sloan_r"})["plot"]

    legend = [c for c in plot["children"] if c["tag"] == "g"]
    assert len(legend) == 1
    assert [t["text"] for t in _all(legend[0], "text")] == ["N→Sloan_r", HOSTILE]
    fills = [c["attrs"]["fill"] for c in plot["children"] if c["tag"] == "circle"]
    assert fills == ["#ec7063", "#ec7063", "#3498db"]      # a malformed colour is not used
    assert len([c for c in plot["children"] if c["tag"] == "path"]) == 1   # lines only within a band
    assert "<img" not in json.dumps(plot["html"]) and not _all(plot, "img")


def test_the_frame_info_shows_the_filter_as_text():
    cutout = {"path": "a.webp", "filename": "a", "date": "2026-09-11 01:12", "filter": HOSTILE}

    time_info = _run([{"time": 0.0, "magnitude": 18.0, "error": 0.1}], cutout)["time"]

    assert time_info["html"] == ""
    assert [c["text"] for c in time_info["children"]] == [
        "Image 1 of 1", "Date: 2026-09-11 01:12", f"Filter: {HOSTILE}"]
