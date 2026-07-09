'use strict';

// ── JSON helpers ─────────────────────────────────────────────────────────────
function deepSanitize(v) {
  if (Array.isArray(v)) return v.map(deepSanitize);
  if (v && typeof v === 'object') {
    const o = {};
    for (const [k, val] of Object.entries(v)) o[k] = deepSanitize(val);
    return o;
  }
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  if (typeof v === 'string') {
    const s = v.trim();
    if (s === 'NaN' || s === 'nan' || s === 'Infinity' || s === '-Infinity') return null;
  }
  return v;
}

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const text = await r.text();
  try {
    return deepSanitize(JSON.parse(text));
  } catch {
    return deepSanitize(JSON.parse(
      text.replace(/\bNaN\b/g, 'null').replace(/\b-?Infinity\b/g, 'null')
    ));
  }
}

// ── Formatting helpers ────────────────────────────────────────────────────────
const fmt = (v, dec = 3) => (v == null ? 'N/A' : Number(v).toFixed(dec));
const fmtMag = v => (v == null ? 'N/A' : Number(v).toFixed(2));

function galaxyBadge(flag) {
  if (!flag || flag === '') return '';
  const labels = { in_galaxy: '🌌 in galaxy', nuclear: '⚠ nuclear', isolated: '· isolated' };
  return `<span class="gal-badge gal-${flag}">${labels[flag] || flag}</span>`;
}

function typeBadge(t) {
  const map = {
    new:         { bg: '#3498db', icon: '✨', label: 'New' },
    brightening: { bg: '#e74c3c', icon: '↑',  label: 'Brightening' },
    fading:      { bg: '#f39c12', icon: '↓',  label: 'Fading' },
    trail:       { bg: '#8e44ad', icon: '—',  label: 'Trail' },
    unknown:     { bg: '#95a5a6', icon: '?',  label: 'Unknown' },
  };
  const d = map[t] || map.unknown;
  return `<span class="type-badge" style="background:${d.bg}">${d.icon} ${d.label}</span>`;
}

function snScoreColor(score) {
  if (score == null) return '#999';
  if (score >= 2.0) return '#1a7a3c';
  if (score >= 1.0) return '#0f3460';
  return '#888';
}

function lcDeltaBadge(delta) {
  if (delta == null || !Number.isFinite(delta)) return '';
  const d = Number(delta);
  // positive delta = brighter now vs historical = brightening event
  if (d >= 1.0) return `<span style="background:#c0392b;color:#fff;font-size:0.68em;padding:1px 6px;border-radius:8px;font-weight:700;margin-left:3px;">↑${d.toFixed(1)} mag</span>`;
  if (d >= 0.3) return `<span style="background:#e67e22;color:#fff;font-size:0.68em;padding:1px 6px;border-radius:8px;font-weight:700;margin-left:3px;">↑${d.toFixed(1)} mag</span>`;
  if (d <= -0.5) return `<span style="background:#7f8c8d;color:#fff;font-size:0.68em;padding:1px 6px;border-radius:8px;margin-left:3px;">↓${Math.abs(d).toFixed(1)}</span>`;
  return `<span style="color:#888;font-size:0.68em;margin-left:3px;">Δ${d > 0 ? '+' : ''}${d.toFixed(2)}</span>`;
}

// ── Lightcurve helpers ────────────────────────────────────────────────────────
let lcChartInstance = null;

const LC_COLORS = {
  // PS1 bands
  'g_PS1': '#2ecc71', 'r_PS1': '#e74c3c', 'i_PS1': '#e67e22',
  'z_PS1': '#9b59b6', 'y_PS1': '#1abc9c',
  // ATLAS bands
  'o_ATLAS': '#f39c12', 'c_ATLAS': '#2980b9', 'i_ATLAS': '#8e44ad',
};

function bandKey(point) {
  return `${point.band}_${point.survey}`;
}

function buildLcDatasets(points) {
  // Group by band+survey
  const groups = {};
  for (const p of points) {
    const key = bandKey(p);
    if (!groups[key]) groups[key] = { detections: [], uplims: [] };
    if (p.ulim) groups[key].uplims.push(p);
    else groups[key].detections.push(p);
  }

  const datasets = [];
  for (const [key, g] of Object.entries(groups)) {
    const [band, survey] = key.split('_');
    const color = LC_COLORS[key] || '#888';
    const label = `${survey} ${band}`;

    if (g.detections.length) {
      datasets.push({
        label,
        data: g.detections.map(p => ({ x: p.mjd, y: p.mag,
          err: p.mag_err, ulim: false })),
        borderColor: color, backgroundColor: color,
        pointRadius: 5, pointStyle: 'circle',
        showLine: true, tension: 0.1,
        _color: color,
      });
    }
    if (g.uplims.length) {
      datasets.push({
        label: `${label} (lim)`,
        data: g.uplims.map(p => ({ x: p.mjd, y: p.mag,
          err: 0, ulim: true })),
        borderColor: color, backgroundColor: 'transparent',
        borderDash: [4, 3],
        pointRadius: 5, pointStyle: 'triangle',
        showLine: false,
        _color: color,
      });
    }
  }
  return datasets;
}

// Custom Chart.js plugin to draw error bars and downward triangles for upper limits
const errorBarPlugin = {
  id: 'errorBar',
  afterDatasetsDraw(chart) {
    const ctx = chart.ctx;
    chart.data.datasets.forEach((ds, di) => {
      const meta = chart.getDatasetMeta(di);
      if (meta.hidden) return;
      ds.data.forEach((pt, pi) => {
        const elem = meta.data[pi];
        if (!elem) return;
        const { x, y } = elem.getProps(['x', 'y'], true);
        ctx.save();
        ctx.strokeStyle = ds._color || ds.borderColor;
        ctx.lineWidth = 1.5;
        if (pt.ulim) {
          // downward arrow for upper limit
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x, y + 12);
          ctx.moveTo(x - 4, y + 8);
          ctx.lineTo(x, y + 12);
          ctx.lineTo(x + 4, y + 8);
          ctx.stroke();
        } else if (pt.err && pt.err > 0) {
          const yScale = chart.scales.y;
          const yTop = yScale.getPixelForValue(pt.y - pt.err);
          const yBot = yScale.getPixelForValue(pt.y + pt.err);
          ctx.beginPath();
          ctx.moveTo(x, yTop); ctx.lineTo(x, yBot);
          ctx.moveTo(x - 3, yTop); ctx.lineTo(x + 3, yTop);
          ctx.moveTo(x - 3, yBot); ctx.lineTo(x + 3, yBot);
          ctx.stroke();
        }
        ctx.restore();
      });
    });
  }
};

// Register after DOM ready (Chart.js loads synchronously from CDN)
document.addEventListener('DOMContentLoaded', () => {
  if (typeof Chart !== 'undefined') Chart.register(errorBarPlugin);
});

// ── Aladin Lite state ─────────────────────────────────────────────────────────
let aladinInstance = null;
let aladinReady = false;

function initAladin(ra, dec) {
  const div = document.getElementById('aladin-lite-div');
  if (!div) return;

  const target = `${Number(ra).toFixed(5)} ${Number(dec).toFixed(5)}`;

  if (aladinInstance && aladinReady) {
    // Just move the view — do not reinitialise the whole widget
    aladinInstance.gotoRaDec(Number(ra), Number(dec));
    return;
  }

  if (typeof A === 'undefined') {
    div.innerHTML = '<div style="padding:20px;color:#888;text-align:center;">Aladin Lite unavailable (no internet connection?)</div>';
    return;
  }

  try {
    aladinInstance = A.aladin('#aladin-lite-div', {
      survey: 'P/PanSTARRS/DR1/color-z-zg-g',
      fov: 0.08,          // ~5 arcmin
      target: target,
      cooFrame: 'ICRSd',
      showReticle: true,
      showZoomControl: true,
      showFullscreenControl: false,
      showLayersControl: true,
    });
    aladinReady = true;
  } catch (e) {
    div.innerHTML = `<div style="padding:20px;color:#c00;text-align:center;">Sky view error: ${e.message}</div>`;
  }
}

// ── Main viewer object ────────────────────────────────────────────────────────
const sv = {
  all: [],          // all candidates from candidates.json
  filtered: [],     // after filters
  selected: null,
  cutoutIdx: 0,

  // Active filters
  typeFilter: 'all',
  galFilter: 'all',

  // ── Init ────────────────────────────────────────────────────────────────
  async init() {
    try {
      this.all = await fetchJson('candidates.json');
      // Sort by sn_score descending (fall back to quality_score)
      this.all.sort((a, b) =>
        (b.sn_score ?? b.quality_score ?? 0) - (a.sn_score ?? a.quality_score ?? 0)
      );
      this.filtered = [...this.all];
      this.buildFilters();
      this.renderList();
    } catch (e) {
      const el = document.getElementById('candidate-list-content')
              || document.getElementById('candidate-list');
      if (el) el.innerHTML =
        `<div style="padding:20px;color:#c00;">Failed to load candidates: ${e.message}</div>`;
    }
  },

  // ── Filters ─────────────────────────────────────────────────────────────
  buildFilters() {
    // Type filter buttons
    const typeCounts = {};
    this.all.forEach(c => {
      const t = c.candidate_type || 'unknown';
      typeCounts[t] = (typeCounts[t] || 0) + 1;
    });

    const typeOrder = ['all', 'new', 'brightening', 'fading', 'trail', 'unknown'];
    const typeLabels = { all: 'All', new: '✨ New', brightening: '↑ Bright',
                         fading: '↓ Fading', trail: '— Trail', unknown: '? Other' };

    document.getElementById('type-filters').innerHTML = typeOrder
      .filter(t => t === 'all' || typeCounts[t])
      .map(t => {
        const count = t === 'all' ? this.all.length : typeCounts[t];
        return `<button class="filter-btn ${t === this.typeFilter ? 'active' : ''}"
                  onclick="sv.setTypeFilter('${t}')">${typeLabels[t]} (${count})</button>`;
      }).join('');

    // Galaxy filter buttons
    const galCounts = {};
    this.all.forEach(c => {
      const g = c.galaxy_flag || 'unknown';
      galCounts[g] = (galCounts[g] || 0) + 1;
    });

    const galOrder = ['all', 'in_galaxy', 'nuclear', 'isolated'];
    const galLabels = { all: 'All', in_galaxy: '🌌 In galaxy',
                        nuclear: '⚠ Nuclear', isolated: '· Isolated' };

    document.getElementById('gal-filters').innerHTML = galOrder
      .filter(g => g === 'all' || galCounts[g])
      .map(g => {
        const count = g === 'all' ? this.all.length : (galCounts[g] || 0);
        return `<button class="filter-btn ${g === this.galFilter ? 'active' : ''}"
                  onclick="sv.setGalFilter('${g}')">${galLabels[g]} (${count})</button>`;
      }).join('');
  },

  setTypeFilter(t) {
    this.typeFilter = t;
    this.applyFilters();
    // Rebuild filter UI to update active states
    this.buildFilters();
  },

  setGalFilter(g) {
    this.galFilter = g;
    this.applyFilters();
    this.buildFilters();
  },

  applyFilters() {
    const tnsOnly = document.getElementById('tns-only')?.checked;
    this.filtered = this.all.filter(c => {
      const typeOk = this.typeFilter === 'all' || c.candidate_type === this.typeFilter;
      const galOk  = this.galFilter  === 'all' || c.galaxy_flag    === this.galFilter;
      const tnsOk  = !tnsOnly || (c.tns_name && c.tns_name !== '');
      return typeOk && galOk && tnsOk;
    });
    this.renderList();
  },

  // ── Candidate list ───────────────────────────────────────────────────────
  renderList() {
    // Support both our new div id and the legacy id used by the old template
    const el = document.getElementById('candidate-list-content')
            || document.getElementById('candidate-list');
    if (!el) return;
    if (!this.filtered.length) {
      el.innerHTML = '<div style="padding:20px;text-align:center;color:#aaa;">No candidates match filters</div>';
      return;
    }

    el.innerHTML = this.filtered.map(c => {
      const score = c.sn_score ?? c.quality_score ?? 0;
      const isSelected = this.selected?.id === c.id;
      const tns = c.tns_name && c.tns_name !== '';
      const galFlag = c.galaxy_flag || '';
      const gSep = c.galaxy_sep_arcsec != null ? `${Number(c.galaxy_sep_arcsec).toFixed(1)}"` : '';

      return `
        <div class="candidate-item ${isSelected ? 'selected' : ''}"
             onclick="sv.select('${c.id}')">
          <div class="cand-top">
            <span class="cand-id" title="${c.id}">
              ${c.id}
              ${tns ? '<span class="tns-dot" title="Matched on TNS"></span>' : ''}
            </span>
            <span class="sn-score-pill" style="background:${snScoreColor(score)}">
              SN ${Number(score).toFixed(2)}
            </span>
          </div>
          <div class="cand-row2">
            <span>${typeBadge(c.candidate_type || 'unknown')} ${galaxyBadge(galFlag)}</span>
            <span>${gSep ? '🌌 ' + gSep : ''}</span>
          </div>
          <div class="cand-row2" style="margin-top:2px;">
            <span>Mag ${fmtMag(c.MAG_CALIB ?? c.mag_weighted_mean)}${lcDeltaBadge(c.lc_mag_delta)}</span>
            <span>Q ${fmt(c.quality_score, 2)}</span>
            <span>${c.reference_catalog || ''}</span>
          </div>
        </div>`;
    }).join('');
  },

  // ── Select & detail ──────────────────────────────────────────────────────
  select(id) {
    this.selected = this.all.find(c => c.id === id) || null;
    this.cutoutIdx = 0;
    this.renderList();     // refresh selection highlight
    this.renderDetail();
  },

  renderDetail() {
    const el = document.getElementById('candidate-detail');
    const c = this.selected;
    if (!c) {
      el.innerHTML = '<div class="placeholder">Select a candidate from the list to view details</div>';
      return;
    }

    // Replacing innerHTML destroys the aladin div — reset so it re-initialises into the new one
    aladinInstance = null;
    aladinReady = false;

    const hasCutouts = c.cutouts && c.cutouts.length > 0;
    const tns = c.tns_name && c.tns_name !== '';
    const score = c.sn_score ?? c.quality_score ?? 0;

    el.innerHTML = `
      <h2 style="margin:0 0 16px; color:#1a1a2e; display:flex; align-items:center; gap:10px;">
        ${c.id}
        ${typeBadge(c.candidate_type || 'unknown')}
        ${galaxyBadge(c.galaxy_flag)}
        ${tns ? `<span style="background:#e74c3c;color:#fff;font-size:0.6em;padding:3px 8px;border-radius:10px;font-weight:700;">TNS ${c.tns_name}</span>` : ''}
        <span class="sn-score-pill" style="font-size:0.7em; background:${snScoreColor(score)}">SN score ${fmt(score, 2)}</span>
      </h2>

      ${this.renderCutouts(c, hasCutouts)}
      ${this.renderForcedLightcurve(c)}
      ${this.renderHostGalaxy(c)}
      ${this.renderTNS(c)}
      ${this.renderProperties(c)}
      ${this.renderSkyView()}
    `;

    if (hasCutouts) this.updateCutout();
    if (c.forced_lc) this.drawLcChart(c);

    // Init/update Aladin after DOM settle
    setTimeout(() => {
      if (c.ALPHA_J2000 != null && c.DELTA_J2000 != null) {
        initAladin(c.ALPHA_J2000, c.DELTA_J2000);
      }
    }, 80);
  },

  // ── Cutout viewer section ────────────────────────────────────────────────
  renderCutouts(c, hasCutouts) {
    if (!hasCutouts) return `
      <div class="card">
        <h3>Cutout Image</h3>
        <div style="text-align:center;color:#aaa;padding:20px;">No cutout images available</div>
      </div>`;

    const n = c.cutouts.length;
    return `
      <div class="card">
        <h3>Cutout Viewer</h3>
        <div class="cutout-nav">
          <button id="btn-prev" onclick="sv.prevCutout()" ${n <= 1 ? 'disabled' : ''}>&lt; Prev</button>
          <div class="cutout-time-info" id="cutout-info">
            Image 1 of ${n}
            <div class="date-display">${c.cutouts[0].date || ''}</div>
          </div>
          <button id="btn-next" onclick="sv.nextCutout()" ${n <= 1 ? 'disabled' : ''}>Next &gt;</button>
        </div>
        <div class="cutout-image-container" id="cutout-img-wrap">
          <img src="${c.cutouts[0].path}" class="cutout-image" id="cutout-img" alt="${c.cutouts[0].filename}">
          <div class="cutout-filename" id="cutout-fname">${c.cutouts[0].filename}</div>
        </div>
        ${n > 1 ? `<input type="range" class="cutout-slider" min="0" max="${n-1}" value="0"
                          oninput="sv.cutoutIdx=+this.value; sv.updateCutout()" id="cutout-slider">` : ''}
        ${n > 1 ? `<div class="cutouts-strip">${c.cutouts.map((cu, i) =>
          `<img src="${cu.path}" title="${cu.filename}" onclick="sv.cutoutIdx=${i};sv.updateCutout()">`
        ).join('')}</div>` : ''}
      </div>`;
  },

  prevCutout() {
    const n = this.selected?.cutouts?.length || 1;
    this.cutoutIdx = (this.cutoutIdx - 1 + n) % n;
    this.updateCutout();
  },

  nextCutout() {
    const n = this.selected?.cutouts?.length || 1;
    this.cutoutIdx = (this.cutoutIdx + 1) % n;
    this.updateCutout();
  },

  updateCutout() {
    const c = this.selected;
    if (!c?.cutouts?.length) return;
    const cu = c.cutouts[this.cutoutIdx];
    const n = c.cutouts.length;
    const img  = document.getElementById('cutout-img');
    const info = document.getElementById('cutout-info');
    const fname = document.getElementById('cutout-fname');
    const slider = document.getElementById('cutout-slider');
    if (img)   { img.src = cu.path; img.alt = cu.filename; }
    if (fname) fname.textContent = cu.filename;
    if (info)  info.innerHTML = `Image ${this.cutoutIdx+1} of ${n}<div class="date-display">${cu.date || ''}</div>`;
    if (slider) slider.value = this.cutoutIdx;
  },

  // ── Host galaxy section ──────────────────────────────────────────────────
  renderHostGalaxy(c) {
    const flag = c.galaxy_flag || 'unknown';
    const sep  = c.galaxy_sep_arcsec;
    const name = c.galaxy_name || '';

    let flagHtml = galaxyBadge(flag);
    let advisory = '';
    if (flag === 'nuclear') {
      advisory = '<p style="color:#856404;background:#fff3cd;padding:8px;border-radius:4px;margin:10px 0 0;font-size:0.85em;">⚠ Candidate is within 2″ of a galaxy nucleus — may be an AGN or TDE rather than a SN.</p>';
    } else if (flag === 'isolated') {
      advisory = '<p style="color:#555;font-size:0.85em;margin:10px 0 0;">No nearby catalogued galaxy found. Could be a hostless SN, an SLSN, or a misidentified artefact.</p>';
    }

    return `
      <div class="card">
        <h3>Host Galaxy (HyperLEDA)</h3>
        <dl class="galaxy-info-grid">
          <div><dt>Name</dt><dd>${name || '<em>unknown</em>'}</dd></div>
          <div><dt>Separation</dt><dd>${sep != null ? Number(sep).toFixed(1) + ' arcsec' : 'N/A'}</dd></div>
          <div><dt>Flag</dt><dd>${flagHtml}</dd></div>
          <div><dt>Candidate RA/Dec</dt>
               <dd>${fmt(c.ALPHA_J2000, 5)}° / ${fmt(c.DELTA_J2000, 5)}°</dd></div>
        </dl>
        ${advisory}
        <div id="aladin-lite-div">
          <div style="padding:20px;text-align:center;color:#aaa;">Loading sky view…</div>
        </div>
      </div>`;
  },

  // ── TNS section ──────────────────────────────────────────────────────────
  renderTNS(c) {
    const matched = c.tns_name && c.tns_name !== '';
    let inner;
    if (matched) {
      const z = c.tns_z != null ? `z = ${Number(c.tns_z).toFixed(4)}` : '';
      inner = `
        <div class="tns-matched">
          <div class="tns-name">${c.tns_name}</div>
          <div style="margin-top:6px; font-size:0.9em;">
            ${c.tns_type ? `<strong>Type:</strong> ${c.tns_type} &nbsp;` : ''}
            ${z}
          </div>
          <div style="margin-top:6px;font-size:0.8em;color:#888;">
            Already reported on IAU Transient Name Server — do not re-report.
          </div>
        </div>`;
    } else {
      inner = '<div class="tns-not-matched">No match on TNS — possible new transient.</div>';
    }
    return `<div class="card"><h3>TNS Cross-match</h3>${inner}</div>`;
  },

  // ── Properties table ─────────────────────────────────────────────────────
  renderProperties(c) {
    const ellOk = c.ELLIPTICITY != null && c.ELLIPTICITY < 0.4;
    const fwhmOk = c.fwhm_ratio != null && c.fwhm_ratio >= 0.5 && c.fwhm_ratio <= 2.0;

    const flagClass = (ok) => ok ? 'val-ok' : 'val-warn';

    return `
      <div class="card">
        <h3>Detection Properties</h3>
        <table class="props">
          <tr>
            <th>Magnitude</th>
            <td>${fmtMag(c.MAG_CALIB ?? c.mag_weighted_mean)} ± ${fmt(c.MAGERR_CALIB, 3)}</td>
            <th>Mag difference</th>
            <td>${fmt(c.magnitude_difference, 3)}</td>
          </tr>
          <tr>
            <th>SN score</th>
            <td><strong style="color:${snScoreColor(c.sn_score)}">${fmt(c.sn_score, 3)}</strong></td>
            <th>Quality score</th>
            <td>${fmt(c.quality_score, 3)}</td>
          </tr>
          <tr>
            <th>Ellipticity</th>
            <td class="${flagClass(ellOk)}">${fmt(c.ELLIPTICITY, 3)}
              ${ellOk ? '✓' : '⚠ elongated'}</td>
            <th>FWHM ratio</th>
            <td class="${flagClass(fwhmOk)}">${fmt(c.fwhm_ratio, 3)}
              ${fwhmOk ? '✓' : '⚠ PSF mismatch'}</td>
          </tr>
          <tr>
            <th>RA</th>
            <td>${fmt(c.ALPHA_J2000, 6)}°</td>
            <th>Dec</th>
            <td>${fmt(c.DELTA_J2000, 6)}°</td>
          </tr>
          <tr>
            <th>Reference cat.</th>
            <td>${c.reference_catalog || 'N/A'}</td>
            <th>Candidate type</th>
            <td>${typeBadge(c.candidate_type || 'unknown')}</td>
          </tr>
          <tr>
            <th>FLAGS</th>
            <td>${c.FLAGS != null ? (c.FLAGS == 0 ? '<span class="val-ok">0 ✓</span>' : `<span class="val-warn">${c.FLAGS} ⚠</span>`) : 'N/A'}</td>
            <th>Nearest source</th>
            <td>${c.nearest_source_dist != null ? fmt(c.nearest_source_dist, 1) + '"' : 'N/A'}</td>
          </tr>
          <tr>
            <th>LC Δ brightening</th>
            <td colspan="3">${(() => {
              const d = c.lc_mag_delta;
              if (d == null || !Number.isFinite(d)) return '<span style="color:#aaa">No forced photometry</span>';
              const flc = c.forced_lc;
              const src = flc ? (flc.atlas?.some(p=>!p.ulim) ? 'ATLAS' : flc.atlas?.length ? 'ATLAS (lim)' : 'PS1') : '—';
              const color = d >= 1.0 ? '#c0392b' : d >= 0.3 ? '#e67e22' : d <= -0.5 ? '#7f8c8d' : '#555';
              const arrow = d > 0 ? '↑' : '↓';
              return `<strong style="color:${color}">${arrow} ${Math.abs(d).toFixed(2)} mag</strong>
                      <span style="color:#888;font-size:0.85em;margin-left:8px;">
                        (now ${fmtMag(c.MAG_CALIB ?? c.mag_weighted_mean)} vs historical ${
                          flc ? (() => {
                            const pts = [...(flc.atlas||[]),...(flc.ps1||[])].filter(p=>!p.ulim);
                            return pts.length ? fmtMag(pts.reduce((s,p)=>s+p.mag,0)/pts.length) : '—';
                          })() : '—'
                        } median, source: ${src})
                      </span>`;
            })()}</td>
          </tr>
          <tr>
            <th>Saturated</th>
            <td>${c.saturated === 'True' ? '<span class="val-bad">Yes</span>' : 'No'}</td>
            <th>Blended</th>
            <td>${c.blended === 'True' ? '<span class="val-warn">Yes</span>' : 'No'}</td>
          </tr>
        </table>
      </div>`;
  },

  // ── Forced photometry lightcurve ─────────────────────────────────────────
  renderForcedLightcurve(c) {
    const flc = c.forced_lc;
    if (!flc) return '';

    const ps1pts  = flc.ps1  || [];
    const atlaspts = flc.atlas || [];
    const allpts = [...ps1pts, ...atlaspts];

    if (!allpts.length) return `
      <div class="card">
        <h3>Historical Lightcurve (PS1 + ATLAS)</h3>
        <div class="lc-no-data">No archival photometry found at this position.</div>
      </div>`;

    const surveys = [];
    if (ps1pts.length)   surveys.push(`PS1: ${ps1pts.filter(p=>!p.ulim).length} det.`);
    if (atlaspts.length) surveys.push(`ATLAS: ${atlaspts.filter(p=>!p.ulim).length} det. + ${atlaspts.filter(p=>p.ulim).length} lim.`);

    return `
      <div class="card">
        <h3>Historical Lightcurve (PS1 + ATLAS)</h3>
        <div style="font-size:0.78em;color:#888;margin-bottom:8px;">${surveys.join(' &nbsp;|&nbsp; ')}</div>
        <div class="lc-canvas-wrap">
          <canvas id="lc-chart"></canvas>
        </div>
        <div class="lc-legend" id="lc-legend"></div>
      </div>`;
  },

  drawLcChart(c) {
    const flc = c.forced_lc;
    if (!flc) return;
    const canvas = document.getElementById('lc-chart');
    if (!canvas || typeof Chart === 'undefined') return;

    if (lcChartInstance) { lcChartInstance.destroy(); lcChartInstance = null; }

    const ps1pts   = flc.ps1   || [];
    const atlaspts = flc.atlas || [];
    const allpts   = [...ps1pts, ...atlaspts];
    if (!allpts.length) return;

    const datasets = buildLcDatasets(allpts);

    // y-axis: magnitude (inverted — lower mag = brighter = up)
    const mags = allpts.filter(p => !p.ulim).map(p => p.mag);
    const yMin = mags.length ? Math.min(...mags) - 0.5 : 14;
    const yMax = mags.length ? Math.max(...mags) + 1.0 : 22;

    lcChartInstance = new Chart(canvas, {
      type: 'scatter',
      data: { datasets },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: 'MJD', font: { size: 11 } },
            type: 'linear',
          },
          y: {
            title: { display: true, text: 'Magnitude (AB)', font: { size: 11 } },
            min: yMin, max: yMax,
            reverse: true,
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => {
                const pt = ctx.raw;
                const err = pt.err ? ` ± ${pt.err.toFixed(3)}` : '';
                const lim = pt.ulim ? ' (upper lim)' : '';
                return `MJD ${pt.x.toFixed(2)}: ${pt.y.toFixed(3)}${err}${lim}`;
              }
            }
          }
        },
      },
      plugins: [errorBarPlugin],
    });

    // Build custom legend
    const legendEl = document.getElementById('lc-legend');
    if (legendEl) {
      legendEl.innerHTML = datasets.map(ds => `
        <span class="lc-legend-item">
          <span class="lc-legend-dot" style="background:${ds._color}"></span>
          ${ds.label}
        </span>`).join('');
    }
  },

  // ── Sky view placeholder (Aladin fills it in renderDetail) ────────────────
  renderSkyView() {
    // The actual Aladin div lives inside renderHostGalaxy so that it's visually
    // grouped with the galaxy context.  Nothing extra needed here.
    return '';
  },
};

document.addEventListener('DOMContentLoaded', () => sv.init());
