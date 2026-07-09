// Simple transient viewer with pre-generated images and lightcurves

// Helper: recursively sanitize values (replace NaN/Infinity and string 'NaN' with 0.0)
function deepSanitize(value) {
  if (Array.isArray(value)) {
    return value.map(deepSanitize);
  }
  if (value && typeof value === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(value)) {
      out[k] = deepSanitize(v);
    }
    return out;
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : 0.0;
  }
  if (typeof value === 'string') {
    const s = value.trim();
    if (s === 'NaN' || s === 'nan' || s === 'Infinity' || s === '-Infinity') {
      return 0.0;
    }
    return value;
  }
  return value;
}

// Helper: fetch JSON as text, then parse with fallback token replacement for NaN/Infinity
async function fetchAndSanitizeJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url} (${response.status})`);
  }
  const text = await response.text();
  try {
    return deepSanitize(JSON.parse(text));
  } catch (e) {
    // Replace bare NaN/Infinity tokens which are invalid in strict JSON
    const sanitizedText = text
      .replace(/\bNaN\b/g, '0.0')
      .replace(/\b-?Infinity\b/g, '0.0');
    return deepSanitize(JSON.parse(sanitizedText));
  }
}
const transientViewer = {
  candidates: [],
  filteredCandidates: [],
  selectedCandidate: null,
  currentCutoutIndex: 0,
  currentFilter: 'all',
  
  // Lazy rendering settings
  renderBatchSize: 50,
  renderedCount: 0,
  
  
  async init() {
    try {
      // Load all candidates from single candidates.json file
      this.candidates = await fetchAndSanitizeJson('candidates.json');
      this.filteredCandidates = [...this.candidates];
      
      this.renderFilterControls();
      this.renderCandidateList();
    } catch (error) {
      console.error('Failed to load candidates:', error);
      document.getElementById('candidate-list').innerHTML = 
        '<div style="padding: 20px; color: red;">Failed to load candidate data: ' + error.message + '</div>';
    }
  },

  renderFilterControls() {
    const filterContainer = document.getElementById('filter-controls');
    if (!filterContainer) return;
    
    const typeCounts = this.candidates.reduce((acc, candidate) => {
      const type = candidate.candidate_type || 'new';
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {});
    
    filterContainer.innerHTML = `
      <div style="padding: 10px; border-bottom: 1px solid #ddd;">
        <h4 style="margin: 0 0 10px 0; font-size: 0.9em;">Filter by Type:</h4>
        <div style="display: flex; flex-direction: column; gap: 5px;">
          <label style="font-size: 0.8em; cursor: pointer;">
            <input type="radio" name="candidateFilter" value="all" ${this.currentFilter === 'all' ? 'checked' : ''} 
                   onchange="transientViewer.filterCandidates(this.value)">
            All (${this.candidates.length})
          </label>
          <label style="font-size: 0.8em; cursor: pointer;">
            <input type="radio" name="candidateFilter" value="new" ${this.currentFilter === 'new' ? 'checked' : ''} 
                   onchange="transientViewer.filterCandidates(this.value)">
            ✨ New (${typeCounts['new'] || 0})
          </label>
          <label style="font-size: 0.8em; cursor: pointer;">
            <input type="radio" name="candidateFilter" value="brightening" ${this.currentFilter === 'brightening' ? 'checked' : ''} 
                   onchange="transientViewer.filterCandidates(this.value)">
            📈 Brightening (${typeCounts['brightening'] || 0})
          </label>
          <label style="font-size: 0.8em; cursor: pointer;">
            <input type="radio" name="candidateFilter" value="fading" ${this.currentFilter === 'fading' ? 'checked' : ''} 
                   onchange="transientViewer.filterCandidates(this.value)">
            📉 Fading (${typeCounts['fading'] || 0})
          </label>
          <label style="font-size: 0.8em; cursor: pointer;">
            <input type="radio" name="candidateFilter" value="trail" ${this.currentFilter === 'trail' ? 'checked' : ''} 
                   onchange="transientViewer.filterCandidates(this.value)">
            🧵 Trail (${typeCounts['trail'] || 0})
          </label>
        </div>
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee;">
          <label style="font-size: 0.8em; cursor: pointer;">
            <input type="checkbox" id="variableOnly" onchange="transientViewer.filterCandidates()">
            Variable only
          </label>
        </div>
      </div>
    `;
  },

  filterCandidates(filterType) {
    if (filterType) {
      this.currentFilter = filterType;
    }
    
    const variableOnly = document.getElementById('variableOnly')?.checked;
    
    this.filteredCandidates = this.candidates.filter(candidate => {
      const typeMatch = this.currentFilter === 'all' || candidate.candidate_type === this.currentFilter;
      const variableMatch = !variableOnly || candidate.is_variable === true || candidate.is_variable === 'True';
      return typeMatch && variableMatch;
    });
    
    this.renderCandidateList();
  },
  
  getCandidateTypeBadge(candidateType) {
    const types = {
      'new': { color: '#3498db', icon: '✨', label: 'New' },
      'brightening': { color: '#e74c3c', icon: '📈', label: 'Bright' },
      'fading': { color: '#f39c12', icon: '📉', label: 'Fade' },
      'trail': { color: '#8e44ad', icon: '🧵', label: 'Trail' }
    };
    
    const type = types[candidateType] || types['new'];
    return `<span class="candidate-type-badge" style="background-color: ${type.color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.7em; font-weight: bold; margin-left: 5px;">${type.icon} ${type.label}</span>`;
  },

  renderCandidateList(append = false) {
    const listEl = document.getElementById('candidate-list-content');
    
    if (this.filteredCandidates.length === 0) {
      listEl.innerHTML = '<div style="padding: 20px; text-align: center;">No candidates match current filter</div>';
      this.renderedCount = 0;
      return;
    }
    
    // Reset rendered count when not appending (new filter)
    if (!append) {
      this.renderedCount = 0;
    }
    
    // Determine how many items to render
    const startIndex = append ? this.renderedCount : 0;
    const endIndex = Math.min(startIndex + this.renderBatchSize, this.filteredCandidates.length);
    const candidatesToRender = this.filteredCandidates.slice(startIndex, endIndex);
    
    const html = candidatesToRender.map(candidate => {
      const hasLightcurve = candidate.lightcurve && candidate.lightcurve.plot;
      const variabilityInfo = candidate.lightcurve && candidate.lightcurve.data ? 
        ` (${candidate.lightcurve.data.n_points} pts)` : '';
      const magDiff = candidate.magnitude_difference || 0;
      const fwhmRatio = candidate.fwhm_ratio || 0;
      const candidateType = candidate.candidate_type || 'new';
      
      return `
        <div class="candidate-item ${this.selectedCandidate?.id === candidate.id ? 'selected' : ''}" 
             onclick="transientViewer.selectCandidate('${candidate.id}')">
          <div class="candidate-id">
            ${candidate.id}
            ${hasLightcurve ? '<span style="color: #27ae60; font-size: 0.8em;">📈</span>' : ''}
            ${this.getCandidateTypeBadge(candidateType)}
          </div>
          <div class="candidate-info">
            <div>Mag: ${Number(candidate.MAG_CALIB ?? candidate.mag_weighted_mean ?? 0).toFixed(2)}</div>
            <div>Quality: ${Number(candidate.quality_score ?? 0).toFixed(2)}</div>
          </div>
          <div class="candidate-stats">
            <div>ΔMag: ${magDiff.toFixed(2)} | FWHM: ${fwhmRatio.toFixed(2)}</div>
            <div>Variable: ${(candidate.is_variable === true || candidate.is_variable === 'True') ? '✓' : '✗'} | Cat: ${candidate.reference_catalog || 'Unknown'}</div>
          </div>
          ${variabilityInfo ? `<div style="font-size: 0.7em; color: #666;">${variabilityInfo}</div>` : ''}
        </div>
      `;
    }).join('');
    
    // Update the rendered count
    this.renderedCount = endIndex;
    
    // Set or append the HTML
    if (append) {
      listEl.innerHTML += html;
    } else {
      listEl.innerHTML = html;
    }
    
    // Add or remove "Load more" button as needed
    this.updateLoadMoreButton();
  },
  
  updateLoadMoreButton() {
    const listEl = document.getElementById('candidate-list-content');
    const existingButton = document.getElementById('load-more-button');
    
    // Remove existing button
    if (existingButton) {
      existingButton.remove();
    }
    
    // Add button if there are more items to show
    if (this.renderedCount < this.filteredCandidates.length) {
      const remaining = this.filteredCandidates.length - this.renderedCount;
      const buttonHtml = `
        <div id="load-more-button" style="text-align: center; margin: 20px 0; padding: 20px; border-top: 1px solid #ddd; background-color: #f9f9f9;">
          <button onclick="transientViewer.loadMoreCandidates()" 
                  style="padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 1em;">
            Load ${Math.min(this.renderBatchSize, remaining)} more candidates (${remaining} remaining)
          </button>
        </div>
      `;
      listEl.innerHTML += buttonHtml;
    }
  },
  
  loadMoreCandidates() {
    this.renderCandidateList(true);
  },
  
  selectCandidate(id) {
    this.selectedCandidate = this.candidates.find(c => c.id === id);
    this.currentCutoutIndex = 0; // Reset cutout index when selecting a new candidate
    this.renderCandidateList();
    this.renderCandidateDetail();
  },
  
  nextCutout() {
    if (!this.selectedCandidate || !this.selectedCandidate.cutouts) return;
    
    this.currentCutoutIndex = (this.currentCutoutIndex + 1) % this.selectedCandidate.cutouts.length;
    this.updateCutoutViewer();
  },
  
  prevCutout() {
    if (!this.selectedCandidate || !this.selectedCandidate.cutouts) return;
    
    this.currentCutoutIndex = (this.currentCutoutIndex - 1 + this.selectedCandidate.cutouts.length) % this.selectedCandidate.cutouts.length;
    this.updateCutoutViewer();
  },
  
  updateCutoutViewer() {
    if (!this.selectedCandidate || !this.selectedCandidate.cutouts) return;
    
    const cutouts = this.selectedCandidate.cutouts;
    const currentCutout = cutouts[this.currentCutoutIndex];
    
    // Update image
    const imgContainer = document.querySelector('.cutout-image-container');
    if (imgContainer) {
      imgContainer.innerHTML = `
        <img src="${currentCutout.path}" alt="${currentCutout.filename}" class="cutout-image">
        <div class="cutout-filename">${currentCutout.filename}</div>
      `;
    }
    
    // Update slider
    const slider = document.getElementById('cutout-slider');
    if (slider) {
      slider.value = this.currentCutoutIndex;
    }
    
    // Update time info
    const timeInfo = document.querySelector('.cutout-time-info');
    if (timeInfo) {
      timeInfo.innerHTML = `
        Image ${this.currentCutoutIndex + 1} of ${cutouts.length}
        <div class="date-display">Date: ${currentCutout.date || 'Unknown'}</div>
      `;
    }
    
    // Update button states
    document.getElementById('prev-cutout').disabled = cutouts.length <= 1;
    document.getElementById('next-cutout').disabled = cutouts.length <= 1;
  },
  
  onSliderChange(value) {
    this.currentCutoutIndex = parseInt(value);
    this.updateCutoutViewer();
  },
  
  renderLightcurveSection(candidate) {
    if (!candidate.lightcurve) {
      return `
        <div class="detail-section">
          <h3>Lightcurve</h3>
          <div style="padding: 20px; text-align: center; color: #666; border: 1px solid #ddd; border-radius: 4px;">
            No lightcurve data available
          </div>
        </div>
      `;
    }
    
    const lc = candidate.lightcurve;
    let content = '';
    
    // Show lightcurve plot if available
    if (lc.plot) {
      content += `
        <div style="text-align: center; margin-bottom: 15px;">
          <img src="${lc.plot}" alt="Lightcurve for ${candidate.id}" 
               style="max-width: 100%; border: 1px solid #ddd; border-radius: 4px;">
        </div>
      `;
    }
    
    // Show lightcurve statistics
    if (lc.data) {
      content += `
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
          <div>
            <strong>Data Points:</strong> ${lc.data.n_points}<br>
            <strong>Time Span:</strong> ${lc.data.time_span_hours ? Number(lc.data.time_span_hours).toFixed(2) + ' hours' : 'N/A'}
          </div>
          <div>
            <strong>Mag Range:</strong> ${lc.data.mag_range ? Number(lc.data.mag_range).toFixed(3) : 'N/A'}<br>
            <strong>Mag Std:</strong> ${lc.data.mag_std ? Number(lc.data.mag_std).toFixed(3) : 'N/A'}
          </div>
        </div>
      `;
    }
    
    // Interactive lightcurve plot using points data
    if (lc.points && lc.points.length > 0) {
      content += `
        <div style="margin-top: 20px;">
          <h4>Interactive Plot</h4>
          <div id="lightcurve-plot" style="width: 100%; height: 300px; border: 1px solid #ddd; border-radius: 4px;">
            <!-- Interactive plot will be rendered here -->
          </div>
        </div>
      `;
    }
    
    return `
      <div class="detail-section">
        <h3>Lightcurve</h3>
        <div style="border: 1px solid #ddd; border-radius: 4px; padding: 15px; background-color: #f9f9f9;">
          ${content}
        </div>
      </div>
    `;
  },
  
  renderInteractiveLightcurve(candidate) {
    if (!candidate.lightcurve || !candidate.lightcurve.points) return;
    
    const container = document.getElementById('lightcurve-plot');
    if (!container) return;
    
    const points = candidate.lightcurve.points;
    
    // Create SVG
    const margin = {top: 20, right: 30, bottom: 40, left: 60};
    const width = container.offsetWidth - margin.left - margin.right;
    const height = 280 - margin.top - margin.bottom;
    
    container.innerHTML = `
      <svg width="${width + margin.left + margin.right}" height="${height + margin.top + margin.bottom}">
        <g transform="translate(${margin.left}, ${margin.top})">
          <!-- Plot will be drawn here -->
        </g>
      </svg>
    `;
    
    const svg = container.querySelector('svg g');
    
    // Scales
    const times = points.map(p => p.time);
    const mags = points.map(p => p.magnitude);
    
    const xScale = {
      min: Math.min(...times),
      max: Math.max(...times),
      range: width
    };
    
    const yScale = {
      min: Math.min(...mags) - 0.1,
      max: Math.max(...mags) + 0.1,
      range: height
    };
    
    // Helper functions
    const xPos = (time) => ((time - xScale.min) / (xScale.max - xScale.min)) * xScale.range;
    const yPos = (mag) => ((mag - yScale.min) / (yScale.max - yScale.min)) * yScale.range;
    
    // Draw axes
    svg.innerHTML += `
      <!-- X axis -->
      <line x1="0" y1="${height}" x2="${width}" y2="${height}" stroke="#333" stroke-width="1"/>
      <!-- Y axis -->
      <line x1="0" y1="0" x2="0" y2="${height}" stroke="#333" stroke-width="1"/>
      
      <!-- X axis label -->
      <text x="${width/2}" y="${height + 35}" text-anchor="middle" font-size="12" fill="#666">
        Time since first detection (hours)
      </text>
      
      <!-- Y axis label -->
      <text transform="rotate(-90)" x="${-height/2}" y="-40" text-anchor="middle" font-size="12" fill="#666">
        Magnitude
      </text>
    `;
    
    // Draw grid lines
    const numXTicks = 5;
    const numYTicks = 5;
    
    for (let i = 0; i <= numXTicks; i++) {
      const x = (i / numXTicks) * width;
      const timeVal = xScale.min + (i / numXTicks) * (xScale.max - xScale.min);
      svg.innerHTML += `
        <line x1="${x}" y1="0" x2="${x}" y2="${height}" stroke="#eee" stroke-width="1"/>
        <text x="${x}" y="${height + 15}" text-anchor="middle" font-size="10" fill="#666">
          ${timeVal.toFixed(1)}
        </text>
      `;
    }
    
    for (let i = 0; i <= numYTicks; i++) {
      const y = (i / numYTicks) * height;
      const magVal = yScale.min + (i / numYTicks) * (yScale.max - yScale.min);
      svg.innerHTML += `
        <line x1="0" y1="${y}" x2="${width}" y2="${y}" stroke="#eee" stroke-width="1"/>
        <text x="-10" y="${y + 4}" text-anchor="end" font-size="10" fill="#666">
          ${magVal.toFixed(2)}
        </text>
      `;
    }
    
    // Draw error bars and points
    points.forEach((point, i) => {
      const x = xPos(point.time);
      const y = yPos(point.magnitude);
      const errorUp = yPos(point.magnitude - point.error);
      const errorDown = yPos(point.magnitude + point.error);
      
      // Error bar
      svg.innerHTML += `
        <line x1="${x}" y1="${errorUp}" x2="${x}" y2="${errorDown}" stroke="#666" stroke-width="1"/>
        <line x1="${x-2}" y1="${errorUp}" x2="${x+2}" y2="${errorUp}" stroke="#666" stroke-width="1"/>
        <line x1="${x-2}" y1="${errorDown}" x2="${x+2}" y2="${errorDown}" stroke="#666" stroke-width="1"/>
      `;
      
      // Data point
      svg.innerHTML += `
        <circle cx="${x}" cy="${y}" r="4" fill="#3498db" stroke="#2980b9" stroke-width="2">
          <title>Time: ${point.time.toFixed(2)}h, Mag: ${point.magnitude.toFixed(3)} ± ${point.error.toFixed(3)}</title>
        </circle>
      `;
    });
    
    // Connect points with lines
    if (points.length > 1) {
      const pathData = points.map((point, i) => {
        const x = xPos(point.time);
        const y = yPos(point.magnitude);
        return (i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`);
      }).join(' ');
      
      svg.innerHTML += `<path d="${pathData}" stroke="#3498db" stroke-width="2" fill="none" opacity="0.7"/>`;
    }
  },
  
  renderCandidateDetail() {
    const detailEl = document.getElementById('candidate-detail');
    
    if (!this.selectedCandidate) {
      detailEl.innerHTML = `
        <div style="text-align: center; color: #666; margin-top: 50px;">
          Select a candidate from the list to view details
        </div>
      `;
      return;
    }
    
    const candidate = this.selectedCandidate;
    const hasCutouts = candidate.cutouts && candidate.cutouts.length > 0;
    
    detailEl.innerHTML = `
      <h2>${candidate.id}</h2>
      
      <div class="detail-container">
        ${hasCutouts ? `
        <div class="detail-section">
          <h3>Cutout Viewer</h3>
          <div class="cutout-viewer">
            <div class="cutout-nav">
              <button id="prev-cutout" onclick="transientViewer.prevCutout()" ${candidate.cutouts.length <= 1 ? 'disabled' : ''}>
                &lt; Prev
              </button>
              <div class="cutout-time-info">
                Image 1 of ${candidate.cutouts.length}
                <div class="date-display">Date: ${candidate.cutouts[0].date || 'Unknown'}</div>
              </div>
              <button id="next-cutout" onclick="transientViewer.nextCutout()" ${candidate.cutouts.length <= 1 ? 'disabled' : ''}>
                Next &gt;
              </button>
            </div>
            
            <div class="cutout-image-container">
              <img src="${candidate.cutouts[0].path}" alt="${candidate.cutouts[0].filename}" class="cutout-image">
              <div class="cutout-filename">${candidate.cutouts[0].filename}</div>
            </div>
            
            <div class="cutout-slider-container">
              <input type="range" id="cutout-slider" class="cutout-slider" 
                     min="0" max="${candidate.cutouts.length - 1}" value="0" 
                     oninput="transientViewer.onSliderChange(this.value)">
            </div>
          </div>
        </div>
        ` : `
        <div class="detail-section">
          <h3>Cutout Image</h3>
          <div class="cutout-viewer">
            <div class="cutout-image-container">
              <div style="padding: 20px; text-align: center; color: #666;">No cutout images available</div>
            </div>
          </div>
        </div>
        `}
      </div>
      
      ${this.renderLightcurveSection(candidate)}
      
      <div class="detail-section">
        <h3>Properties</h3>
        <table class="properties-table">
          <tr>
            <th>Type:</th>
            <td>${this.getCandidateTypeBadge(candidate.candidate_type || 'new')}</td>
            <th>Quality:</th>
            <td>${Number(candidate.quality_score ?? 0).toFixed(3)}</td>
          </tr>
          <tr>
            <th>RA:</th>
            <td>${candidate.ALPHA_J2000 ? Number(candidate.ALPHA_J2000).toFixed(6) : 'N/A'}°</td>
            <th>Dec:</th>
            <td>${candidate.DELTA_J2000 ? Number(candidate.DELTA_J2000).toFixed(6) : 'N/A'}°</td>
          </tr>
          <tr>
            <th>Magnitude:</th>
            <td>${Number(candidate.MAG_CALIB ?? candidate.mag_weighted_mean ?? 0).toFixed(2)} ± ${Number(candidate.MAGERR_CALIB ?? 0).toFixed(3)}</td>
            <th>Mag difference:</th>
            <td>${Number(candidate.magnitude_difference ?? 0).toFixed(3)}</td>
          </tr>
          <tr>
            <th>Detections:</th>
            <td>${candidate.n_detections || candidate.lightcurve?.data?.n_points || 1}</td>
            <th>Epochs:</th>
            <td>${candidate.n_epochs || 'N/A'}</td>
          </tr>
          <tr>
            <th>FWHM:</th>
            <td>${Number(candidate.FWHM_IMAGE ?? 0).toFixed(2)}</td>
            <th>FWHM ratio:</th>
            <td>${Number(candidate.fwhm_ratio ?? 0).toFixed(3)}</td>
          </tr>
          <tr>
            <th>Ellipticity:</th>
            <td>${Number(candidate.ELLIPTICITY ?? 0).toFixed(3)}</td>
            <th>Catalog:</th>
            <td>${candidate.reference_catalog || 'Unknown'}</td>
          </tr>
          <tr>
            <th>Time span:</th>
            <td>${candidate.time_span_hours ? Number(candidate.time_span_hours).toFixed(2) + ' hours' : 'N/A'}</td>
            <th>Position scatter:</th>
            <td>${candidate.position_scatter_arcsec ? Number(candidate.position_scatter_arcsec).toFixed(3) + ' arcsec' : 'N/A'}</td>
          </tr>
          <tr>
            <th>Variability:</th>
            <td>${(candidate.is_variable === true || candidate.is_variable === 'True') ? '<span style="color: #e74c3c; font-weight: bold;">Yes</span>' : '<span style="color: #27ae60;">No</span>'}</td>
            <th>Mag range:</th>
            <td>${candidate.mag_range ? Number(candidate.mag_range).toFixed(3) : 'N/A'}</td>
          </tr>
          <tr>
            <th>Mag std:</th>
            <td>${candidate.mag_std ? Number(candidate.mag_std).toFixed(3) : 'N/A'}</td>
            <th>Chi² reduced:</th>
            <td>${candidate.mag_chi2_reduced ? Number(candidate.mag_chi2_reduced).toFixed(2) : 'N/A'}</td>
          </tr>
          <tr>
            <th>Nearby sources:</th>
            <td>${candidate.nearby_sources || 0}</td>
            <th>Nearest distance:</th>
            <td>${candidate.nearest_source_dist ? Number(candidate.nearest_source_dist).toFixed(1) + ' arcsec' : 'N/A'}</td>
          </tr>
          <tr>
            <th>Saturated:</th>
            <td>${candidate.saturated === 'True' ? '<span style="color: #e74c3c;">Yes</span>' : 'No'}</td>
            <th>Blended:</th>
            <td>${candidate.blended === 'True' ? '<span style="color: #e74c3c;">Yes</span>' : 'No'}</td>
          </tr>
          <tr>
            <th>Near bright:</th>
            <td>${candidate.near_bright === 'True' ? '<span style="color: #f39c12;">Yes</span>' : 'No'}</td>
            <th>Source density:</th>
            <td>${candidate.source_density ? Number(candidate.source_density).toFixed(6) : '0.000000'}</td>
          </tr>
          ${candidate.ps1_n_det != null ? `
          <tr>
            <th>PS1 history:</th>
            <td>${candidate.ps1_n_det} det${candidate.ps1_brightest != null ? ` (brightest ${Number(candidate.ps1_brightest).toFixed(2)})` : ''}</td>
            <th>PS1 &Delta;mag:</th>
            <td>${(() => {
              const cur = candidate.MAG_CALIB ?? candidate.mag_weighted_mean;
              const bright = candidate.ps1_brightest;
              if (bright == null || cur == null) return 'N/A';
              const delta = Number(bright) - Number(cur);
              const col = delta > 0.3 ? '#27ae60' : delta < -0.3 ? '#e74c3c' : '#888';
              return `<span style="color:${col};font-weight:bold">${delta >= 0 ? '+' : ''}${delta.toFixed(2)}</span>`;
            })()}</td>
          </tr>` : ''}
        </table>
      </div>
      
      <div class="detail-section">
        <h3>All Cutout Images</h3>
        <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">
          ${hasCutouts ? candidate.cutouts.map((cutout, index) => `
            <div style="width: 120px; margin-bottom: 15px; text-align: center; cursor: pointer;" 
                 onclick="transientViewer.currentCutoutIndex=${index}; transientViewer.updateCutoutViewer();">
              <img src="${cutout.path}" alt="${cutout.filename}" 
                   title="${cutout.filename}" style="max-width: 100%; border: 1px solid #ddd;">
              <div style="font-size: 0.8em; color: #666; margin-top: 5px; font-family: monospace;">
                ${cutout.filename}
              </div>
            </div>
          `).join('') : '<div style="padding: 20px; color: #666;">No cutout images available</div>'}
        </div>
      </div>
    `;
    
    if (hasCutouts) {
      this.updateCutoutViewer();
    }
    
    // Render interactive lightcurve after DOM is updated
    setTimeout(() => {
      this.renderInteractiveLightcurve(candidate);
    }, 100);
  }
};

// Initialize the viewer when the page loads
document.addEventListener('DOMContentLoaded', () => {
  transientViewer.init();
});
