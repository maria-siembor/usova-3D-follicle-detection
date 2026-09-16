const form = document.querySelector('#upload-form');
const fileInput = document.querySelector('#volume-file');
const fileLabel = document.querySelector('#file-label');
const dropzone = document.querySelector('#dropzone');
const errorBox = document.querySelector('#error');
const workspace = document.querySelector('#workspace');
const slider = document.querySelector('#slice-slider');
const canvas = document.querySelector('#slice-canvas');
const ctx = canvas.getContext('2d');
let analysis = null;
let selectedId = null;
let currentSlice = null;

fileInput.addEventListener('change', () => {
  fileLabel.textContent = fileInput.files[0]?.name || 'Choose a VTK volume';
});

['dragenter', 'dragover'].forEach(eventName => dropzone.addEventListener(eventName, event => {
  event.preventDefault();
  dropzone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach(eventName => dropzone.addEventListener(eventName, event => {
  event.preventDefault();
  dropzone.classList.remove('dragging');
}));
dropzone.addEventListener('drop', event => {
  fileInput.files = event.dataTransfer.files;
  fileLabel.textContent = fileInput.files[0]?.name || 'Choose a VTK volume';
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  errorBox.textContent = '';
  if (!fileInput.files.length) return;
  const button = form.querySelector('button');
  button.disabled = true;
  button.querySelector('span').textContent = 'Analysing volume...';
  try {
    const response = await fetch('/api/analyse', { method: 'POST', body: new FormData(form) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error);
    analysis = payload;
    selectedId = null;
    renderAnalysis(payload);
    workspace.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    errorBox.textContent = error.message || 'The analysis could not be completed.';
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = 'Run detection';
  }
});

function renderAnalysis(payload) {
  workspace.classList.remove('hidden');
  const measurements = payload.measurements;
  document.querySelector('#total-count').textContent = payload.summary.total_follicle_count;
  document.querySelector('#optimal-count').textContent = payload.summary.optimal_for_retrieval_count;
  document.querySelector('#mean-volume').textContent = formatNumber(
    measurements.reduce((sum, item) => sum + item.volume_mm3, 0) / (measurements.length || 1)
  );
  document.querySelector('#largest-volume').textContent = formatNumber(
    Math.max(0, ...measurements.map(item => item.volume_mm3))
  );
  document.querySelector('#volume-meta').textContent = `${payload.shape.join(' x ')} voxels - ${payload.spacing.map(value => value.toFixed(2)).join(' x ')} mm spacing`;
  document.querySelector('#table-count').textContent = `${measurements.length} structures`;
  slider.max = payload.shape[0] - 1;
  slider.value = payload.first_slice.index;
  analysis.measurementsById = new Map(measurements.map(item => [item.follicle_id, item]));
  updateSlice(payload.first_slice);
  renderTable(measurements);
}

slider.addEventListener('input', async () => {
  if (!analysis) return;
  const index = Number(slider.value);
  const response = await fetch(`/api/results/${analysis.result_id}/slice/${index}`);
  updateSlice(await response.json());
});

function updateSlice(slice) {
  currentSlice = slice;
  const height = slice.image.length;
  const width = slice.image[0].length;
  canvas.width = width;
  canvas.height = height;
  const pixels = ctx.createImageData(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const offset = (y * width + x) * 4;
      const value = slice.image[y][x];
      const label = slice.labels[y][x];
      const status = label ? analysis.measurementsById.get(label)?.maturity : null;
      const color = statusColor(status, label === selectedId);
      pixels.data[offset] = label ? color[0] : value;
      pixels.data[offset + 1] = label ? color[1] : value;
      pixels.data[offset + 2] = label ? color[2] : value;
      pixels.data[offset + 3] = 255;
    }
  }
  ctx.putImageData(pixels, 0, 0);
  document.querySelector('#viewer-empty').style.display = 'none';
  document.querySelector('#slice-output').textContent = slice.index;
  document.querySelector('#slice-readout').textContent = `Slice ${slice.index + 1} / ${analysis.shape[0]}`;
}

canvas.addEventListener('click', event => {
  if (!currentSlice || !analysis) return;
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width / canvas.width, rect.height / canvas.height);
  const drawnWidth = canvas.width * scale;
  const drawnHeight = canvas.height * scale;
  const x = Math.floor((event.clientX - rect.left - (rect.width - drawnWidth) / 2) / scale);
  const y = Math.floor((event.clientY - rect.top - (rect.height - drawnHeight) / 2) / scale);
  if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return;
  const label = currentSlice.labels[y][x];
  if (label) selectMeasurement(label);
});

function renderTable(measurements) {
  document.querySelector('#measurements-body').innerHTML = measurements.map(item =>
    `<tr data-id="${item.follicle_id}"><td>#${item.follicle_id}</td><td>${item.equivalent_diameter_mm.toFixed(1)} mm</td><td>${formatNumber(item.volume_mm3)} mm³</td><td><span class="status-pill ${item.maturity}">${item.maturity.replaceAll('_', ' ')}</span></td></tr>`
  ).join('');
  document.querySelectorAll('tbody tr').forEach(row => row.addEventListener('click', () => {
    selectMeasurement(Number(row.dataset.id));
  }));
}

function selectMeasurement(id) {
  selectedId = Number(id);
  document.querySelectorAll('tbody tr').forEach(row => {
    row.classList.toggle('selected', Number(row.dataset.id) === selectedId);
  });
  if (currentSlice) updateSlice(currentSlice);
}

async function updateSliceFromSlider() {
  const response = await fetch(`/api/results/${analysis.result_id}/slice/${slider.value}`);
  updateSlice(await response.json());
}

function formatNumber(value) {
  return Number.isFinite(value) ? value.toFixed(1) : '0.0';
}

function statusColor(status, selected) {
  if (selected) return [255, 255, 255];
  if (status === 'optimal_for_retrieval') return [55, 190, 125];
  if (status === 'post_mature') return [238, 112, 86];
  return [239, 194, 65];
}