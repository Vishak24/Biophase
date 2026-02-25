/* ============================================================
   BioPhase AI — app.js
   Three.js particle canvas + D3 growth curve + ONNX inference
   ============================================================ */

'use strict';

// ─── Phase Metadata ─────────────────────────────────────────
const PHASES = [
    {
        index: 0, name: 'Lag Phase', short: 'LAG',
        color: '#3b82f6', bandId: 'band-lag',
        desc: 'Bacteria adapting — RNA & enzyme synthesis ramping up. No binary fission yet.',
        progressRange: [0.0, 0.15],
    },
    {
        index: 1, name: 'Exponential', short: 'LOG',
        color: '#22c55e', bandId: 'band-log',
        desc: 'Peak health — rapid binary fission, doubling at a constant rate. Max metabolic activity.',
        progressRange: [0.15, 0.50],
    },
    {
        index: 2, name: 'Stationary', short: 'STATIONARY',
        color: '#f59e0b', bandId: 'band-stationary',
        desc: 'Birth rate equals death rate. Nutrients depleted, toxic metabolites accumulate.',
        progressRange: [0.50, 0.80],
    },
    {
        index: 3, name: 'Death Phase', short: 'DEATH',
        color: '#ef4444', bandId: 'band-death',
        desc: 'Cell death exceeds division. Population crashes logarithmically as energy sources vanish.',
        progressRange: [0.80, 1.00],
    },
];

// ─── App State ───────────────────────────────────────────────
const state = {
    onnxSession: null,
    scalerMean: null,
    scalerScale: null,
    featureNames: null,
    currentPhaseIdx: 1,
    currentConfidence: 0,
    currentProgress: 0.3,
    modelReady: false,
    particlePhase: 1,       // drives Three.js mode
    particleIntensity: 0.7, // confidence-driven
};

// ─── DOM Refs ────────────────────────────────────────────────
const loadingOverlay = document.getElementById('model-loading');
const loadingText = document.getElementById('loading-text');
const loadingSubtext = document.getElementById('loading-sub');
const predictBtn = document.getElementById('predict-btn');
const serverStatus = document.getElementById('server-status');
const phaseName = document.getElementById('phase-name');
const phaseDesc = document.getElementById('phase-desc');
const gaugeProgress = document.getElementById('gauge-progress');
const confLabel = document.getElementById('confidence-val');

// ─── Slider refs ─────────────────────────────────────────────
const odInput = document.getElementById('od-input');
const rateInput = document.getElementById('rate-input');
const phInput = document.getElementById('ph-input');
const nutInput = document.getElementById('nut-input');

// ─────────────────────────────────────────────────────────────
//  1. THREE.JS PARTICLE CANVAS
// ─────────────────────────────────────────────────────────────
(function initThree() {
    const canvas = document.getElementById('bg-canvas');
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(0, 1, 1, 0, -1, 1);

    // ── Particle setup ──────────────────────────────
    const PARTICLE_COUNT = 180;
    const positions = new Float32Array(PARTICLE_COUNT * 3);
    const velocities = new Float32Array(PARTICLE_COUNT * 3);
    const sizes = new Float32Array(PARTICLE_COUNT);
    const alphas = new Float32Array(PARTICLE_COUNT);
    const phases = new Float32Array(PARTICLE_COUNT); // orbit angle for stationary

    // Color arrays: each particle has RGB
    const colors = new Float32Array(PARTICLE_COUNT * 3);

    function randomizeParticle(i) {
        positions[i * 3] = Math.random();
        positions[i * 3 + 1] = Math.random();
        positions[i * 3 + 2] = 0;
        velocities[i * 3] = (Math.random() - 0.5) * 0.002;
        velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.002;
        velocities[i * 3 + 2] = 0;
        sizes[i] = Math.random() * 8 + 3;
        alphas[i] = Math.random() * 0.6 + 0.2;
        phases[i] = Math.random() * Math.PI * 2;
    }

    for (let i = 0; i < PARTICLE_COUNT; i++) randomizeParticle(i);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const material = new THREE.PointsMaterial({
        size: 0.012,
        vertexColors: true,
        transparent: true,
        opacity: 0.55,
        sizeAttenuation: false,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    // ── Phase color palettes ────────────────────────
    const phaseColors = [
        [0.24, 0.51, 0.96],  // Lag   – blue
        [0.13, 0.77, 0.37],  // Log   – green
        [0.96, 0.62, 0.04],  // Stat  – amber
        [0.94, 0.27, 0.27],  // Death – red
    ];

    let clock = 0;
    // Split state: each particle has a "child" born during log phase
    const splitProgress = new Float32Array(PARTICLE_COUNT).fill(0);

    function animateParticles() {
        clock += 0.016;
        const phase = state.particlePhase;
        const intensity = state.particleIntensity;
        const [r, g, b] = phaseColors[phase];

        for (let i = 0; i < PARTICLE_COUNT; i++) {
            const ix = i * 3;

            if (phase === 0) {
                // ── LAG: slow random drift, dim blueish
                positions[ix] += velocities[ix] * 0.4;
                positions[ix + 1] += velocities[ix + 1] * 0.4;
                // bounce
                if (positions[ix] < 0 || positions[ix] > 1) velocities[ix] *= -1;
                if (positions[ix + 1] < 0 || positions[ix + 1] > 1) velocities[ix + 1] *= -1;
                colors[ix] = r * (0.5 + 0.5 * intensity);
                colors[ix + 1] = g * (0.5 + 0.5 * intensity);
                colors[ix + 2] = b * (0.5 + 0.5 * intensity);
                sizes[i] = 4 + intensity * 4;
                alphas[i] = 0.3 + intensity * 0.3;

            } else if (phase === 1) {
                // ── LOG: binary fission — particles split & sprint
                const speed = 0.003 * (1 + intensity * 2);
                positions[ix] += velocities[ix] * speed / 0.002;
                positions[ix + 1] += velocities[ix + 1] * speed / 0.002;
                if (positions[ix] < 0 || positions[ix] > 1) { velocities[ix] *= -1; randomizeParticle(i); }
                if (positions[ix + 1] < 0 || positions[ix + 1] > 1) { velocities[ix + 1] *= -1; randomizeParticle(i); }
                // binary-fission flash: pulse size rapidly
                const fissionFlash = 0.5 + 0.5 * Math.sin(clock * 8 + i * 1.3);
                sizes[i] = (5 + intensity * 8) * (0.7 + 0.3 * fissionFlash);
                colors[ix] = r * (0.8 + 0.2 * fissionFlash);
                colors[ix + 1] = g * (0.8 + 0.2 * fissionFlash);
                colors[ix + 2] = b * 0.4;
                alphas[i] = 0.5 + intensity * 0.4;

            } else if (phase === 2) {
                // ── STATIONARY: slow circular orbits
                const cx = 0.5, cy = 0.5;
                const orbitR = 0.05 + (i % 5) * 0.08;
                const orbitSpeed = 0.3 * (1 + intensity * 0.5) * (i % 2 === 0 ? 1 : -1);
                phases[i] += orbitSpeed * 0.016;
                positions[ix] = cx + Math.cos(phases[i] + i * 0.4) * orbitR + (Math.random() - 0.5) * 0.01;
                positions[ix + 1] = cy + Math.sin(phases[i] + i * 0.4) * orbitR + (Math.random() - 0.5) * 0.01;
                const pulse = 0.7 + 0.3 * Math.sin(clock * 2 + i);
                sizes[i] = (4 + intensity * 5) * pulse;
                colors[ix] = r * pulse;
                colors[ix + 1] = g * pulse;
                colors[ix + 2] = b * 0.3;
                alphas[i] = 0.4 + intensity * 0.3;

            } else {
                // ── DEATH: particles fade and fall downward
                positions[ix] += (Math.random() - 0.5) * 0.0005;
                positions[ix + 1] += 0.001 * (1 + intensity);   // drift down
                alphas[i] -= 0.001 * (1 + intensity);
                if (alphas[i] <= 0.02 || positions[ix + 1] > 1.05) {
                    randomizeParticle(i);
                    positions[ix + 1] = 0; // restart from top
                    alphas[i] = 0.6;
                }
                const fade = Math.max(0, alphas[i]);
                colors[ix] = r * fade;
                colors[ix + 1] = g * fade;
                colors[ix + 2] = b * fade;
                sizes[i] = (6 + intensity * 6) * fade;
            }
        }

        geometry.attributes.position.needsUpdate = true;
        geometry.attributes.color.needsUpdate = true;
        geometry.attributes.size.needsUpdate = true;

        renderer.render(scene, camera);
        requestAnimationFrame(animateParticles);
    }

    function onResize() {
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
    window.addEventListener('resize', onResize);
    onResize();
    animateParticles();
})();


// ─────────────────────────────────────────────────────────────
//  2. D3 GROWTH CURVE
// ─────────────────────────────────────────────────────────────
let growthPathEl, currentPointEl, svgEl;

function initGrowthCurve() {
    svgEl = document.getElementById('growth-svg');
    growthPathEl = document.getElementById('growth-path');
    currentPointEl = document.getElementById('current-point');

    buildPathData();

    // Animate path draw on load
    const pathLength = growthPathEl.getTotalLength();
    growthPathEl.style.strokeDasharray = pathLength;
    growthPathEl.style.strokeDashoffset = pathLength;
    growthPathEl.style.transition = 'stroke-dashoffset 2.8s cubic-bezier(0.4, 0, 0.2, 1)';

    requestAnimationFrame(() => {
        growthPathEl.style.strokeDashoffset = '0';
    });

    // Place dot at default position after curve draws
    setTimeout(() => moveDotToProgress(state.currentProgress), 400);
}

function buildPathData() {
    const container = document.getElementById('chart-container');
    const W = container.offsetWidth || 800;
    const H = container.offsetHeight || 400;

    const points = [];
    const STEPS = 200;

    for (let step = 0; step <= STEPS; step++) {
        const t = step / STEPS;
        let y;

        if (t < 0.15) {
            // Lag: flat baseline
            const localT = t / 0.15;
            y = H * 0.88 - localT * H * 0.04;
        } else if (t < 0.50) {
            // Log: sigmoid rise
            const localT = (t - 0.15) / 0.35;
            const s = 1 / (1 + Math.exp(-10 * (localT - 0.5)));
            y = H * 0.84 - s * H * 0.68;
        } else if (t < 0.80) {
            // Stationary: plateau with micro-fluctuation
            const localT = (t - 0.50) / 0.30;
            y = H * 0.16 + Math.sin(localT * Math.PI * 4) * H * 0.012;
        } else {
            // Death: decline
            const localT = (t - 0.80) / 0.20;
            y = H * 0.16 + localT * H * 0.38;
        }

        // Biological micro-jitter
        y += Math.sin(step * 0.22) * 2.5;

        const x = t * W;
        points.push([x, y]);
    }

    const d = 'M ' + points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' L ');
    growthPathEl.setAttribute('d', d);
}

function moveDotToProgress(progress) {
    if (!growthPathEl) return;
    const pathLength = growthPathEl.getTotalLength();
    if (!pathLength) return;
    const pt = growthPathEl.getPointAtLength(pathLength * Math.max(0, Math.min(1, progress)));
    currentPointEl.style.transition = 'cx 0.9s cubic-bezier(0.4,0,0.2,1), cy 0.9s cubic-bezier(0.4,0,0.2,1)';
    currentPointEl.setAttribute('cx', pt.x.toFixed(2));
    currentPointEl.setAttribute('cy', pt.y.toFixed(2));
}

function highlightPhaseBand(phaseIdx) {
    const bandIds = ['band-lag', 'band-log', 'band-stationary', 'band-death'];
    bandIds.forEach((id, i) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.classList.toggle('active-phase', i === phaseIdx);
    });
}

// Rebuild path on window resize
window.addEventListener('resize', () => {
    if (growthPathEl) {
        buildPathData();
        moveDotToProgress(state.currentProgress);
    }
});


// ─────────────────────────────────────────────────────────────
//  3. ONNX INFERENCE + SCALER
// ─────────────────────────────────────────────────────────────
async function loadScaler() {
    loadingSubtext.textContent = 'Loading scaler parameters…';
    const resp = await fetch('scaler.json');
    if (!resp.ok) throw new Error(`Failed to load scaler.json: ${resp.status}`);
    const data = await resp.json();
    state.scalerMean = new Float32Array(data.mean);
    state.scalerScale = new Float32Array(data.scale);
    state.featureNames = data.features;
}

async function loadModel() {
    loadingText.textContent = 'LOADING MODEL…';
    loadingSubtext.textContent = 'Fetching bacteria_model.onnx (≈620 KB)';

    // Configure ONNX wasm path
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';

    state.onnxSession = await ort.InferenceSession.create('bacteria_model.onnx', {
        executionProviders: ['wasm'],
    });
    loadingSubtext.textContent = 'Model loaded ✓';
}

function scaleFeatures(raw) {
    // raw = [od600, gr, acc, nut, ph, od_std, log_od, tnorm]
    const scaled = new Float32Array(8);
    for (let i = 0; i < 8; i++) {
        scaled[i] = (raw[i] - state.scalerMean[i]) / state.scalerScale[i];
    }
    return scaled;
}

async function runInference() {
    const od600 = parseFloat(odInput.value);
    const rateMu = parseFloat(rateInput.value);  // display μ [0.1 .. 2.5]
    const ph = parseFloat(phInput.value);
    const nutGpL = parseFloat(nutInput.value);   // grams/L [0 .. 50]

    // ── Feature engineering to match training distribution ──────────
    //
    // Training 'gr' = dOD600/dt  (range ≈ -0.52 .. 0.58, mean ≈ 0.03)
    //   rate slider (μ) 0.1→2.5 is re-centred so that:
    //     μ=0.1  →  gr = -0.30  (dying / no growth)
    //     μ=1.3  →  gr =  0.00  (plateau / stationary)
    //     μ=2.5  →  gr = +0.28  (rapid exponential growth)
    const gr = (rateMu - 1.3) * 0.23;

    // Training 'nut' was stored as a fraction [0..1], not g/L
    const nut = nutGpL / 50.0;

    // tnorm = time / T_total; proxy: OD600 fraction through full range
    const tnorm = Math.min(od600 / 2.0, 1.0);

    const acc = 0;
    const od_std = 0.02;
    const log_od = Math.log1p(od600);

    const rawFeatures = [od600, gr, acc, nut, ph, od_std, log_od, tnorm];
    const scaled = scaleFeatures(rawFeatures);

    const tensor = new ort.Tensor('float32', scaled, [1, 8]);
    const feeds = { float_input: tensor };
    const output = await state.onnxSession.run(feeds);

    // label → Int64 bigint (wasm) or number
    const labelRaw = output['label'].data[0];
    const phaseIdx = Number(labelRaw);

    // probabilities → Float32Array [1,4] flat
    const probData = output['probabilities'].data;
    const confidence = Number(probData[phaseIdx]) || 0;

    // Also pass od600 so updateUI can position the dot continuously
    return { phaseIdx, confidence: Math.max(0, Math.min(1, confidence)), od600 };
}


// ─────────────────────────────────────────────────────────────
//  4. GAUGE ANIMATION
// ─────────────────────────────────────────────────────────────
let gaugeAnimFrame = null;
let gaugeCurrentOffset = 283; // starts at empty

function animateGaugeTo(targetPct, color) {
    if (gaugeAnimFrame) cancelAnimationFrame(gaugeAnimFrame);

    const CIRCUMFERENCE = 283;
    const targetOffset = CIRCUMFERENCE - (targetPct / 100) * CIRCUMFERENCE;

    gaugeProgress.style.stroke = color;

    let start = null;
    const from = gaugeCurrentOffset;
    const DURATION = 1200;

    function step(ts) {
        if (!start) start = ts;
        const t = Math.min((ts - start) / DURATION, 1);
        const eased = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; // ease-in-out
        const offset = from + (targetOffset - from) * eased;
        gaugeProgress.style.strokeDashoffset = offset.toFixed(2);
        gaugeCurrentOffset = offset;
        if (t < 1) gaugeAnimFrame = requestAnimationFrame(step);
    }

    gaugeAnimFrame = requestAnimationFrame(step);
}


// ─────────────────────────────────────────────────────────────
//  5. UI UPDATE
// ─────────────────────────────────────────────────────────────
function updateUI(phaseIdx, confidence, od600) {
    const phase = PHASES[phaseIdx];
    const confPct = Math.round(confidence * 100);
    const od = od600 ?? parseFloat(odInput.value);

    // Phase card
    phaseName.textContent = phase.name;
    phaseName.style.color = phase.color;
    phaseDesc.textContent = phase.desc;

    // Confidence label
    confLabel.textContent = `${confPct}%`;

    // Gauge animation
    animateGaugeTo(confPct, phase.color);

    // Phase band highlight
    highlightPhaseBand(phaseIdx);

    // ── Continuous dot position: driven by OD600, not fixed midpoint ──
    // Map od600 [0..2] → progress [0..1] along the growth curve svg path.
    // Clamp to whichever phase band is predicted for visual consistency.
    const rawProgress = Math.min(od / 2.0, 1.0);
    const [lo, hi] = phase.progressRange;
    // If OD already places us inside the phase band, use it exactly;
    // otherwise clamp to the band's range so dot stays in the right section.
    const progress = rawProgress >= lo && rawProgress <= hi
        ? rawProgress
        : Math.min(Math.max(rawProgress, lo), hi);
    state.currentProgress = progress;
    moveDotToProgress(progress);

    // Three.js particle mode
    state.particlePhase = phaseIdx;
    state.particleIntensity = 0.4 + confidence * 0.6;

    // Metrics panel
    document.getElementById('metric-od').textContent = od.toFixed(3);
    document.getElementById('metric-gr').textContent = parseFloat(rateInput.value).toFixed(2) + ' h⁻¹';
    document.getElementById('metric-ph').textContent = parseFloat(phInput.value).toFixed(1);
    document.getElementById('metric-nut').textContent = parseFloat(nutInput.value).toFixed(1) + ' g/L';
    document.getElementById('metric-logod').textContent = Math.log1p(od).toFixed(4);
}


// ─────────────────────────────────────────────────────────────
//  6. SLIDER LIVE LABELS + DEBOUNCED AUTO-PREDICT
// ─────────────────────────────────────────────────────────────
let _debounceTimer = null;
function debouncedPredict() {
    if (!state.modelReady) return;
    clearTimeout(_debounceTimer);
    _debounceTimer = setTimeout(async () => {
        try {
            const { phaseIdx, confidence, od600 } = await runInference();
            updateUI(phaseIdx, confidence, od600);
        } catch (err) {
            console.warn('Auto-predict error:', err.message);
        }
    }, 280); // 280ms debounce — snappy but not spammy
}

function bindSliders() {
    const pairs = [
        ['od-input', 'od-label', v => parseFloat(v).toFixed(3)],
        ['rate-input', 'rate-label', v => parseFloat(v).toFixed(2)],
        ['ph-input', 'ph-label', v => parseFloat(v).toFixed(1)],
        ['nut-input', 'nut-label', v => parseFloat(v).toFixed(1)],
    ];
    pairs.forEach(([inputId, labelId, fmt]) => {
        const inp = document.getElementById(inputId);
        const label = document.getElementById(labelId);
        inp.addEventListener('input', (e) => {
            label.textContent = fmt(e.target.value);
            // Move dot immediately using OD600 as proxy (instant feedback)
            if (inputId === 'od-input') {
                moveDotToProgress(Math.min(parseFloat(e.target.value) / 2.0, 1.0));
            }
            // Fire inference shortly after user stops dragging
            debouncedPredict();
        });
    });
}


// ─────────────────────────────────────────────────────────────
//  7. PREDICT BUTTON
// ─────────────────────────────────────────────────────────────
function bindPredictButton() {
    predictBtn.addEventListener('click', async () => {
        if (!state.modelReady) return;
        predictBtn.disabled = true;
        predictBtn.textContent = 'ANALYZING…';
        try {
            const { phaseIdx, confidence, od600 } = await runInference();
            updateUI(phaseIdx, confidence, od600);
        } catch (err) {
            console.error('Inference error:', err);
            phaseDesc.textContent = `⚠ Inference error: ${err.message}`;
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = 'RUN PREDICTION';
        }
    });
}


// ─────────────────────────────────────────────────────────────
//  8. BOOTSTRAP
// ─────────────────────────────────────────────────────────────
async function bootstrap() {
    try {
        // Init growth curve immediately (visual)
        initGrowthCurve();
        bindSliders();
        bindPredictButton();

        // Load ML assets
        await loadScaler();
        await loadModel();

        // Ready
        state.modelReady = true;
        serverStatus.textContent = 'MODEL: READY';
        document.getElementById('server-dot').style.backgroundColor = '#22c55e';

        predictBtn.disabled = false;
        predictBtn.textContent = 'RUN PREDICTION';

        // Hide loading overlay
        loadingOverlay.style.opacity = '0';
        setTimeout(() => loadingOverlay.style.display = 'none', 650);

        // Run a default prediction to make the UI feel alive
        const { phaseIdx, confidence, od600 } = await runInference();
        updateUI(phaseIdx, confidence, od600);

    } catch (err) {
        console.error('Bootstrap error:', err);
        loadingText.textContent = 'LOAD ERROR';
        loadingSubtext.textContent = err.message;
        serverStatus.textContent = 'MODEL: ERROR';
        document.getElementById('server-dot').style.backgroundColor = '#ef4444';
    }
}

window.addEventListener('load', bootstrap);
