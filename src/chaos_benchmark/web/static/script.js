// Lorenz Attractor Canvas Animation
const canvas = document.getElementById('lorenz-canvas');
const ctx = canvas.getContext('2d');

let width, height;
let x = 0.01, y = 0, z = 0;
const sigma = 10;
const rho = 28;
const beta = 8/3;
const dt = 0.005;

function resize() {
    width = canvas.width = canvas.offsetWidth;
    height = canvas.height = canvas.offsetHeight;
    ctx.fillStyle = '#FAFAFA';
    ctx.fillRect(0, 0, width, height);
}

window.addEventListener('resize', resize);
resize();

ctx.lineWidth = 1;
ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';

function drawLorenz() {
    for (let i = 0; i < 15; i++) { // steps per frame
        const dx = sigma * (y - x) * dt;
        const dy = (x * (rho - z) - y) * dt;
        const dz = (x * y - beta * z) * dt;
        
        const oldX = x;
        const oldZ = z;
        
        x += dx;
        y += dy;
        z += dz;
        
        // Map 3D to 2D
        const scale = 7;
        const screenX = width / 2 + x * scale;
        const screenY = height - 50 - z * scale;
        
        const oldScreenX = width / 2 + oldX * scale;
        const oldScreenY = height - 50 - oldZ * scale;
        
        ctx.beginPath();
        ctx.moveTo(oldScreenX, oldScreenY);
        ctx.lineTo(screenX, screenY);
        ctx.stroke();
    }
    requestAnimationFrame(drawLorenz);
}
drawLorenz();

// Contextual UI Logic
const checkRossler = document.getElementById('check-rossler');
const noteRossler = document.getElementById('note-rossler');

checkRossler.addEventListener('change', (e) => {
    if (e.target.checked) {
        noteRossler.classList.remove('hidden');
    } else {
        noteRossler.classList.add('hidden');
    }
});

// Generator Logic
const form = document.getElementById('generate-form');
const submitBtn = document.getElementById('submit-btn');
const resultsContainer = document.getElementById('results-container');
const progressTable = document.getElementById('progress-table').querySelector('tbody');
const completionView = document.getElementById('completion-view');
const loadingSpinner = document.getElementById('loading-spinner');
const downloadBtn = document.getElementById('download-btn');
const terminateBtn = document.getElementById('terminate-btn');
const previewHead = document.getElementById('preview-head');
const previewBody = document.getElementById('preview-body');

let activeJobId = null;
let pollInterval = null;
let expectedTotal = 0;
let isSubmitting = false;

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (isSubmitting) return;
    isSubmitting = true;
    
    const rowsPerClass = document.getElementById('rows_per_class').value;
    const maxAttempts = document.getElementById('max_attempts').value;
    const systems = Array.from(document.querySelectorAll('input[name="systems"]:checked')).map(cb => cb.value);

    if (systems.length === 0) {
        alert("Select at least one system.");
        isSubmitting = false;
        return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = "Initiating...";
    
    // Reset Views
    resultsContainer.classList.remove('hidden');
    completionView.classList.add('hidden');
    loadingSpinner.classList.remove('hidden');
    terminateBtn.classList.remove('hidden');
    terminateBtn.disabled = false;
    terminateBtn.textContent = 'Terminate';
    
    progressTable.innerHTML = '';
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('progress-text').textContent = '0%';
    expectedTotal = systems.length * 3 * parseInt(rowsPerClass);

    // Pre-populate table with zeros
    renderProgress(Object.fromEntries(systems.map(s => [s, {Stable: 0, Periodic: 0, Chaotic: 0}])));
    
    try {
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                rows_per_class: parseInt(rowsPerClass),
                max_attempts: parseInt(maxAttempts),
                systems: systems
            })
        });
        
        const data = await response.json();
        activeJobId = data.job_id;
        
        submitBtn.textContent = "Simulating...";
        pollInterval = setInterval(pollProgress, 1000);
        
    } catch (err) {
        alert("Error starting simulation: " + err.message);
        isSubmitting = false;
        submitBtn.disabled = false;
        submitBtn.textContent = "Begin Simulation";
    }
});

async function pollProgress() {
    if (!activeJobId) return;
    
    try {
        const response = await fetch(`/api/progress/${activeJobId}`);
        const data = await response.json();
        
        renderProgress(data.progress);
        
        if (data.status === 'complete') {
            clearInterval(pollInterval);
            finishSimulation(data);
        } else if (data.status === 'error') {
            clearInterval(pollInterval);
            alert("Simulation Error: " + data.error);
            resetForm();
        } else if (data.status === 'cancelled') {
            clearInterval(pollInterval);
            resetForm();
        }
    } catch (e) {
        console.error("Polling error", e);
    }
}

terminateBtn.addEventListener('click', async () => {
    if (!activeJobId) return;
    try {
        const response = await fetch(`/api/cancel/${activeJobId}`, { method: 'POST' });
        if (response.ok) {
            terminateBtn.disabled = true;
            terminateBtn.textContent = 'Terminating...';
        }
    } catch (e) {
        console.error("Terminate failed", e);
    }
});

function renderProgress(progressDict) {
    progressTable.innerHTML = '';
    let currentTotal = 0;
    for (const [sys, counts] of Object.entries(progressDict)) {
        currentTotal += counts.Stable + counts.Periodic + counts.Chaotic;
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${sys}</td>
            <td>${counts.Stable}</td>
            <td>${counts.Periodic}</td>
            <td>${counts.Chaotic}</td>
        `;
        progressTable.appendChild(tr);
    }
    
    if (expectedTotal > 0) {
        let percent = (currentTotal / expectedTotal) * 100;
        if (percent > 100) percent = 100;
        document.getElementById('progress-bar').style.width = percent + '%';
        document.getElementById('progress-text').textContent = Math.round(percent) + '%';
    }
}

function finishSimulation(data) {
    isSubmitting = false;
    submitBtn.disabled = false;
    submitBtn.textContent = "Begin Simulation";
    loadingSpinner.classList.add('hidden');
    terminateBtn.classList.add('hidden');
    completionView.classList.remove('hidden');
    document.getElementById('progress-bar').style.width = '100%';
    document.getElementById('progress-text').textContent = '100%';
    
    // Render Preview
    if (data.preview && data.preview.length > 0) {
        const columns = Object.keys(data.preview[0]);
        previewHead.innerHTML = columns.map(c => `<th>${c}</th>`).join('');
        previewBody.innerHTML = data.preview.map(row => {
            return `<tr>${columns.map(c => {
                let val = row[c];
                if (typeof val === 'number') val = val.toFixed(4);
                return `<td>${val}</td>`;
            }).join('')}</tr>`;
        }).join('');
    }
}

function resetForm() {
    isSubmitting = false;
    submitBtn.disabled = false;
    submitBtn.textContent = "Begin Simulation";
    loadingSpinner.classList.add('hidden');
    terminateBtn.classList.add('hidden');
    document.getElementById('progress-text').textContent = '';
}

downloadBtn.addEventListener('click', () => {
    if (!activeJobId) return;
    window.location.href = `/api/download/${activeJobId}`;
});

// System Equation Card Canvases Animation
const sysCanvases = [
    { id: 'canvas-logistic', update: null },
    { id: 'canvas-henon', update: null },
    { id: 'canvas-lorenz', update: null },
    { id: 'canvas-rossler', update: null },
    { id: 'canvas-mg', update: null }
].map(item => {
    const canvas = document.getElementById(item.id);
    if (!canvas) return null;
    return {
        canvas,
        ctx: canvas.getContext('2d'),
        width: 0,
        height: 0,
        state: {}
    };
}).filter(c => c !== null);

function resizeSysCanvases() {
    sysCanvases.forEach(c => {
        c.width = c.canvas.width = c.canvas.offsetWidth;
        c.height = c.canvas.height = c.canvas.offsetHeight;
    });
}
window.addEventListener('resize', resizeSysCanvases);

sysCanvases.forEach(c => {
    c.ctx.lineWidth = 1;
    c.ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
    
    if (c.canvas.id === 'canvas-logistic') {
        c.state = { r: 3.99, x: 0.1 };
        c.update = function() {
            c.ctx.fillStyle = 'rgba(0,0,0,0.5)';
            for(let i=0; i<5; i++) {
                const nx = c.state.r * c.state.x * (1 - c.state.x);
                const px = c.width * c.state.x;
                const py = c.height - (c.height * nx);
                c.ctx.fillRect(px, py, 1.5, 1.5);
                c.state.x = nx;
            }
        };
    }
    
    if (c.canvas.id === 'canvas-henon') {
        c.state = { a: 1.4, b: 0.3, x: 0, y: 0 };
        c.update = function() {
            c.ctx.fillStyle = 'rgba(0,0,0,0.4)';
            for(let i=0; i<15; i++) {
                const nx = 1 - c.state.a * c.state.x * c.state.x + c.state.y;
                const ny = c.state.b * c.state.x;
                const px = c.width/2 + (c.state.x * (c.width/3.2));
                const py = c.height/2 - (c.state.y * (c.height/0.9));
                c.ctx.fillRect(px, py, 1, 1);
                c.state.x = nx;
                c.state.y = ny;
            }
        };
    }
    
    if (c.canvas.id === 'canvas-lorenz') {
        c.state = { x: 0.1, y: 0, z: 0, sigma: 10, rho: 28, beta: 8/3, dt: 0.005 };
        c.update = function() {
            c.ctx.beginPath();
            for(let i=0; i<8; i++) {
                const dx = c.state.sigma * (c.state.y - c.state.x) * c.state.dt;
                const dy = (c.state.x * (c.state.rho - c.state.z) - c.state.y) * c.state.dt;
                const dz = (c.state.x * c.state.y - c.state.beta * c.state.z) * c.state.dt;
                
                const oldPx = c.width/2 + c.state.x * (c.height/40);
                const oldPy = c.height - c.state.z * (c.height/40);
                
                c.state.x += dx;
                c.state.y += dy;
                c.state.z += dz;
                
                const px = c.width/2 + c.state.x * (c.height/40);
                const py = c.height - c.state.z * (c.height/40);
                
                if (i===0) c.ctx.moveTo(oldPx, oldPy);
                c.ctx.lineTo(px, py);
            }
            c.ctx.stroke();
        };
    }
    
    if (c.canvas.id === 'canvas-rossler') {
        c.state = { x: 0.1, y: 0, z: 0, a: 0.2, b: 0.2, c: 5.7, dt: 0.05 };
        c.update = function() {
            c.ctx.beginPath();
            for(let i=0; i<8; i++) {
                const dx = (-c.state.y - c.state.z) * c.state.dt;
                const dy = (c.state.x + c.state.a * c.state.y) * c.state.dt;
                const dz = (c.state.b + c.state.z * (c.state.x - c.state.c)) * c.state.dt;
                
                const oldPx = c.width/2 + c.state.x * (c.height/25);
                const oldPy = c.height/2 - c.state.y * (c.height/25);
                
                c.state.x += dx;
                c.state.y += dy;
                c.state.z += dz;
                
                const px = c.width/2 + c.state.x * (c.height/25);
                const py = c.height/2 - c.state.y * (c.height/25);
                
                if (i===0) c.ctx.moveTo(oldPx, oldPy);
                c.ctx.lineTo(px, py);
            }
            c.ctx.stroke();
        };
    }
    
    if (c.canvas.id === 'canvas-mg') {
        c.state = { hist: Array(100).fill(1.2), t: 0, beta: 0.2, gamma: 0.1, n: 10, tau: 17, dt: 0.2, step: 0 };
        c.update = function() {
            c.ctx.beginPath();
            for(let i=0; i<3; i++) {
                const x_tau = c.state.hist[0];
                const x_cur = c.state.hist[c.state.hist.length - 1];
                
                const dx = (c.state.beta * x_tau / (1 + Math.pow(x_tau, c.state.n)) - c.state.gamma * x_cur) * c.state.dt;
                const nx = x_cur + dx;
                
                c.state.hist.shift();
                c.state.hist.push(nx);
                
                const oldPx = (c.state.step % c.width);
                const oldPy = c.height - (x_cur * (c.height/2.2));
                
                c.state.step += 1;
                
                const px = (c.state.step % c.width);
                const py = c.height - (nx * (c.height/2.2));
                
                if (px < oldPx) {
                    c.ctx.clearRect(0, 0, c.width, c.height);
                    c.ctx.beginPath();
                    c.ctx.moveTo(px, py);
                } else {
                    if (i===0) c.ctx.moveTo(oldPx, oldPy);
                    c.ctx.lineTo(px, py);
                }
            }
            c.ctx.stroke();
        };
    }
});

setTimeout(() => {
    resizeSysCanvases();
    function renderCanvases() {
        sysCanvases.forEach(c => c.update());
        requestAnimationFrame(renderCanvases);
    }
    renderCanvases();
}, 200);
