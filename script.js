const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const backgroundCanvas = document.getElementById('backgroundCanvas');
const bgCtx = backgroundCanvas.getContext('2d');
let drawing = false;
let paths = [];
let currentPath = [];
let drawingCompleted = false;

function resizeCanvas() {
    const container = document.querySelector('.canvas-container');
    const size = Math.min(container.clientWidth, container.clientHeight);
    canvas.width = size;
    canvas.height = size;
    backgroundCanvas.width = size;
    backgroundCanvas.height = size;
    drawBackground();
    redraw();
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Mouse events
canvas.addEventListener('mousedown', (e) => {
    if (!drawingCompleted) startDrawing(e);
});
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mousemove', draw);

// Touch events
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    if (!drawingCompleted) startDrawing(e.touches[0]);
});

canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    draw(e.touches[0]);
});

canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    stopDrawing();
});

function startDrawing(e) {
    drawing = true;
    currentPath = [];
    draw(e);
}

function stopDrawing() {
    drawing = false;
    if (currentPath.length > 0) {
        paths.push(currentPath);
        drawingCompleted = true; // Disable further drawing
    }
    ctx.beginPath();
}

function draw(e) {
    if (!drawing) return;
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    currentPath.push({ x, y });

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// Handle submit button click
document.getElementById('submitBtn').addEventListener('click', () => {
    alert('Drawing submitted successfully!');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    paths = [];
    drawingCompleted = false; // Allow drawing again
    redraw();
});

// Handle clear button click
document.getElementById('clearBtn').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    paths = [];
    drawingCompleted = false; // Allow drawing again
    redraw();
});

function redraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    paths.forEach(path => {
        ctx.beginPath();
        path.forEach((point, index) => {
            if (index === 0) {
                ctx.moveTo(point.x, point.y);
            } else {
                ctx.lineTo(point.x, point.y);
            }
        });
        ctx.stroke();
    });
}

function drawBackground() {
    bgCtx.clearRect(0, 0, backgroundCanvas.width, backgroundCanvas.height);
    drawAxes(bgCtx);
    drawBox(bgCtx);
}

function drawAxes(ctx) {
    ctx.strokeStyle = 'gray';
    ctx.lineWidth = 1;

    // Draw x-axis
    ctx.beginPath();
    ctx.moveTo(0, backgroundCanvas.height / 2);
    ctx.lineTo(backgroundCanvas.width, backgroundCanvas.height / 2);
    ctx.stroke();

    // Draw y-axis
    ctx.beginPath();
    ctx.moveTo(backgroundCanvas.width / 2, 0);
    ctx.lineTo(backgroundCanvas.width / 2, backgroundCanvas.height);
    ctx.stroke();
}

function drawBox(ctx) {
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;

    // Draw box around the canvas
    ctx.beginPath();
    ctx.rect(0, 0, backgroundCanvas.width, backgroundCanvas.height);
    ctx.stroke();
}