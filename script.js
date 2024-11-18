const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;
let paths = [];
let currentPath = [];

function resizeCanvas() {
    canvas.width = window.innerWidth * 0.9;
    canvas.height = window.innerHeight * 0.6;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Mouse events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mousemove', draw);

// Touch events
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startDrawing(e.touches[0]);
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
    }
    ctx.beginPath();
}

function draw(e) {
    if (!drawing) return;
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    const x = e.clientX - canvas.offsetLeft;
    const y = e.clientY - canvas.offsetTop;
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
});

// Handle undo button click
document.getElementById('undoBtn').addEventListener('click', () => {
    if (paths.length > 0) {
        paths.pop();
        redraw();
    }
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