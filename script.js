const canvas = document.getElementById('graphCanvas');
const ctx = canvas.getContext('2d');
let drawing = false;

// Adjust canvas size to fit the screen
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
canvas.addEventListener('touchstart', (e) => startDrawing(e.touches[0]));
canvas.addEventListener('touchend', stopDrawing);
canvas.addEventListener('touchmove', (e) => draw(e.touches[0]));

function startDrawing(e) {
    drawing = true;
    draw(e);
}

function stopDrawing() {
    drawing = false;
    ctx.beginPath();
}

function draw(e) {
    if (!drawing) return;
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
}

// Handle submit button click
document.getElementById('submitBtn').addEventListener('click', () => {
    alert('Drawing submitted successfully!');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});