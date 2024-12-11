console.log("script.js is loaded");

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
    ctx.lineWidth = 1;  // Thinner line width
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
    drawBox(bgCtx);
}

function drawBox(ctx) {
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;

    // Draw box around the canvas
    ctx.beginPath();
    ctx.rect(0, 0, backgroundCanvas.width, backgroundCanvas.height);
    ctx.stroke();
}

// Handle clear button click
document.getElementById('clearBtn').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    paths = [];
    drawingCompleted = false; // Allow drawing again
    redraw();
});

function getBase64Image() {
    // Create an off-screen canvas for processing
    const offScreenCanvas = document.createElement('canvas');
    offScreenCanvas.width = 250;  // Resize to 250x250
    offScreenCanvas.height = 250;
    const offScreenCtx = offScreenCanvas.getContext('2d');

    // Draw the current canvas content to the off-screen canvas
    offScreenCtx.drawImage(canvas, 0, 0, 250, 250);

    // Get the base64-encoded image data
    const base64Image = offScreenCanvas.toDataURL('image/png');
    console.log('Base64 image:', base64Image);  // Debugging statement
    return base64Image;
}

// Handle submit button click
document.getElementById('submitBtn').addEventListener('click', () => {
    const base64Image = getBase64Image();
    console.log('Base64 image length:', base64Image.length);  // Debugging statement

    // Send base64 image to the backend for prediction
    fetch('https://protos.ddns.net/predict', {  // Updated URL
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: base64Image }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Prediction response:', data);  // Debugging statement
        alert(`Predicted graph type: ${data.prediction}`);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});