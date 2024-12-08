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

// Handle submit button click
document.getElementById('submitBtn').addEventListener('click', () => {
    const graphArray = getGraphArray();
    console.log('Graph array length:', graphArray.length); // Debugging statement
    console.log('Graph array sample:', graphArray.slice(0, 10)); // Debugging statement
    // Send graphArray to the backend for prediction
    fetch('http://localhost:5000/predict', {  // Update the URL to use localhost
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ graph: graphArray }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Prediction response:', data); // Debugging statement
        alert(`Predicted graph type: ${data.prediction}`);
    })
    .catch(error => {
        console.error('Error:', error);
    });

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

function getGraphArray() {
    // Create an off-screen canvas for processing
    const offScreenCanvas = document.createElement('canvas');
    offScreenCanvas.width = 200;  // Resize to 200x200
    offScreenCanvas.height = 200;
    const offScreenCtx = offScreenCanvas.getContext('2d');

    // Draw the current canvas content to the off-screen canvas
    offScreenCtx.drawImage(canvas, 0, 0, 200, 200);

    // Get the image data from the off-screen canvas
    const imageData = offScreenCtx.getImageData(0, 0, 200, 200);
    const data = imageData.data;

    // Convert the image data to a grayscale array
    const grayArray = [];
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        grayArray.push(gray);
    }

    // Apply Gaussian smoothing
    const smoothedArray = gaussianSmoothing(grayArray, 200, 200);

    // Normalize the pixel values
    const normalizedArray = normalize(smoothedArray);

    return normalizedArray;
}

function gaussianSmoothing(array, width, height) {
    const kernel = [
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ];
    const kernelSize = 5;
    const kernelSum = 273; // Sum of all kernel values

    const smoothedArray = new Array(array.length).fill(0);

    for (let y = 2; y < height - 2; y++) {
        for (let x = 2; x < width - 2; x++) {
            let sum = 0;
            for (let ky = 0; ky < kernelSize; ky++) {
                for (let kx = 0; kx < kernelSize; kx++) {
                    const pixel = array[(y + ky - 2) * width + (x + kx - 2)];
                    sum += pixel * kernel[ky][kx];
                }
            }
            smoothedArray[y * width + x] = sum / kernelSum;
        }
    }

    return smoothedArray;
}

function normalize(array) {
    const min = Math.min(...array);
    const max = Math.max(...array);
    return array.map(value => (value - min) / (max - min));
}