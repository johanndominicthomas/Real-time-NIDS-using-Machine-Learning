// Dashboard state
let stats = {
    totalPackets: 0,
    totalAnomalies: 0,
    startTime: new Date(),
    trafficData: {
        normal: [],
        anomaly: []
    }
};

// Initialize Chart.js
const ctx = document.getElementById('trafficChart').getContext('2d');
const trafficChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Normal Traffic',
                data: [],
                backgroundColor: '#2ecc71',
                borderColor: '#2ecc71',
                borderWidth: 2,
                barThickness: 2,
            },
            {
                label: 'Anomalies',
                data: [],
                backgroundColor: '#e74c3c',
                borderColor: '#e74c3c',
                borderWidth: 2,
                barThickness: 2,
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                ticks: {
                    autoSkip: true,
                    maxTicksLimit: 10
                }
            },
            y: {
                beginAtZero: true
            }
        },
        animation: {
            duration: 0
        }
    }
});

// Function to reset the chart if an error occurs
function resetChart() {
    console.warn("Resetting chart due to error...");
    trafficChart.data.labels = [];
    trafficChart.data.datasets.forEach(dataset => dataset.data = []);
    trafficChart.update();
}

// WebSocket connection
let ws = null;

function connectWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);

    ws.onopen = function() {
        console.log('WebSocket connection established');
        document.getElementById('systemStatus').textContent = 'Active';
        document.getElementById('systemStatus').style.backgroundColor = '#2ecc71';
    };

    ws.onmessage = function(event) {
        if (ws.readyState === WebSocket.OPEN) {
            try {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            } catch (error) {
                console.error('Error processing WebSocket message:', error);
            }
        }
    };

    ws.onclose = function() {
        console.log('WebSocket connection closed, stopping updates...');
        document.getElementById('systemStatus').textContent = 'Disconnected';
        document.getElementById('systemStatus').style.backgroundColor = '#e74c3c';
        ws = null;
        resetChart();
    };

    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        document.getElementById('systemStatus').textContent = 'Error';
        document.getElementById('systemStatus').style.backgroundColor = '#e74c3c';
    };
}

// Establish WebSocket connection
connectWebSocket();

// Function to update the dashboard with data received from the WebSocket
function updateDashboard(data) {
    try {
        stats.totalPackets++;
        if (data.prediction_label === 'anomaly') {
            stats.totalAnomalies++;
        }

        // Update stats
        document.getElementById('totalPackets').textContent = stats.totalPackets;
        document.getElementById('totalAnomalies').textContent = stats.totalAnomalies;
        document.getElementById('detectionRate').textContent =
        `${((stats.totalAnomalies / stats.totalPackets) * 100).toFixed(2)}%`;

        const timestamp = new Date().toLocaleTimeString();
        trafficChart.data.labels.push(timestamp);

        // Update chart datasets
        trafficChart.data.datasets[0].data.push(data.prediction_label === 'normal' ? 1 : 0);
        trafficChart.data.datasets[1].data.push(data.prediction_label === 'anomaly' ? 1 : 0);

        // Keep only the last 50 data points
        if (trafficChart.data.labels.length > 50) {
            trafficChart.data.labels.shift();
            trafficChart.data.datasets.forEach(dataset => dataset.data.shift());
        }

        trafficChart.update();

        // Update the intrusion log
        const intrusionLog = document.getElementById("log-container");
        const logEntry = `
        <div class="alert">
        <span class="timestamp">${data.timestamp}</span>
        <span class="severity ${data.severity.toLowerCase()}">${data.severity}</span>
        <span class="intrusion-type">${data.intrusion_type}</span> <!-- Dynamically added intrusion type -->
        </div>`;
        intrusionLog.innerHTML += logEntry;

        // Scroll log to the bottom to show latest entries
        intrusionLog.scrollTop = intrusionLog.scrollHeight;

    } catch (error) {
        console.error("Chart update failed, resetting chart...", error);
        resetChart();
    }
}
