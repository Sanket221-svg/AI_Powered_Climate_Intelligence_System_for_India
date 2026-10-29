// Frontend JS
let charts = {};

async function loadOverviewCharts() {
    try {
        const res = await fetch('/api/charts/overview');
        const data = await res.json();
        const cityCtx = document.getElementById('cityAqiChart').getContext('2d');
        if (charts.cityAqi) charts.cityAqi.destroy();
        charts.cityAqi = new Chart(cityCtx, {
            type: 'bar',
            data: { labels: data.city_aqi_labels || [], datasets: [{ label: 'Average AQI', data: data.city_aqi_values || [], backgroundColor: 'rgba(13,110,253,0.7)' }] },
            options: { responsive: true, scales: { y: { beginAtZero: true } } }
        });
        const stateCtx = document.getElementById('stateRainChart').getContext('2d');
        if (charts.stateRain) charts.stateRain.destroy();
        charts.stateRain = new Chart(stateCtx, {
            type: 'bar',
            data: { labels: data.state_rain_labels || [], datasets: [{ label: 'Average Rainfall (mm)', data: data.state_rain_values || [], backgroundColor: 'rgba(25,135,84,0.7)' }] },
            options: { responsive: true, scales: { y: { beginAtZero: true } } }
        });
    } catch(e) { console.error(e); }
}

async function loadAnalyticsCharts() {
    try {
        const res = await fetch('/api/charts/overview');
        const data = await res.json();
        const monthCtx = document.getElementById('monthlyRainChart').getContext('2d');
        if (charts.monthlyRain) charts.monthlyRain.destroy();
        charts.monthlyRain = new Chart(monthCtx, {
            type: 'line',
            data: { labels: data.month_rain_labels || [], datasets: [{ label: 'Avg Rainfall (mm)', data: data.month_rain_values || [], borderColor: 'rgb(75,192,192)', tension: 0.1 }] },
            options: { responsive: true }
        });
        const seasonCtx = document.getElementById('seasonTempChart').getContext('2d');
        if (charts.seasonTemp) charts.seasonTemp.destroy();
        charts.seasonTemp = new Chart(seasonCtx, {
            type: 'bar',
            data: { labels: data.season_temp_labels || [], datasets: [{ label: 'Avg Temperature (°C)', data: data.season_temp_values || [], backgroundColor: ['#0d6efd','#198754','#ffc107','#dc3545'] }] },
            options: { responsive: true, scales: { y: { beginAtZero: true } } }
        });
    } catch(e) { console.error(e); }
}

async function loadClustering() {
    try {
        const res = await fetch('/api/clustering');
        const data = await res.json();
        if (data.error) {
            document.getElementById('clusterSummary').innerHTML = '<div class="alert alert-warning">' + data.error + '</div>';
            return;
        }
        let html = '<table class="table table-sm"><thead><tr><th>Cluster</th>';
        if (data.features) data.features.forEach(f => html += '<th>' + f + '</th>');
        html += '</tr></thead><tbody>';
        if (data.cluster_summary) {
            Object.keys(data.cluster_summary).sort((a,b)=>parseInt(a)-parseInt(b)).forEach(k => {
                html += '<tr><td>' + k + '</td>';
                data.features.forEach(f => {
                    html += '<td>' + (data.cluster_summary[k][f] !== undefined ? data.cluster_summary[k][f] : '-') + '</td>';
                });
                html += '</tr>';
            });
        }
        html += '</tbody></table>';
        document.getElementById('clusterSummary').innerHTML = html;
        const sample = data.sample || [];
        if (sample.length > 0 && data.features && data.features.length >= 2) {
            const ctx = document.getElementById('clusterScatterChart').getContext('2d');
            if (charts.clusterScatter) charts.clusterScatter.destroy();
            const colors = ['red','blue','green','orange','purple'];
            const datasets = {};
            sample.forEach(s => {
                const c = s.Cluster;
                if (!datasets[c]) datasets[c] = { label: 'Cluster ' + c, data: [], backgroundColor: colors[c % colors.length] };
                datasets[c].data.push({x: s[data.features[0]], y: s[data.features[1]]});
            });
            charts.clusterScatter = new Chart(ctx, { type: 'scatter', data: { datasets: Object.values(datasets) }, options: { responsive: true } });
        }
    } catch(e) { console.error(e); }
}

async function loadAnomalies() {
    try {
        const res = await fetch('/api/anomalies');
        const data = await res.json();
        if (data.error) {
            document.getElementById('anomalyCount').innerHTML = '<div class="alert alert-warning">' + data.error + '</div>';
            return;
        }
        document.getElementById('anomalyCount').innerHTML = '<strong>' + data.count + '</strong> anomalies detected (statistical outliers)';
        const anomalies = data.anomalies || [];
        if (anomalies.length > 0) {
            let html = '<thead><tr><th>Date</th><th>City</th><th>State</th><th>Temp</th><th>Humidity</th><th>AQI</th></tr></thead><tbody>';
            anomalies.slice(0, 15).forEach(a => {
                html += '<tr><td>' + (a.Date||'-') + '</td><td>' + (a.City||'-') + '</td><td>' + (a.State||'-') + '</td><td>' + (a['Temperature_Avg (°C)']||'-') + '</td><td>' + (a['Humidity (%)']||'-') + '</td><td>' + (a.AQI||'-') + '</td></tr>';
            });
            html += '</tbody>';
            document.getElementById('anomalyTable').innerHTML = html;
        } else {
            document.getElementById('anomalyTable').innerHTML = '';
        }
    } catch(e) { console.error(e); }
}

async function loadModelInfo() {
    try {
        const res = await fetch('/api/model/info');
        const data = await res.json();
        document.getElementById('modelInfo').textContent = JSON.stringify(data, null, 2);
    } catch(e) {
        document.getElementById('modelInfo').textContent = 'Error loading model info';
    }
}

document.getElementById('tempForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
        'Humidity (%)': parseFloat(document.getElementById('temp_humidity').value),
        'Rainfall (mm)': parseFloat(document.getElementById('temp_rainfall').value),
        'Wind_Speed (km/h)': parseFloat(document.getElementById('temp_wind').value),
        'Pressure (hPa)': parseFloat(document.getElementById('temp_pressure').value),
        'AQI': parseFloat(document.getElementById('temp_aqi').value),
        'Month': parseInt(document.getElementById('temp_month').value)
    };
    try {
        const res = await fetch('/api/predict/temperature', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
        const data = await res.json();
        if (data.error) {
            document.getElementById('tempResult').innerHTML = '<div class="alert alert-danger">' + data.error + '</div>';
        } else {
            let html = '<div class="alert alert-success"><strong>Predicted Average Temperature:</strong> ' + data.prediction.toFixed(2) + ' ' + data.unit + ' <br><small>' + data.note + '</small></div>';
            if (data.metrics) {
                html += '<div class="alert alert-info"><small>MAE: ' + data.metrics.mae.toFixed(3) + ', RMSE: ' + data.metrics.rmse.toFixed(3) + ', R²: ' + data.metrics.r2.toFixed(3) + '</small></div>';
            }
            document.getElementById('tempResult').innerHTML = html;
        }
    } catch(e) {
        document.getElementById('tempResult').innerHTML = '<div class="alert alert-danger">Error</div>';
    }
});

document.getElementById('aqiForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
        'Temperature_Avg (°C)': parseFloat(document.getElementById('aqi_temp').value),
        'Humidity (%)': parseFloat(document.getElementById('aqi_humidity').value),
        'Rainfall (mm)': parseFloat(document.getElementById('aqi_rainfall').value),
        'Wind_Speed (km/h)': parseFloat(document.getElementById('aqi_wind').value),
        'Pressure (hPa)': parseFloat(document.getElementById('aqi_pressure').value),
        'AQI': parseFloat(document.getElementById('aqi_val').value),
        'Month': parseInt(document.getElementById('aqi_month').value)
    };
    try {
        const res = await fetch('/api/predict/aqi', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
        const data = await res.json();
        if (data.error) {
            document.getElementById('aqiResult').innerHTML = '<div class="alert alert-danger">' + data.error + '</div>';
        } else {
            let html = '<div class="alert alert-success"><strong>Predicted AQI Category:</strong> ' + data.prediction + ' <br><small>' + data.note + '</small></div>';
            if (data.metrics) {
                html += '<div class="alert alert-info"><small>Acc: ' + data.metrics.accuracy.toFixed(3) + ', F1: ' + data.metrics.f1_weighted.toFixed(3) + '</small></div>';
            }
            document.getElementById('aqiResult').innerHTML = html;
        }
    } catch(e) {
        document.getElementById('aqiResult').innerHTML = '<div class="alert alert-danger">Error</div>';
    }
});

document.addEventListener('DOMContentLoaded', () => {
    loadOverviewCharts();
    loadAnalyticsCharts();
    loadClustering();
    loadAnomalies();
    loadModelInfo();
});
