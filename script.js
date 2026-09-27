// --- NAVIGATION LOGIC ---
let isDashboardLoaded = false;

function openTab(evt, tabName) {
    const tabcontent = document.getElementsByClassName("tab-content");
    for (let i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    const tablinks = document.getElementsByClassName("tab-link");
    for (let i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";

    // Only load the heavy dashboard data once
    if (tabName === 'DashboardView' && !isDashboardLoaded) {
        loadDashboard();
        isDashboardLoaded = true;
    }
}

// --- UPLOAD TESTER LOGIC ---
async function compareImages() {
    const file1 = document.getElementById('image1').files[0];
    const file2 = document.getElementById('image2').files[0];
    const resultDiv = document.getElementById('result');

    if (!file1 || !file2) { alert("Please select both images to proceed."); return; }
    
    resultDiv.style.display = "block";
    resultDiv.className = "result-card"; 
    resultDiv.innerText = "Analyzing feature vectors...";

    const formData = new FormData();
    formData.append("image1", file1);
    formData.append("image2", file2);

    try {
        const response = await fetch("/compare", { method: "POST", body: formData });
        const data = await response.json();
        
        if(data.is_match) {
            resultDiv.className = "result-card match-success";
            resultDiv.innerText = `MATCH CONFIRMED\nEuclidean Distance: ${data.distance.toFixed(4)}`;
        } else {
            resultDiv.className = "result-card match-fail";
            resultDiv.innerText = `NO MATCH\nEuclidean Distance: ${data.distance.toFixed(4)}`;
        }
    } catch (error) { 
        resultDiv.innerText = "Connection error. Ensure the FastAPI backend is running."; 
    }
}

// --- DASHBOARD LOGIC ---
async function loadDashboard() {
    try {
        const response = await fetch("/dashboard-data");
        const data = await response.json();

        const trace = {
            x: data.x,
            y: data.y,
            mode: 'markers',
            type: 'scatter',
            marker: { 
                size: 8, 
                color: data.labels.map(l => stringToColor(l)), 
                opacity: 0.85,
                line: { width: 1, color: '#ffffff' }
            },
            text: data.labels, 
            customdata: data.images, 
            hoverinfo: 'text'
        };

        const layout = { 
            hovermode: 'closest', 
            margin: { l: 20, r: 20, b: 20, t: 20 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            xaxis: { showgrid: false, zeroline: false, showticklabels: false },
            yaxis: { showgrid: false, zeroline: false, showticklabels: false }
        };
        
        Plotly.newPlot('plot', [trace], layout, {displayModeBar: false});

        const plotDiv = document.getElementById('plot');
        plotDiv.on('plotly_hover', function(eventData) {
            const point = eventData.points[0];
            document.getElementById('hover-hint').style.display = 'none';
            document.getElementById('hover-image').src = point.customdata;
            document.getElementById('hover-image').style.display = 'block';
            document.getElementById('hover-label').innerText = point.text;
        });

    } catch (error) {
        console.error("Dashboard failed to load:", error);
        document.getElementById('plot').innerHTML = "<p style='text-align:center; margin-top:200px; color:#6b7280;'>Unable to load embeddings data.</p>";
    }
}

// Helper function to assign colors to the Plotly scatter points
function stringToColor(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) hash = str.charCodeAt(i) + ((hash << 5) - hash);
    return `hsl(${hash % 360}, 75%, 60%)`;
}
