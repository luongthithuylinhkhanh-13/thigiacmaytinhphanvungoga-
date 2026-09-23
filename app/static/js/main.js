document.addEventListener('DOMContentLoaded', () => {
    const uploadBtn = document.getElementById('uploadBtn');
    const imageInput = document.getElementById('imageInput');
    const segmentBtn = document.getElementById('segmentBtn');
    const clearBtn = document.getElementById('clearBtn');
    
    const imageContainer = document.getElementById('imageContainer');
    const emptyState = document.getElementById('emptyState');
    const displayImage = document.getElementById('displayImage');
    const thumbImage = document.getElementById('thumbImage');
    const thumbEmpty = document.getElementById('thumbEmpty');
    const infoFilename = document.getElementById('infoFilename');
    const infoSize = document.getElementById('infoSize');
    const loadingOverlay = document.getElementById('loadingOverlay');
    
    // Sliders
    const confSlider = document.getElementById('confSlider');
    const confValue = document.getElementById('confValue');
    const iouSlider = document.getElementById('iouSlider');
    const iouValue = document.getElementById('iouValue');

    if (confSlider && confValue) {
        confSlider.addEventListener('input', (e) => confValue.textContent = e.target.value);
    }
    if (iouSlider && iouValue) {
        iouSlider.addEventListener('input', (e) => iouValue.textContent = e.target.value);
    }
    
    // Status Flow UI
    const statusSteps = document.querySelectorAll('.status-step');
    
    // Chart Instance
    let potholeChartInstance = null;
    
    let selectedFile = null;

    function setStatus(stepIndex) {
        statusSteps.forEach((step, idx) => {
            if (idx <= stepIndex) step.classList.add('active');
            else step.classList.remove('active');
        });
    }
    
    // Init state
    setStatus(-1);

    uploadBtn.addEventListener('click', () => {
        imageInput.click();
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
            e.target.value = ''; // reset
        }
    });

    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function handleFileSelect(file) {
        if (!file.type.match('image.*')) {
            alert('Vui lòng chọn file ảnh!');
            return;
        }

        selectedFile = file;
        infoFilename.value = file.name;
        infoSize.value = formatBytes(file.size);

        const reader = new FileReader();
        reader.onload = (e) => {
            displayImage.src = e.target.result;
            thumbImage.src = e.target.result;
            
            emptyState.classList.add('hidden');
            displayImage.classList.remove('hidden');
            
            thumbEmpty.style.display = 'none';
            thumbImage.style.display = 'block';
            
            segmentBtn.disabled = false;
            segmentBtn.classList.add('btn-blue');
            segmentBtn.classList.remove('btn-gray');
            
            setStatus(0); // Loaded
        };
        reader.readAsDataURL(file);
    }

    segmentBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        loadingOverlay.classList.remove('hidden');
        segmentBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);
        if (confSlider) formData.append('conf_threshold', confSlider.value);
        if (iouSlider) formData.append('iou_threshold', iouSlider.value);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                alert('Lỗi: ' + (data.error || 'Server error'));
                return;
            }

            // Update Image to Segmented Result
            displayImage.src = data.result_image;
            
            // Update Quick Report
            document.getElementById('resCount').textContent = data.pothole_count;
            document.getElementById('resArea').textContent = data.area;
            document.getElementById('resConf').textContent = data.confidence > 0 ? data.confidence.toFixed(1) + '%' : '--';

            // Risk Prediction Logic (Mocked based on Area/Count since real model doesn't output risk explicitly)
            // But we will use real pothole count to populate the risk list
            const riskList = document.getElementById('riskList');
            riskList.innerHTML = ''; // clear
            
            if (data.pothole_count > 0) {
                // If model provides bounding boxes details, we could list them.
                // Since our backend currently only provides aggregate, we'll create a generic list item
                let riskLevel = 'Medium (Yellow)';
                let riskClass = 'medium';
                let markerPos = 50;
                
                if (data.pothole_count > 3 || (data.area !== "N/A" && parseInt(data.area) > 10000)) {
                    riskLevel = 'High (Red)';
                    riskClass = 'high';
                    markerPos = 90;
                } else if (data.pothole_count === 1) {
                    riskLevel = 'Low (Green)';
                    riskClass = 'low';
                    markerPos = 10;
                }

                riskList.innerHTML = `
                    <div class="risk-item ${riskClass}">
                        <span>Tổng hợp</span>
                        <span>Area: ${data.area}</span>
                        <span>Risk: ${riskLevel}</span>
                    </div>
                `;
                document.getElementById('riskMarker').style.left = markerPos + '%';
            } else {
                riskList.innerHTML = '<div class="risk-item empty-risk">Không phát hiện ổ gà</div>';
                document.getElementById('riskMarker').style.left = '0%';
            }

            // Render Chart
            const chartPlaceholderText = document.getElementById('chartPlaceholderText');
            const potholeChartCanvas = document.getElementById('potholeChart');
            const chartStats = document.getElementById('chartStats');
            
            if (potholeChartInstance) {
                potholeChartInstance.destroy();
            }

            if (data.pothole_count > 0 && data.potholes_data && data.potholes_data.length > 0) {
                chartPlaceholderText.style.display = 'none';
                potholeChartCanvas.style.display = 'block';
                chartStats.style.display = 'grid';

                const labels = data.potholes_data.map(p => p.id);
                const areas = data.potholes_data.map(p => p.area);

                // Calculate stats
                const totalArea = areas.reduce((a, b) => a + b, 0);
                const maxArea = Math.max(...areas);
                const avgArea = Math.round(totalArea / areas.length);

                document.getElementById('statCount').textContent = data.pothole_count;
                document.getElementById('statTotalArea').textContent = totalArea.toLocaleString('vi-VN');
                document.getElementById('statMaxArea').textContent = maxArea.toLocaleString('vi-VN');
                document.getElementById('statAvgArea').textContent = avgArea.toLocaleString('vi-VN');

                const ctx = potholeChartCanvas.getContext('2d');
                potholeChartInstance = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Diện tích (px²)',
                            data: areas,
                            backgroundColor: 'rgba(59, 130, 246, 0.7)',
                            borderColor: 'rgba(59, 130, 246, 1)',
                            borderWidth: 1,
                            borderRadius: 4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return 'Diện tích: ' + context.raw.toLocaleString('vi-VN') + ' px²';
                                    }
                                }
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: { display: true, text: 'Diện tích (px²)' }
                            }
                        }
                    }
                });
            } else {
                chartPlaceholderText.style.display = 'block';
                chartPlaceholderText.textContent = 'Không phát hiện ổ gà';
                potholeChartCanvas.style.display = 'none';
                chartStats.style.display = 'none';
            }

            setStatus(2); // Segmented & Reported

        } catch (error) {
            console.error('API Error:', error);
            alert('Lỗi kết nối đến máy chủ.');
        } finally {
            loadingOverlay.classList.add('hidden');
            segmentBtn.disabled = false;
        }
    });

    clearBtn.addEventListener('click', () => {
        selectedFile = null;
        imageInput.value = '';
        displayImage.src = '';
        thumbImage.src = '';
        
        emptyState.classList.remove('hidden');
        displayImage.classList.add('hidden');
        thumbEmpty.style.display = 'block';
        thumbImage.style.display = 'none';
        
        infoFilename.value = '--';
        infoSize.value = '--';
        
        document.getElementById('resCount').textContent = '--';
        document.getElementById('resArea').textContent = '--';
        document.getElementById('resConf').textContent = '--';
        
        document.getElementById('riskList').innerHTML = '<div class="risk-item empty-risk">Chưa có dữ liệu phân tích</div>';
        document.getElementById('riskMarker').style.left = '0%';
        
        // Reset Chart
        if (potholeChartInstance) {
            potholeChartInstance.destroy();
            potholeChartInstance = null;
        }
        document.getElementById('chartPlaceholderText').style.display = 'block';
        document.getElementById('chartPlaceholderText').textContent = 'Biểu đồ cập nhật thời gian thực (Real time updating chart)';
        document.getElementById('potholeChart').style.display = 'none';
        document.getElementById('chartStats').style.display = 'none';

        segmentBtn.disabled = true;
        segmentBtn.classList.remove('btn-blue');
        segmentBtn.classList.add('btn-gray');
        
        setStatus(-1);
    });
});
