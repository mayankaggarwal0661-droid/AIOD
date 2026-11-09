/**
 * =====================================================================
 * AIOD ENTERPRISE APPLICATION LOGIC & UI ORCHESTRATION
 * High-performance state management, synchronized inspection overlays,
 * split image comparison slider, Chart.js analytics, and safe history.
 * =====================================================================
 */

(function () {
    'use strict';

    // Application State
    const state = {
        currentSection: 'overview',
        currentTheme: localStorage.getItem('aiod_theme') || 'obsidian',
        singleFile: null,
        batchFiles: [],
        lastDetection: null,
        lastAnalytics: null,
        lastHistory: null,
        selectedBoxIndex: null,
        viewMode: 'inspector', // 'inspector' or 'comparison'
        historyData: [],
        charts: {}
    };

    // =====================================================================
    // INITIALIZATION & EVENT BINDINGS
    // =====================================================================
    document.addEventListener('DOMContentLoaded', () => {
        initThemeSelector();
        initNavigation();
        initDetectionControls();
        initComparisonSlider();
        initBatchControls();
        initHistorySearchAndExport();
        initRealtimeClock();

        // Initialize Three.js scene (contained within Overview)
        if (window.ThreeSceneManager) {
            window.ThreeSceneManager.init('three-bg-container');
        }

        // Preload recent history count for overview telemetry
        fetchOverviewMetrics();
    });

    // =====================================================================
    // THEME SELECTOR & PERSISTENCE
    // =====================================================================
    function initThemeSelector() {
        // Apply saved theme immediately
        setTheme(state.currentTheme, false);

        // Bind all theme buttons across header and settings
        document.querySelectorAll('.theme-pill-btn[data-theme]').forEach(btn => {
            btn.addEventListener('click', () => {
                const selected = btn.getAttribute('data-theme');
                setTheme(selected, true);
            });
        });
    }

    function setTheme(themeName, persist = true) {
        state.currentTheme = themeName;
        document.documentElement.setAttribute('data-theme', themeName);

        if (persist) {
            try {
                localStorage.setItem('aiod_theme', themeName);
            } catch (e) {
                // Ignore storage errors in restricted contexts
            }
        }

        // Update active class on all selector buttons
        document.querySelectorAll('.theme-pill-btn[data-theme]').forEach(btn => {
            if (btn.getAttribute('data-theme') === themeName) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Notify Three.js visualizer of theme palette change
        if (window.ThreeSceneManager && window.ThreeSceneManager.setTheme) {
            window.ThreeSceneManager.setTheme(themeName);
        }

        // Re-render Chart.js if analytics data has been fetched
        if (state.lastAnalytics && state.lastHistory) {
            renderAnalyticsCharts(state.lastAnalytics, state.lastHistory);
        }
    }

    // =====================================================================
    // NAVIGATION & SECTIONS
    // =====================================================================
    function initNavigation() {
        const navLinks = document.querySelectorAll('.nav-link[data-section]');
        const workspaces = document.querySelectorAll('.workspace');
        const headerTitle = document.getElementById('current-section-title');
        const sidebar = document.getElementById('sidebar');
        const sidebarToggle = document.getElementById('sidebar-toggle');
        const appShell = document.querySelector('.app-shell');

        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = link.getAttribute('data-section');
                switchSection(section);
            });
        });

        // Sidebar collapse toggle
        if (sidebarToggle) {
            sidebarToggle.addEventListener('click', () => {
                sidebar.classList.toggle('collapsed');
                appShell.classList.toggle('sidebar-collapsed');
            });
        }

        // Mobile menu toggle
        const mobileMenuBtn = document.getElementById('mobile-menu-btn');
        if (mobileMenuBtn) {
            mobileMenuBtn.addEventListener('click', () => {
                sidebar.classList.toggle('mobile-open');
            });
        }
    }

    function switchSection(sectionName) {
        state.currentSection = sectionName;

        // Update nav active classes
        document.querySelectorAll('.nav-link[data-section]').forEach(link => {
            if (link.getAttribute('data-section') === sectionName) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });

        // Update workspace visibility
        document.querySelectorAll('.workspace').forEach(ws => {
            if (ws.id === `workspace-${sectionName}`) {
                ws.classList.add('active');
            } else {
                ws.classList.remove('active');
            }
        });

        // Update top breadcrumb
        const titleMap = {
            overview: 'Platform Overview',
            single: 'Single Image Detection',
            batch: 'Batch Processing Hub',
            analytics: 'Vision Analytics & Insights',
            history: 'Detection Audit History',
            settings: 'System & Architecture Settings'
        };
        const headerTitle = document.getElementById('current-section-title');
        if (headerTitle) headerTitle.innerText = titleMap[sectionName] || 'Dashboard';

        // Notify Three.js to pause/resume
        if (window.ThreeSceneManager) {
            window.ThreeSceneManager.setSection(sectionName);
        }

        // Trigger on-demand data loads
        if (sectionName === 'analytics') {
            loadAnalyticsAndTrends();
        } else if (sectionName === 'history') {
            loadHistoryTable();
        } else if (sectionName === 'overview') {
            fetchOverviewMetrics();
        }
    }

    // =====================================================================
    // SINGLE DETECTION WORKSPACE
    // =====================================================================
    function initDetectionControls() {
        const dropZone = document.getElementById('single-drop-zone');
        const fileInput = document.getElementById('single-file-input');
        const confSlider = document.getElementById('conf-slider');
        const confBadge = document.getElementById('conf-badge');
        const detectBtn = document.getElementById('detect-btn');
        const modeInspectorBtn = document.getElementById('mode-inspector-btn');
        const modeComparisonBtn = document.getElementById('mode-comparison-btn');

        if (confSlider && confBadge) {
            confSlider.addEventListener('input', (e) => {
                confBadge.innerText = parseFloat(e.target.value).toFixed(2);
            });
        }

        if (dropZone && fileInput) {
            dropZone.addEventListener('click', () => fileInput.click());

            ['dragenter', 'dragover'].forEach(name => {
                dropZone.addEventListener(name, (e) => {
                    e.preventDefault();
                    dropZone.classList.add('dragover');
                }, false);
            });

            ['dragleave', 'drop'].forEach(name => {
                dropZone.addEventListener(name, (e) => {
                    e.preventDefault();
                    dropZone.classList.remove('dragover');
                }, false);
            });

            dropZone.addEventListener('drop', (e) => {
                if (e.dataTransfer.files.length > 0) {
                    handleSingleFile(e.dataTransfer.files[0]);
                }
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleSingleFile(e.target.files[0]);
                }
            });
        }

        if (detectBtn) {
            detectBtn.addEventListener('click', runSingleDetection);
        }

        // View Mode Switcher: Inspector vs Comparison
        if (modeInspectorBtn && modeComparisonBtn) {
            modeInspectorBtn.addEventListener('click', () => setViewMode('inspector'));
            modeComparisonBtn.addEventListener('click', () => setViewMode('comparison'));
        }

        // Class Filter Quick Chips
        document.querySelectorAll('.filter-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const input = document.getElementById('classes-filter');
                if (!input) return;
                const cls = chip.getAttribute('data-class');
                const current = input.value.split(',').map(s => s.trim()).filter(Boolean);
                if (!current.includes(cls)) {
                    current.push(cls);
                    input.value = current.join(', ');
                }
            });
        });

        // Gemini AI Button
        const geminiBtn = document.getElementById('gemini-btn');
        if (geminiBtn) {
            geminiBtn.addEventListener('click', runGeminiAnalysis);
        }
    }

    function handleSingleFile(file) {
        state.singleFile = file;
        const filenameText = document.getElementById('single-drop-filename');
        const detectBtn = document.getElementById('detect-btn');
        const placeholder = document.getElementById('preview-placeholder');
        const singleRawImg = document.getElementById('single-raw-img');
        const comparisonRawImg = document.getElementById('comparison-raw-img');

        if (filenameText) filenameText.innerText = file.name;
        if (detectBtn) detectBtn.disabled = false;

        const reader = new FileReader();
        reader.onload = (e) => {
            const dataUrl = e.target.result;
            if (singleRawImg) {
                singleRawImg.src = dataUrl;
                singleRawImg.style.display = 'block';
            }
            if (comparisonRawImg) {
                comparisonRawImg.src = dataUrl;
            }
            if (placeholder) placeholder.style.display = 'none';

            // Clear previous overlays
            clearBboxOverlay();
            resetComparisonSlider();
        };
        reader.readAsDataURL(file);
    }

    function setViewMode(mode) {
        state.viewMode = mode;
        const inspectorView = document.getElementById('inspector-view');
        const comparisonContainer = document.getElementById('comparison-container');
        const modeInspectorBtn = document.getElementById('mode-inspector-btn');
        const modeComparisonBtn = document.getElementById('mode-comparison-btn');

        if (mode === 'inspector') {
            if (inspectorView) inspectorView.style.display = 'block';
            if (comparisonContainer) comparisonContainer.classList.remove('active');
            if (modeInspectorBtn) modeInspectorBtn.classList.add('active');
            if (modeComparisonBtn) modeComparisonBtn.classList.remove('active');
        } else {
            if (inspectorView) inspectorView.style.display = 'none';
            if (comparisonContainer) comparisonContainer.classList.add('active');
            if (modeInspectorBtn) modeInspectorBtn.classList.remove('active');
            if (modeComparisonBtn) modeComparisonBtn.classList.add('active');
            syncComparisonDimensions();
        }
    }

    async function runSingleDetection() {
        if (!state.singleFile) return;

        const detectBtn = document.getElementById('detect-btn');
        const detectBtnText = document.getElementById('detect-btn-text');
        const scanner = document.getElementById('holographic-scanner');
        const confVal = document.getElementById('conf-slider').value;
        const classesVal = document.getElementById('classes-filter').value;

        // UI Loading State
        detectBtn.disabled = true;
        detectBtnText.innerHTML = '<span class="status-dot blue"></span> Analyzing with YOLO26n...';
        if (scanner) scanner.classList.add('scanning');

        const formData = new FormData();
        formData.append('image', state.singleFile);
        formData.append('confidence', confVal);
        formData.append('classes', classesVal);

        try {
            const response = await fetch('/api/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                alert('Detection Error: ' + (data.error || 'Request failed.'));
                return;
            }

            state.lastDetection = data;
            renderDetectionResults(data);

        } catch (err) {
            console.error('[AIOD] Detection error:', err);
            alert('Detection request failed: ' + err.message);
        } finally {
            detectBtn.disabled = false;
            detectBtnText.innerText = 'Run YOLO Detection';
            if (scanner) scanner.classList.remove('scanning');
        }
    }

    function renderDetectionResults(data) {
        const singleAnnotatedImg = document.getElementById('single-annotated-img');
        const comparisonAnnotatedImg = document.getElementById('comparison-annotated-img');
        const statsRow = document.getElementById('detection-stats-row');
        const timingPill = document.getElementById('timing-pill');
        const dbStatusBadge = document.getElementById('db-status-badge');
        const modeSwitcher = document.getElementById('mode-switcher-container');

        // Cache-busted image URLs
        const cacheBuster = `?t=${Date.now()}`;
        const annotatedUrl = data.annotated_image_url + cacheBuster;

        if (singleAnnotatedImg) {
            singleAnnotatedImg.src = annotatedUrl;
            singleAnnotatedImg.style.display = 'block';
            singleAnnotatedImg.onload = () => {
                renderInteractiveBboxes(data);
                syncComparisonDimensions();
            };
        }

        if (comparisonAnnotatedImg) {
            comparisonAnnotatedImg.src = annotatedUrl;
        }

        // Show mode switcher
        if (modeSwitcher) modeSwitcher.style.display = 'flex';

        // Update latency metrics
        if (timingPill) {
            timingPill.innerText = `${Math.round(data.inference_time_ms)} ms`;
            timingPill.title = `Preprocess: ${data.preprocess_time_ms}ms | Inference: ${data.inference_time_ms}ms | Postprocess: ${data.postprocess_time_ms}ms`;
        }

        // Update DB status
        if (dbStatusBadge) {
            dbStatusBadge.innerText = data.saved_to_db ? `DB: ${data.db_backend.toUpperCase()}` : 'DB: Offline';
            dbStatusBadge.className = data.saved_to_db ? 'status-pill' : 'status-pill warning';
        }

        // Summary Statistics Counters
        if (statsRow) statsRow.style.display = 'grid';
        animateCounter('stat-total-objects', data.total_count);
        animateCounter('stat-unique-classes', Object.keys(data.counts).length);
        animateCounter('stat-inference-time', Math.round(data.inference_time_ms), ' ms');

        // Class breakdown chips
        const pillsContainer = document.getElementById('detection-pills');
        if (pillsContainer) {
            pillsContainer.innerHTML = '';
            for (const [cls, count] of Object.entries(data.counts)) {
                pillsContainer.innerHTML += `<span class="pill-chip">${count}x ${cls}</span>`;
            }
        }

        // Synchronized Coordinates Table
        renderDetectionTable(data.detections);

        // Reset Gemini drawer
        const geminiOutput = document.getElementById('gemini-output');
        if (geminiOutput) {
            geminiOutput.innerText = 'Click "Generate AI Analysis" to get deep multimodal visual intelligence from Google Gemini.';
        }
        const geminiCard = document.getElementById('gemini-card');
        if (geminiCard) geminiCard.style.display = 'block';
    }

    // =====================================================================
    // SYNCHRONIZED BOUNDING BOX INSPECTOR
    // =====================================================================
    function renderInteractiveBboxes(data) {
        const overlay = document.getElementById('bbox-overlay');
        const img = document.getElementById('single-annotated-img');
        if (!overlay || !img || !data.detections) return;

        overlay.innerHTML = '';

        // Calculate scale ratios between rendered image and natural dimensions
        const naturalW = img.naturalWidth || 640;
        const naturalH = img.naturalHeight || 640;
        const renderedW = img.clientWidth;
        const renderedH = img.clientHeight;

        const scaleX = renderedW / naturalW;
        const scaleY = renderedH / naturalH;

        data.detections.forEach((box, index) => {
            const bboxEl = document.createElement('div');
            bboxEl.className = 'interactive-bbox';
            bboxEl.setAttribute('data-index', index);

            const left = box.x1 * scaleX;
            const top = box.y1 * scaleY;
            const width = box.width * scaleX;
            const height = box.height * scaleY;

            bboxEl.style.left = `${left}px`;
            bboxEl.style.top = `${top}px`;
            bboxEl.style.width = `${width}px`;
            bboxEl.style.height = `${height}px`;

            // Tag tooltip
            const tagEl = document.createElement('div');
            tagEl.className = 'bbox-tag';
            tagEl.innerText = `${box.class_name} ${(box.confidence * 100).toFixed(0)}%`;
            bboxEl.appendChild(tagEl);

            // Synchronized hover & click with Table
            bboxEl.addEventListener('mouseenter', () => highlightBoxAndRow(index));
            bboxEl.addEventListener('mouseleave', () => unhighlightBoxAndRow(index));
            bboxEl.addEventListener('click', () => selectBoxAndRow(index));

            overlay.appendChild(bboxEl);
        });
    }

    function clearBboxOverlay() {
        const overlay = document.getElementById('bbox-overlay');
        if (overlay) overlay.innerHTML = '';
    }

    function renderDetectionTable(detections) {
        const tbody = document.getElementById('detection-table-tbody');
        const container = document.getElementById('detection-table-container');
        if (!tbody || !container) return;

        tbody.innerHTML = '';

        if (!detections || detections.length === 0) {
            container.style.display = 'none';
            return;
        }

        container.style.display = 'block';

        detections.forEach((box, index) => {
            const tr = document.createElement('tr');
            tr.className = 'table-row';
            tr.setAttribute('data-index', index);
            tr.innerHTML = `
                <td><b>${box.class_name}</b></td>
                <td><span class="mono-font">${(box.confidence * 100).toFixed(1)}%</span></td>
                <td><span class="mono-font">[${box.bbox.join(', ')}]</span></td>
                <td><span class="mono-font">${Math.round(box.width)} × ${Math.round(box.height)} px</span></td>
                <td><span class="mono-font">${Math.round(box.area).toLocaleString()}</span></td>
            `;

            tr.addEventListener('mouseenter', () => highlightBoxAndRow(index));
            tr.addEventListener('mouseleave', () => unhighlightBoxAndRow(index));
            tr.addEventListener('click', () => selectBoxAndRow(index));

            tbody.appendChild(tr);
        });
    }

    function highlightBoxAndRow(index) {
        const bbox = document.querySelector(`.interactive-bbox[data-index="${index}"]`);
        const row = document.querySelector(`.table-row[data-index="${index}"]`);
        if (bbox) bbox.classList.add('highlighted');
        if (row) row.classList.add('row-highlighted');
    }

    function unhighlightBoxAndRow(index) {
        if (state.selectedBoxIndex === index) return;
        const bbox = document.querySelector(`.interactive-bbox[data-index="${index}"]`);
        const row = document.querySelector(`.table-row[data-index="${index}"]`);
        if (bbox) bbox.classList.remove('highlighted');
        if (row) row.classList.remove('row-highlighted');
    }

    function selectBoxAndRow(index) {
        // Clear previous selected
        document.querySelectorAll('.interactive-bbox').forEach(el => el.classList.remove('highlighted'));
        document.querySelectorAll('.table-row').forEach(el => el.classList.remove('row-highlighted'));

        state.selectedBoxIndex = index;
        highlightBoxAndRow(index);

        // Auto-scroll table row into view
        const row = document.querySelector(`.table-row[data-index="${index}"]`);
        if (row) {
            row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    // =====================================================================
    // BEFORE / AFTER SPLIT COMPARISON SLIDER
    // =====================================================================
    function initComparisonSlider() {
        const container = document.getElementById('comparison-wrapper');
        const topLayer = document.getElementById('comparison-top-layer');
        const handle = document.getElementById('comparison-handle');
        if (!container || !topLayer || !handle) return;

        let isDragging = false;

        function updateSliderPosition(clientX) {
            const rect = container.getBoundingClientRect();
            let offsetX = clientX - rect.left;
            offsetX = Math.max(0, Math.min(offsetX, rect.width));
            const percentage = (offsetX / rect.width) * 100;

            topLayer.style.width = `${percentage}%`;
            handle.style.left = `${percentage}%`;
        }

        handle.addEventListener('mousedown', () => { isDragging = true; });
        window.addEventListener('mouseup', () => { isDragging = false; });
        window.addEventListener('mousemove', (e) => {
            if (isDragging) updateSliderPosition(e.clientX);
        });

        // Touch support
        handle.addEventListener('touchstart', () => { isDragging = true; }, { passive: true });
        window.addEventListener('touchend', () => { isDragging = false; });
        window.addEventListener('touchmove', (e) => {
            if (isDragging && e.touches[0]) updateSliderPosition(e.touches[0].clientX);
        }, { passive: true });
    }

    function syncComparisonDimensions() {
        const rawImg = document.getElementById('comparison-raw-img');
        const topImg = document.querySelector('#comparison-top-layer img');
        if (rawImg && topImg) {
            topImg.style.width = `${rawImg.clientWidth}px`;
            topImg.style.height = `${rawImg.clientHeight}px`;
        }
    }

    function resetComparisonSlider() {
        const topLayer = document.getElementById('comparison-top-layer');
        const handle = document.getElementById('comparison-handle');
        if (topLayer) topLayer.style.width = '50%';
        if (handle) handle.style.left = '50%';
    }

    // =====================================================================
    // BATCH PROCESSING (Honest Indeterminate Loading)
    // =====================================================================
    function initBatchControls() {
        const dropZone = document.getElementById('batch-drop-zone');
        const fileInput = document.getElementById('batch-files-input');
        const batchBtn = document.getElementById('batch-start-btn');
        const confSlider = document.getElementById('batch-conf-slider');
        const confBadge = document.getElementById('batch-conf-badge');

        if (confSlider && confBadge) {
            confSlider.addEventListener('input', (e) => {
                confBadge.innerText = parseFloat(e.target.value).toFixed(2);
            });
        }

        if (dropZone && fileInput) {
            dropZone.addEventListener('click', () => fileInput.click());

            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            });
            dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
                if (e.dataTransfer.files.length > 0) {
                    handleBatchFiles(Array.from(e.dataTransfer.files));
                }
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleBatchFiles(Array.from(e.target.files));
                }
            });
        }

        if (batchBtn) {
            batchBtn.addEventListener('click', runBatchDetection);
        }
    }

    function handleBatchFiles(files) {
        state.batchFiles = files;
        const text = document.getElementById('batch-drop-text');
        const startBtn = document.getElementById('batch-start-btn');

        if (text) text.innerText = `${files.length} images queued for processing`;
        if (startBtn) startBtn.disabled = files.length === 0;
    }

    async function runBatchDetection() {
        if (!state.batchFiles || state.batchFiles.length === 0) return;

        const startBtn = document.getElementById('batch-start-btn');
        const btnText = document.getElementById('batch-btn-text');
        const progressBar = document.getElementById('batch-indeterminate-bar');
        const statusMsg = document.getElementById('batch-status-message');
        const summaryCard = document.getElementById('batch-summary-card');
        const gallery = document.getElementById('batch-gallery');

        startBtn.disabled = true;
        btnText.innerText = 'Processing Batch Queue...';
        if (progressBar) progressBar.classList.add('active');
        if (statusMsg) {
            statusMsg.style.display = 'block';
            statusMsg.innerText = `Sequential inference on ${state.batchFiles.length} images. Please wait...`;
        }

        const formData = new FormData();
        state.batchFiles.forEach(file => {
            formData.append('images', file);
        });
        formData.append('confidence', document.getElementById('batch-conf-slider').value);

        try {
            const response = await fetch('/api/batch', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                alert('Batch Error: ' + (data.error || 'Failed to process batch'));
                return;
            }

            // Render Exact Summary Stats
            if (summaryCard) summaryCard.style.display = 'block';
            animateCounter('batch-stat-processed', data.total_successful, `/${data.total_discovered}`);
            animateCounter('batch-stat-failed', data.total_failed);
            animateCounter('batch-stat-objects', data.total_objects_detected);
            animateCounter('batch-stat-avg-time', Math.round(data.average_time_per_image_ms), ' ms');

            // Render Aggregated Class Chips
            const pillsDiv = document.getElementById('batch-class-pills');
            if (pillsDiv) {
                pillsDiv.innerHTML = '';
                for (const [cls, count] of Object.entries(data.aggregated_class_counts || {})) {
                    pillsDiv.innerHTML += `<span class="pill-chip">${count}x ${cls}</span>`;
                }
            }

            // Render Result Gallery
            if (gallery) {
                gallery.innerHTML = '';
                data.items.forEach(item => {
                    const isSuccess = item.status === 'success';
                    const imgHtml = isSuccess && item.annotated_output_url
                        ? `<img src="${item.annotated_output_url}" alt="${item.image_name}" loading="lazy">`
                        : `<div class="empty-state" style="padding: 30px;"><div style="color: var(--accent-rose); font-size: 24px;">⚠️</div></div>`;

                    gallery.innerHTML += `
                        <div class="gallery-card">
                            ${imgHtml}
                            <div class="gallery-card-body">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                    <span style="font-weight: 700; font-size: 13px; text-overflow: ellipsis; overflow: hidden; white-space: nowrap; max-width: 140px;">${item.image_name}</span>
                                    <span class="status-pill ${isSuccess ? '' : 'warning'}" style="font-size: 11px; padding: 2px 8px;">${item.status}</span>
                                </div>
                                <div style="font-size: 12px; color: var(--text-secondary);">
                                    ${isSuccess ? `${item.total_objects} objects found (${Math.round(item.processing_time_ms)} ms)` : (item.error_message || 'Processing failed')}
                                </div>
                            </div>
                        </div>
                    `;
                });
            }

        } catch (err) {
            console.error('[AIOD] Batch error:', err);
            alert('Batch run failed: ' + err.message);
        } finally {
            startBtn.disabled = false;
            btnText.innerText = 'Start Batch Detection';
            if (progressBar) progressBar.classList.remove('active');
            if (statusMsg) statusMsg.style.display = 'none';
        }
    }

    // =====================================================================
    // VISION ANALYTICS & RECENT TRENDS CHARTS
    // =====================================================================
    async function loadAnalyticsAndTrends() {
        try {
            const [analyticsRes, historyRes] = await Promise.all([
                fetch('/api/analytics'),
                fetch('/api/history')
            ]);

            const analytics = await analyticsRes.json();
            const historyPayload = await historyRes.json();
            const history = historyPayload.history || [];

            state.lastAnalytics = analytics;
            state.lastHistory = history;

            // Update Lifetime Numbers
            animateCounter('analytics-total-images', analytics.total_images_processed || 0);
            animateCounter('analytics-total-objects', analytics.total_objects_detected || 0);
            animateCounter('analytics-avg-conf', Math.round((analytics.overall_average_confidence || 0) * 100), '%');

            // Render Charts via Chart.js
            renderAnalyticsCharts(analytics, history);

        } catch (err) {
            console.error('[AIOD] Analytics loading error:', err);
        }
    }

    function renderAnalyticsCharts(analytics, history) {
        if (typeof Chart === 'undefined') {
            console.warn('[AIOD] Chart.js not loaded. Skipping chart rendering.');
            return;
        }

        const isNoirGold = document.documentElement.getAttribute('data-theme') === 'noir-gold';

        // Chart 1: Lifetime Class Distribution (Doughnut)
        const donutCtx = document.getElementById('chart-class-distribution');
        if (donutCtx) {
            if (state.charts.distribution) state.charts.distribution.destroy();
            const labels = Object.keys(analytics.class_distribution || {});
            const values = Object.values(analytics.class_distribution || {});

            const donutPalette = isNoirGold
                ? ['#D4AF37', '#111111', '#E6C65A', '#2D2D2D', '#B8962E', '#4B4B4B', '#C9A028', '#8E8E8E']
                : ['#3B82F6', '#8B5CF6', '#06B6D4', '#10B981', '#F59E0B', '#EC4899', '#6366F1', '#14B8A6'];

            state.charts.distribution = new Chart(donutCtx, {
                type: 'doughnut',
                data: {
                    labels: labels.slice(0, 8),
                    datasets: [{
                        data: values.slice(0, 8),
                        backgroundColor: donutPalette,
                        borderWidth: 2,
                        borderColor: isNoirGold ? '#FFFFFF' : '#080A10'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'right', labels: { color: isNoirGold ? '#171717' : '#94A3B8', font: { family: 'Plus Jakarta Sans' } } }
                    }
                }
            });
        }

        // Chart 2: Top Detected Classes (Horizontal Bar)
        const barCtx = document.getElementById('chart-top-classes');
        if (barCtx) {
            if (state.charts.topClasses) state.charts.topClasses.destroy();
            const topPairs = (analytics.top_detected_classes || []).slice(0, 6);

            state.charts.topClasses = new Chart(barCtx, {
                type: 'bar',
                data: {
                    labels: topPairs.map(p => p[0]),
                    datasets: [{
                        label: 'Detections Count',
                        data: topPairs.map(p => p[1]),
                        backgroundColor: isNoirGold ? 'rgba(212, 175, 55, 0.88)' : 'rgba(59, 130, 246, 0.75)',
                        borderColor: isNoirGold ? '#D4AF37' : '#3B82F6',
                        borderWidth: 1,
                        borderRadius: 4
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: isNoirGold ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)' }, ticks: { color: isNoirGold ? '#707070' : '#94A3B8' } },
                        y: { grid: { display: false }, ticks: { color: isNoirGold ? '#171717' : '#F8FAFC' } }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }

        // Chart 3: Recent Inference Latency Profile (Line Chart from History Latest 50)
        const latencyCtx = document.getElementById('chart-recent-latencies');
        if (latencyCtx) {
            if (state.charts.latency) state.charts.latency.destroy();
            const reversedHistory = [...history].reverse();

            state.charts.latency = new Chart(latencyCtx, {
                type: 'line',
                data: {
                    labels: reversedHistory.map((_, i) => `#${i + 1}`),
                    datasets: [{
                        label: 'Inference Latency (ms)',
                        data: reversedHistory.map(h => Math.round(h.inference_time_ms || 0)),
                        borderColor: isNoirGold ? '#D4AF37' : '#06B6D4',
                        backgroundColor: isNoirGold ? 'rgba(212, 175, 55, 0.12)' : 'rgba(6, 182, 212, 0.1)',
                        fill: true,
                        tension: 0.35,
                        pointRadius: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: isNoirGold ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)' }, ticks: { color: isNoirGold ? '#707070' : '#94A3B8' } },
                        y: { grid: { color: isNoirGold ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)' }, ticks: { color: isNoirGold ? '#707070' : '#94A3B8' } }
                    },
                    plugins: {
                        legend: { labels: { color: isNoirGold ? '#171717' : '#94A3B8' } }
                    }
                }
            });
        }
    }

    // =====================================================================
    // HISTORY TABLE & SAFE URL RESOLUTION
    // =====================================================================
    async function loadHistoryTable() {
        const tbody = document.getElementById('history-tbody');
        if (!tbody) return;

        tbody.innerHTML = '<tr><td colspan="6" class="skeleton" style="height: 48px; text-align: center;">Loading history records...</td></tr>';

        try {
            const resp = await fetch('/api/history');
            const data = await resp.json();
            state.historyData = data.history || [];
            renderHistoryRows(state.historyData);
        } catch (err) {
            console.error('[AIOD] History load error:', err);
            tbody.innerHTML = `<tr><td colspan="6" style="color: var(--accent-rose); text-align: center;">Failed to load history: ${err.message}</td></tr>`;
        }
    }

    function renderHistoryRows(rows) {
        const tbody = document.getElementById('history-tbody');
        if (!tbody) return;

        tbody.innerHTML = '';

        if (!rows || rows.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 32px;">No detection records found in database.</td></tr>';
            return;
        }

        rows.forEach(row => {
            const safeThumb = resolveSafeStaticUrl(row.annotated_image_path || row.image_path);
            const thumbHtml = safeThumb
                ? `<img src="${safeThumb}" alt="thumbnail" style="width: 44px; height: 32px; object-fit: cover; border-radius: 4px;">`
                : `<span class="pill-chip" style="font-size: 11px;">Local File</span>`;

            const tr = document.createElement('tr');
            tr.className = 'table-row';
            tr.innerHTML = `
                <td><span class="mono-font">#${row.id}</span></td>
                <td>${thumbHtml}</td>
                <td><b>${extractFilename(row.image_path)}</b></td>
                <td><span class="mono-font">${row.detection_time}</span></td>
                <td><span class="mono-font">${Math.round(row.inference_time_ms || 0)} ms</span></td>
                <td><span style="color: #93C5FD; font-size: 12px;">${row.detected_objects || 'None'}</span></td>
            `;
            tbody.appendChild(tr);
        });
    }

    function resolveSafeStaticUrl(filepath) {
        if (!filepath || typeof filepath !== 'string') return null;
        const normalized = filepath.replace(/\\/g, '/');

        // Check if path is within static uploads or annotated directory
        if (normalized.includes('/static/annotated/')) {
            const filename = normalized.split('/static/annotated/').pop();
            return `/static/annotated/${filename}`;
        }
        if (normalized.includes('/static/uploads/')) {
            const filename = normalized.split('/static/uploads/').pop();
            return `/static/uploads/${filename}`;
        }
        return null;
    }

    function extractFilename(filepath) {
        if (!filepath) return 'unknown';
        return filepath.replace(/\\/g, '/').split('/').pop();
    }

    function initHistorySearchAndExport() {
        const searchInput = document.getElementById('history-search-input');
        const exportCsvBtn = document.getElementById('export-csv-btn');
        const exportJsonBtn = document.getElementById('export-json-btn');

        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const query = e.target.value.toLowerCase().trim();
                const filtered = state.historyData.filter(row => {
                    const fname = extractFilename(row.image_path).toLowerCase();
                    const objs = (row.detected_objects || '').toLowerCase();
                    const idStr = String(row.id);
                    return fname.includes(query) || objs.includes(query) || idStr.includes(query);
                });
                renderHistoryRows(filtered);
            });
        }

        if (exportCsvBtn) {
            exportCsvBtn.addEventListener('click', () => {
                if (!state.historyData.length) return alert('No history data to export.');
                const headers = ['id', 'detection_time', 'image_path', 'model_name', 'confidence_threshold', 'inference_time_ms', 'total_detected_objects', 'detected_objects'];
                const csvRows = [headers.join(',')];

                state.historyData.forEach(row => {
                    const values = headers.map(h => `"${String(row[h] || '').replace(/"/g, '""')}"`);
                    csvRows.push(values.join(','));
                });

                downloadBlob(csvRows.join('\n'), 'aiod_detection_history.csv', 'text/csv');
            });
        }

        if (exportJsonBtn) {
            exportJsonBtn.addEventListener('click', () => {
                if (!state.historyData.length) return alert('No history data to export.');
                const jsonStr = JSON.stringify(state.historyData, null, 2);
                downloadBlob(jsonStr, 'aiod_detection_history.json', 'application/json');
            });
        }
    }

    function downloadBlob(content, filename, contentType) {
        const blob = new Blob([content], { type: contentType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // =====================================================================
    // GEMINI VISION MULTIMODAL AI INTEGRATION
    // =====================================================================
    async function runGeminiAnalysis() {
        if (!state.lastDetection || !state.lastDetection.original_image_url) {
            return alert('Please run a YOLO detection first.');
        }

        const btn = document.getElementById('gemini-btn');
        const output = document.getElementById('gemini-output');
        btn.disabled = true;
        btn.innerHTML = '<span class="status-dot blue"></span> Analyzing scene with Gemini...';
        output.innerText = 'Transmitting image to Google Gemini Multimodal Vision API...';

        try {
            const summaryText = Object.entries(state.lastDetection.counts || {})
                .map(([k, v]) => `${v}x ${k}`)
                .join(', ');

            const resp = await fetch('/api/gemini/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_url: state.lastDetection.original_image_url,
                    detection_summary: summaryText
                })
            });

            const res = await resp.json();
            if (res.success) {
                output.innerText = res.analysis;
            } else {
                output.innerHTML = `<span style="color: var(--accent-rose);">Gemini Notice: ${res.error || 'Failed to analyze'}</span>`;
            }
        } catch (err) {
            output.innerHTML = `<span style="color: var(--accent-rose);">Network Exception: ${err.message}</span>`;
        } finally {
            btn.disabled = false;
            btn.innerHTML = '✨ Refresh Analysis';
        }
    }

    // =====================================================================
    // OVERVIEW TELEMETRY & UTILITIES
    // =====================================================================
    async function fetchOverviewMetrics() {
        try {
            const [analyticsRes, histRes] = await Promise.all([
                fetch('/api/analytics'),
                fetch('/api/history')
            ]);
            const analytics = await analyticsRes.json();
            const histData = await histRes.json();

            animateCounter('hero-stat-total-processed', analytics.total_images_processed || 0);
            animateCounter('hero-stat-total-objects', analytics.total_objects_detected || 0);

            // Compute average latency from recent 50 history rows if available
            const rows = histData.history || [];
            if (rows.length > 0) {
                const totalLatency = rows.reduce((acc, r) => acc + (r.inference_time_ms || 0), 0);
                const avg = Math.round(totalLatency / rows.length);
                animateCounter('hero-stat-avg-latency', avg, ' ms');
            }
        } catch (e) {
            // Silently fallback to defaults
        }
    }

    function animateCounter(elementId, targetValue, suffix = '') {
        const el = document.getElementById(elementId);
        if (!el) return;

        const target = typeof targetValue === 'number' ? targetValue : parseInt(targetValue, 10) || 0;
        let current = 0;
        const duration = 600;
        const start = performance.now();

        function step(now) {
            const progress = Math.min((now - start) / duration, 1);
            current = Math.floor(progress * target);
            el.innerText = `${current.toLocaleString()}${suffix}`;
            if (progress < 1) {
                requestAnimationFrame(step);
            } else {
                el.innerText = `${target.toLocaleString()}${suffix}`;
            }
        }

        requestAnimationFrame(step);
    }

    function initRealtimeClock() {
        const clockEl = document.getElementById('realtime-clock');
        if (!clockEl) return;
        function tick() {
            const now = new Date();
            clockEl.innerText = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }
        tick();
        setInterval(tick, 1000);
    }

    // Expose helpers globally if needed
    window.AIODApp = {
        switchSection: switchSection,
        setViewMode: setViewMode,
        setTheme: setTheme
    };

})();

