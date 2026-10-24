(function () {
    "use strict";

    const PROB_LABELS = {
        focal_aware: "Focal aware",
        focal_impaired_awareness: "Focal impaired awareness",
        focal_to_bilateral_tonic_clonic: "Focal to bilateral tonic-clonic",
    };

    const CHANNEL_COLORS = [
        "#375e8f", "#6b8060", "#8265a8", "#9a6a35", "#4a7d92",
        "#8f8a4d", "#b5653a", "#5d6d8f", "#7d8c5a", "#a34a68",
        "#6b5f96", "#9a7b4a", "#47747c", "#8c6aa8", "#b3772f",
        "#5d6a3d", "#4a6e94", "#7a5a46",
    ];

    const PLOT_EVENT_FILL = "rgba(194, 60, 43, 0.12)";
    const PLOT_EVENT_LINE = "#c23c2b";
    const PLOT_AXIS = "#8f8a83";
    const PLOT_GRID = "rgba(0, 0, 0, 0.07)";
    const PLOT_ZERO = "rgba(0, 0, 0, 0.16)";
    const PLOT_TICK = "rgba(0, 0, 0, 0.22)";

    let currentData = null;
    let plotLayout = null;

    const $ = (id) => document.getElementById(id);

    const els = {
        dropZone: $("dropZone"),
        fileInput: $("fileInput"),

        viewUpload: $("viewUpload"),
        viewProcessing: $("viewProcessing"),
        viewResults: $("viewResults"),
        viewError: $("viewError"),

        procTitle: $("procTitle"),
        procSub: $("procSub"),
        procFill: $("procFill"),
        dzProgress: $("dzProgress"),

        resFileName: $("resFileName"),
        resDuration: $("resDuration"),
        resChannels: $("resChannels"),
        resSfreq: $("resSfreq"),

        eegPlot: $("eegPlot"),

        verdict: $("verdict"),
        verdictIcon: $("verdictIcon"),
        verdictTitle: $("verdictTitle"),
        verdictSub: $("verdictSub"),


        eventList: $("eventList"),
        elItems: $("elItems"),

        btnZoomIn: $("btnZoomIn"),
        btnZoomOut: $("btnZoomOut"),
        btnReset: $("btnReset"),
        btnRetry: $("btnRetry"),

        statusDot: $("statusDot"),
        statusText: $("statusText"),
        errTitle: $("errTitle"),
        errMsg: $("errMsg"),
    };

    /* ---------------- Model status ---------------- */

    async function checkStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            const ok = data.models_loaded;
            setStatus(
                ok ? "ready" : "error",
                ok
                    ? (data.classifier_available
                        ? "Models ready"
                        : "Backbone only")
                    : "Models unavailable"
            );
        } catch (e) {
            setStatus("error", "Server unreachable");
        }
    }

    function setStatus(state, text) {
        els.statusDot.className = "live-dot " + state;
        els.statusText.textContent = text;
    }

    /* ---------------- Views ---------------- */

    function showView(name) {
        [
            "viewUpload", "viewProcessing", "viewResults", "viewError"
        ].forEach((id) => els[id].classList.remove("active"));
        els["view" + cap(name)].classList.add("active");
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function cap(s) {
        return s.charAt(0).toUpperCase() + s.slice(1);
    }

    /* ---------------- Upload ---------------- */

    function bindUpload() {
        els.dropZone.addEventListener("click", () => els.fileInput.click());

        els.dropZone.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                els.fileInput.click();
            }
        });

        els.dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            els.dropZone.classList.add("dragover");
        });

        els.dropZone.addEventListener("dragleave", () => {
            els.dropZone.classList.remove("dragover");
        });

        els.dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            els.dropZone.classList.remove("dragover");
            const file = e.dataTransfer.files[0];
            if (file) handleFile(file);
        });

        els.fileInput.addEventListener("change", () => {
            const file = els.fileInput.files[0];
            if (file) handleFile(file);
        });

        els.btnRetry.addEventListener("click", () => showView("upload"));
    }

    function handleFile(file) {
        if (!/\.edf$/i.test(file.name)) {
            fail("Unsupported file type", "Please upload a valid .edf recording.");
            return;
        }

        els.dzProgress.classList.remove("hidden");
        els.dropZone.classList.add("is-busy");

        showView("processing");
        runProcessingFrames();

        const formData = new FormData();
        formData.append("file", file);

        fetch("/api/analyze", {
            method: "POST",
            body: formData,
        })
            .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
            .then(({ ok, data }) => {
                if (!ok) {
                    throw new Error((data && data.error) || "Analysis failed.");
                }
                currentData = data;
                renderResults(data);
            })
            .catch((err) => {
                fail("Analysis failed", err.message);
            });
    }

    const PROC_STEPS = [
        { pct: 15, t: "Reading recording", s: "Parsing EDF header and channels…" },
        { pct: 40, t: "Preprocessing signal", s: "Applying filters and resampling…" },
        { pct: 65, t: "Detecting seizures", s: "Scanning windows with CNN v4…" },
        { pct: 85, t: "Classifying subtype", s: "Extracting dual-embeddings…" },
        { pct: 96, t: "Finalizing", s: "Assembling the report…" },
    ];

    function runProcessingFrames() {
        let i = 0;
        els.procFill.style.width = "5%";
        const timer = setInterval(() => {
            if (i < PROC_STEPS.length) {
                const s = PROC_STEPS[i];
                els.procTitle.textContent = s.t;
                els.procSub.textContent = s.s;
                els.procFill.style.width = s.pct + "%";
                i++;
            } else {
                clearInterval(timer);
            }
        }, 1000);
    }

    /* ---------------- Render results ---------------- */

    function renderResults(data) {
        // File info
        els.resFileName.textContent = data.file_info.name;
        els.resDuration.textContent = fmtDuration(data.file_info.duration_s);
        els.resChannels.textContent = data.file_info.n_channels + " ch";
        els.resSfreq.textContent = data.file_info.sfreq + " Hz";

        // Verdict
        if (data.seizure_detected) {
            els.verdictIcon.classList.add("warn");
            els.verdictTitle.textContent =
                data.n_seizure_events +
                (data.n_seizure_events > 1 ? " events" : " event") +
                " detected";
            els.verdictSub.textContent =
                "Ictal regions are marked on the signal below";
        } else {
            els.verdictIcon.classList.remove("warn");
            els.verdictTitle.textContent = "No seizure detected";
            els.verdictSub.textContent = "Recording appears to be normal";
        }

        // Classification
        const regions = data.seizure_regions || [];
        if (regions.length > 0) {
            renderEvents(regions);
            els.eventList.classList.remove("hidden");
        } else {
            els.eventList.classList.add("hidden");
        }

        renderEEG(data);

        els.dropZone.classList.remove("is-busy");
        els.dzProgress.classList.add("hidden");
        showView("results");
    }

    function fmtDuration(sec) {
        if (sec < 60) return Math.round(sec) + "s";
        const m = Math.floor(sec / 60);
        const s = Math.round(sec % 60);
        return m + "m " + s + "s";
    }

    function fmtTime(sec) {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return m + ":" + String(s).padStart(2, "0");
    }

    function distribHTML(probs) {
        const entries = Object.entries(probs || {}).sort((a, b) => b[1] - a[1]);
        const topKey = entries.length ? entries[0][0] : null;

        return (
            '<div class="distrib">' +
            entries
                .map(([key, val]) => {
                    const pct = Math.round(val * 100);
                    return (
                        '<div class="d-row' +
                        (key === topKey ? " top" : "") +
                        '">' +
                        '<div class="d-row-top">' +
                        '<span class="d-name">' +
                        (PROB_LABELS[key] || key) +
                        "</span>" +
                        '<span class="d-val">' + pct + "%</span></div>" +
                        '<div class="d-track"><div class="d-fill" data-pct="' +
                        pct +
                        '"></div></div></div>'
                    );
                })
                .join("") +
            "</div>"
        );
    }

    function renderEvents(regions) {
        els.elItems.innerHTML = "";
        regions.forEach((r, i) => {
            const item = document.createElement("div");
            item.className = "el-item";
            item.innerHTML =
                '<div class="el-time">' +
                "<span>Event " + (i + 1) + " · " +
                fmtTime(r.start_s) + " — " + fmtTime(r.end_s) + "</span>" +
                "<span>" + Math.round(r.duration_s) + "s duration</span></div>" +
                distribHTML(r.probabilities);
            els.elItems.appendChild(item);
        });

        requestAnimationFrame(() => {
            els.elItems.querySelectorAll(".d-fill").forEach((f) => {
                f.style.width = f.dataset.pct + "%";
            });
        });
    }

    /* ---------------- EEG plot ---------------- */

    function renderEEG(data) {
        const traces = data.eeg_data.traces;
        const channels = data.file_info.channels;
        const sfreq = data.eeg_data.downsampled_sfreq;
        const nSamples = traces[0] ? traces[0].length : 0;
        const duration = nSamples / sfreq;
        const spacing = 120;

        const plotTraces = traces.map((chData, i) => {
            const offset = i * spacing;
            return {
                x: chData.map((_, j) => j / sfreq),
                y: chData.map((v) => v + offset),
                name: channels[i] || "CH" + (i + 1),
                mode: "lines",
                line: {
                    color: CHANNEL_COLORS[i % CHANNEL_COLORS.length],
                    width: 1.1,
                },
                hovertemplate:
                    "%{y:.0f} µV · %{x:.0f}s<extra>" +
                    (channels[i] || "") + "</extra>",
            };
        });

        const shapes = (data.seizure_regions || []).map((region) => ({
            type: "rect",
            xref: "x",
            yref: "paper",
            x0: region.start_s,
            x1: region.end_s,
            y0: 0,
            y1: 1,
            fillcolor: PLOT_EVENT_FILL,
            line: { color: PLOT_EVENT_LINE, width: 1.5 },
            layer: "below",
        }));

        const maxOffset = (traces.length - 1) * spacing;

        plotLayout = {
            autosize: true,
            margin: { l: 48, r: 16, t: 8, b: 48 },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            font: {
                family: "'DM Mono', monospace",
                color: PLOT_AXIS,
                size: 11,
            },
            xaxis: {
                title: {
                    text: "Time (seconds)",
                    font: {
                        family: "'Manrope', sans-serif",
                        color: PLOT_AXIS,
                        size: 12,
                    },
                },
                range: [0, duration],
                gridcolor: PLOT_GRID,
                zerolinecolor: PLOT_ZERO,
                ticks: "outside",
                tickcolor: PLOT_TICK,
            },
            yaxis: {
                title: {
                    text: "EEG amplitude (µV)",
                    font: {
                        family: "'Manrope', sans-serif",
                        color: PLOT_AXIS,
                        size: 12,
                    },
                },
                range: [-spacing * 1.4, maxOffset + spacing * 1.4],
                showticklabels: false,
                gridcolor: PLOT_GRID,
                zeroline: false,
                ticks: "",
            },
            hovermode: "closest",
            shapes: shapes,
            showlegend: false,
            dragmode: "pan",
            modebar: {
                orientation: "v",
                bgcolor: "transparent",
                color: PLOT_AXIS,
                activecolor: PLOT_EVENT_LINE,
            },
        };

        const config = {
            responsive: true,
            displaylogo: false,
            scrollZoom: true,
            modeBarButtonsToRemove: [
                "lasso2d", "select2d", "autoScale2d", "toggleSpikelines",
                "hoverClosestCartesian", "hoverCompareCartesian",
            ],
            displayModeBar: true,
        };

        Plotly.newPlot(els.eegPlot, plotTraces, plotLayout, config);
    }

    function updatePlot() {
        if (!plotLayout || !currentData) return;
        Plotly.relayout(els.eegPlot, plotLayout);
    }

    function zoomBy(factor) {
        if (!plotLayout || !currentData) return;
        const xr = plotLayout.xaxis.range;
        if (!xr) return;
        const span = (xr[1] - xr[0]) * factor;
        const c = (xr[0] + xr[1]) / 2;
        plotLayout.xaxis.range = [
            Math.max(0, c - span / 2),
            Math.min(currentData.file_info.duration_s, c + span / 2),
        ];
        updatePlot();
    }

    els.btnReset.addEventListener("click", () => {
        if (!plotLayout || !currentData) return;
        plotLayout.xaxis.range = [0, currentData.file_info.duration_s];
        updatePlot();
    });

    els.btnZoomIn.addEventListener("click", () => zoomBy(0.5));
    els.btnZoomOut.addEventListener("click", () => zoomBy(2));

    /* ---------------- Error ---------------- */

    function fail(title, message) {
        els.errTitle.textContent = title;
        els.errMsg.textContent = message;
        els.dropZone.classList.remove("is-busy");
        els.dzProgress.classList.add("hidden");
        showView("error");
    }

    /* ---------------- Init ---------------- */

    checkStatus();
    bindUpload();
})();