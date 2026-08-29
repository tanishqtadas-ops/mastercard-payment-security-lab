document.addEventListener("DOMContentLoaded", () => {
    fetch('/api/demo')
        .then(res => res.json())
        .then(data => {
            initDashboard(data);
        })
        .catch(err => {
            console.error("Failed to load demo data:", err);
            document.body.innerHTML = `<h2 style="color:var(--danger); padding:20px;">Error loading demo data: ${err.message}</h2><pre style="color:white; padding:20px;">${err.stack}</pre>`;
        });
});

function initDashboard(data) {
    // 1. Populate summary cards
    document.querySelector('#card-rounds .card-value').innerText = data.total_rounds;
    document.querySelector('#card-detection .card-value').innerText = `${(data.dashboard_summary.overall_detection_rate * 100).toFixed(1)}%`;
    document.querySelector('#card-updates .card-value').innerText = data.model_updates;
    const recoveryStr = data.post_learning_recovery_observed ? "OBSERVED" : "NOT OBSERVED";
    const recoveryEl = document.querySelector('#card-recovery .card-value');
    recoveryEl.innerText = recoveryStr;
    if (data.post_learning_recovery_observed) {
        recoveryEl.style.color = "var(--danger)";
    } else {
        recoveryEl.style.color = "var(--text-main)";
    }

    // 2. Populate Arms-Race Summary
    const armsHtml = `
        <div class="summary-stat">
            <div class="summary-stat-label">Rounds Tracked</div>
            <div class="summary-stat-value">${data.dashboard_summary.total_rounds}</div>
        </div>
        <div class="summary-stat">
            <div class="summary-stat-label">Overall Detection</div>
            <div class="summary-stat-value">${(data.dashboard_summary.overall_detection_rate * 100).toFixed(1)}%</div>
        </div>
        <div class="summary-stat">
            <div class="summary-stat-label">Avg Attack Sophistication</div>
            <div class="summary-stat-value">${data.dashboard_summary.average_attack_difficulty.toFixed(4)}</div>
        </div>
        <div class="summary-stat">
            <div class="summary-stat-label">Model Updates</div>
            <div class="summary-stat-value">${data.dashboard_summary.model_update_count}</div>
        </div>
        <div class="summary-stat">
            <div class="summary-stat-label">Post-Learning Recovery</div>
            <div class="summary-stat-value" style="color: ${data.post_learning_recovery_observed ? 'var(--danger)' : 'var(--text-main)'}">${recoveryStr}</div>
        </div>
    `;
    document.getElementById('arms-race-summary').innerHTML = armsHtml;

    // 3. Render Family 1
    renderTimeline('f1-timeline', data.family1_results, data.update_records.filter(u => u.family === 'Family 1 - Adaptive Transaction-Pattern Evasion' || u.family === 'ADAPTIVE_EVASION'));
    renderLearningPanel('f1-learning', data.update_records.filter(u => u.family === 'Family 1 - Adaptive Transaction-Pattern Evasion' || u.family === 'ADAPTIVE_EVASION'));
    renderEvaluation('f1-evaluation', data, 'ADAPTIVE_EVASION');
    drawRiskChart('f1-risk-chart', data.family1_results, 'Family 1 Risk Score');

    // 4. Render Family 2
    renderTimeline('f2-timeline', data.family2_results, []);
    renderEvaluation('f2-evaluation', data, 'AGENT_BEHAVIOR');
    drawRiskChart('f2-risk-chart', data.family2_results, 'Family 2 Risk Score');

    // 5. Render Family 3
    renderTimeline('f3-timeline', data.family3_results, []);
    renderEvaluation('f3-evaluation', data, 'SYNTHETIC_IDENTITY');
    drawRiskChart('f3-risk-chart', data.family3_results, 'Family 3 Risk Score');
    renderXGBoostSummary('f3-xgboost', data.family3_results);

    // Setup Tabs
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(tab.dataset.target).classList.add('active');
        });
    });
}

function renderTimeline(elementId, results, updates) {
    const container = document.getElementById(elementId);
    let html = '';

    // Create a combined list of events
    let events = [];
    results.forEach(r => {
        events.push({ type: 'round', data: r });
        // Check if an update occurred after this round
        const correspondingUpdate = updates.find(u => u.round_index === r.outcome_metrics.round_index);
        if (correspondingUpdate) {
            events.push({ type: 'update', data: correspondingUpdate });
        }
    });

    events.forEach(ev => {
        if (ev.type === 'round') {
            const r = ev.data;
            const detected = r.feedback.detected;
            const statusClass = detected ? 'detected' : 'missed';
            const statusText = detected ? 'DETECTED' : 'MISSED (FN)';
            html += `
                <div class="timeline-item ${statusClass}">
                    <div class="timeline-marker">Round ${r.outcome_metrics.round_index}</div>
                    <div class="timeline-content">
                        <strong>${statusText} (Risk: ${r.prediction_result.risk_score.toFixed(4)})</strong>
                        <span>Model: ${r.prediction_result.model_version}</span>
                        ${r.feedback.explanation_data ? `<span><br>${r.feedback.explanation_data.explanation || ''}</span>` : ''}
                    </div>
                </div>
            `;
        } else if (ev.type === 'update') {
            const u = ev.data;
            html += `
                <div class="timeline-item update">
                    <div class="timeline-marker">UPDATE</div>
                    <div class="timeline-content">
                        <strong>BLUE TEAM MODEL RETRAINING</strong>
                        <span>${u.previous_model_version} &rarr; ${u.new_model_version}</span>
                    </div>
                </div>
            `;
        }
    });

    if (events.length === 0) {
        html = '<div>No timeline events.</div>';
    }

    container.innerHTML = html;
}

function renderLearningPanel(elementId, updates) {
    const container = document.getElementById(elementId);
    if (updates.length === 0) {
        container.innerHTML = '<div>No learning updates recorded.</div>';
        return;
    }

    let html = '';
    updates.forEach(u => {
        html += `
            <div class="update-card">
                <div><strong>Update at Round ${u.round_index}</strong></div>
                <div>Models: <code>${u.previous_model_version}</code> &rarr; <code>${u.new_model_version}</code></div>
                <div>False Negatives Used: ${u.false_negative_count}</div>
                <div>Legitimate Baselines: ${u.baseline_count}</div>
                <div>Total Training Samples: ${u.training_sample_count}</div>
            </div>
        `;
    });
    container.innerHTML = html;
}

function renderEvaluation(elementId, data, familyKey) {
    const live = data.live_evaluation.per_family_results[familyKey];
    const clean = data.clean_evaluation[familyKey];

    let html = '';

    if (live) {
        html += `
            <h5 style="color:var(--text-highlight); margin-bottom:10px;">Live Attack Round Evaluation</h5>
            <div class="eval-metric"><span class="eval-label">Detection Rate (Recall)</span><span class="eval-value">${(live.metrics.recall * 100).toFixed(1)}%</span></div>
            <div class="eval-metric"><span class="eval-label">Accuracy</span><span class="eval-value">${(live.metrics.accuracy * 100).toFixed(1)}%</span></div>
            <div class="eval-metric"><span class="eval-label">Confusion Matrix</span><span class="eval-value">TP:${live.confusion_matrix.true_positives} FN:${live.confusion_matrix.false_negatives}</span></div>
            <div class="eval-metric"><span class="eval-label">Avg Risk Score</span><span class="eval-value">${live.risk_metrics.average_risk.toFixed(4)}</span></div>
        `;
    }

    if (clean) {
        let cleanTitle = "Clean Baseline Generalization";
        let sourceHtml = "";

        if (clean.details && clean.details.held_out_type === 'legitimate_identities') {
            cleanTitle = "Isolated Held-Out Generalization";
            sourceHtml = `<div class="eval-metric"><span class="eval-label">Source</span><span class="eval-value">data/held_out/heldout_identities.json</span></div>`;
        }

        html += `
            <h5 style="color:var(--success); margin-top:20px; margin-bottom:10px;">${cleanTitle}</h5>
            ${sourceHtml}
            <div class="eval-metric"><span class="eval-label">Samples Tested</span><span class="eval-value">${clean.sample_count}</span></div>
            <div class="eval-metric"><span class="eval-label">Clean Pass Rate</span><span class="eval-value">${(clean.clean_pass_rate * 100).toFixed(1)}%</span></div>
            <div class="eval-metric"><span class="eval-label">False Positive Rate</span><span class="eval-value">${(clean.false_positive_rate * 100).toFixed(1)}%</span></div>
        `;
    }

    document.getElementById(elementId).innerHTML = html;
}

function renderXGBoostSummary(elementId, results) {
    const container = document.getElementById(elementId);
    if (results.length === 0) return;
    const lastResult = results[results.length - 1];

    let html = `
        <div style="margin-bottom:10px;"><strong>Model Version:</strong> ${lastResult.prediction_result.model_version}</div>
    `;

    const feats = lastResult.prediction_result.feature_contributions;
    if (feats && Object.keys(feats).length > 0) {
        html += `<div style="margin-bottom:10px;"><strong>Feature Contributions:</strong></div><ul>`;
        for (const [k, v] of Object.entries(feats)) {
            html += `<li>${k}: ${v.toFixed(4)}</li>`;
        }
        html += `</ul>`;
    } else {
        html += `<div>No feature contribution data exposed.</div>`;
    }

    container.innerHTML = html;
}

function drawRiskChart(canvasId, results, title) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.length === 0) return;

    const padding = 30;
    const width = canvas.width - padding * 2;
    const height = canvas.height - padding * 2;

    // Background Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, padding + height);
    ctx.lineTo(padding + width, padding + height);
    ctx.stroke();

    // Threshold Line (Risk = 0.5)
    const thresholdY = padding + height - (0.5 * height);
    ctx.strokeStyle = 'rgba(255, 76, 76, 0.5)';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(padding, thresholdY);
    ctx.lineTo(padding + width, thresholdY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Data points
    const stepX = width / Math.max(1, (results.length - 1));

    ctx.beginPath();
    ctx.strokeStyle = '#66fcf1';
    ctx.lineWidth = 2;

    results.forEach((r, i) => {
        const x = padding + (i * stepX);
        const y = padding + height - (r.prediction_result.risk_score * height);

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Dots
    results.forEach((r, i) => {
        const x = padding + (i * stepX);
        const y = padding + height - (r.prediction_result.risk_score * height);

        ctx.fillStyle = r.feedback.detected ? '#ff4c4c' : '#45a29e';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
    });

    // Labels
    ctx.fillStyle = '#c5c6c7';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('1.0', padding - 5, padding + 5);
    ctx.fillText('0.5', padding - 5, thresholdY + 3);
    ctx.fillText('0.0', padding - 5, padding + height);

    ctx.textAlign = 'center';
    results.forEach((r, i) => {
        const x = padding + (i * stepX);
        ctx.fillText(`R${r.outcome_metrics.round_index}`, x, padding + height + 15);
    });
}
