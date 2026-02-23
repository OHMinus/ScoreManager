document.addEventListener('DOMContentLoaded', function() {
    console.log('UI.js loaded');

    // ==========================================
    // Page: Index (page-index) & Scan (page-scan)
    // ==========================================
    const indexForm = document.querySelector('form');
    if (indexForm && (document.body.classList.contains('page-index') || document.body.classList.contains('page-scan'))) {
        indexForm.addEventListener('submit', function(event) {
            // Check if the clicked button was a submit button
            // If so, show loading overlay
            // Note: form submission triggers page reload, so we just show the overlay
            showLoading();
        });
    }

    // Also handle specific buttons if needed, or rely on form submit
    const btnProcess = document.getElementById('btn-process');
    const btnScan = document.getElementById('btn-scan');
    const btnFinish = document.querySelector('.btn-finish');

    if (btnProcess) btnProcess.addEventListener('click', () => { /* let form submit handle it */ });
    if (btnScan) btnScan.addEventListener('click', () => { /* let form submit handle it */ });
    if (btnFinish) btnFinish.addEventListener('click', () => { /* let form submit handle it */ });

    function showLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) overlay.style.display = 'block';

        const buttons = document.querySelectorAll('button');
        buttons.forEach(btn => btn.style.display = 'none');
    }


    // ==========================================
    // Page: Preview (page-preview)
    // ==========================================
    if (document.body.classList.contains('page-preview')) {
        const pieceInput = document.getElementById('pieceInput');
        const hiddenPiece = document.getElementById('hiddenPiece');
        const instrumentInput = document.querySelector('input[name="instrument"]');
        const hiddenInst = document.getElementById('hiddenInst');
        const profileSelection = document.getElementById('profileSelection');
        const newPieceFields = document.getElementById('newPieceFields');
        const rotateUrl = document.body.dataset.rotateUrl;
        const profilesUrl = document.body.dataset.profilesUrl || '/api/get_profiles';

        // Initialize
        if (pieceInput) {
            fetchProfiles(pieceInput.value.trim());

            pieceInput.addEventListener('input', function() {
                if (hiddenPiece) hiddenPiece.value = this.value;
                fetchProfiles(this.value.trim());
            });
        }

        if (instrumentInput) {
            instrumentInput.addEventListener('input', function() {
                if (hiddenInst) hiddenInst.value = this.value;
            });
        }

        // Profile Selection Logic (Event Delegation)
        if (profileSelection) {
            profileSelection.addEventListener('click', function(e) {
                const label = e.target.closest('.profile-option');
                if (label) {
                    // Check radio
                    const radio = label.querySelector('input[type="radio"]');
                    if (radio) {
                        radio.checked = true;
                        updateProfileUI(radio.value);
                    }
                }
            });
        }

        function updateProfileUI(mode) {
            document.querySelectorAll('.profile-option').forEach(el => el.classList.remove('selected'));

            let labelId;
            if (mode === 'new') {
                labelId = 'lbl_mode_new';
            } else if (mode.startsWith('existing_')) {
                const idx = mode.split('_')[1];
                labelId = 'lbl_mode_ex_' + idx;
            }

            const labelEl = document.getElementById(labelId);
            if (labelEl) labelEl.classList.add('selected');

            if (newPieceFields) {
                newPieceFields.style.display = (mode === 'new') ? 'block' : 'none';
            }
        }

        function fetchProfiles(piece) {
            if (!piece) {
                renderProfiles([]);
                return;
            }

            fetch(profilesUrl + '?piece=' + encodeURIComponent(piece))
                .then(res => res.json())
                .then(data => renderProfiles(data))
                .catch(err => console.error('Profile fetch error:', err));
        }

        function renderProfiles(profiles) {
            let html = '';

            // "New" Option
            html += `
                <label class="profile-option selected" id="lbl_mode_new">
                    <input type="radio" id="mode_new" name="save_mode" value="new" checked>
                    <strong>✨ 新規楽曲として登録</strong>
                </label>
            `;

            // Existing Options
            profiles.forEach((p, index) => {
                const comp = p.composer ? p.composer : '未設定';
                const arr = p.arranger ? p.arranger : '未設定';

                html += `
                    <label class="profile-option" id="lbl_mode_ex_${index}">
                        <input type="radio" id="mode_ex_${index}" name="save_mode" value="existing_${index}">
                        <strong>📂 既存のデータに追加</strong>
                        <div class="profile-details">作曲: ${comp} / 編曲: ${arr}</div>
                        <input type="hidden" name="ex_id_${index}" value="${p.id}">
                    </label>
                `;
            });

            if (profileSelection) {
                profileSelection.innerHTML = html;
                updateProfileUI('new');
            }
        }

        // Rotate Image Logic (Event Delegation)
        const gridContainer = document.querySelector('.grid-container');
        if (gridContainer) {
            gridContainer.addEventListener('click', function(e) {
                const btn = e.target.closest('button');
                if (btn && btn.dataset.action === 'rotate') {
                    const filename = btn.dataset.filename;
                    const direction = btn.dataset.direction;
                    rotateImage(filename, direction);
                }
            });
        }

        function rotateImage(filename, direction) {
            const imgElement = document.getElementById('img-' + filename);
            if (imgElement) imgElement.style.opacity = '0.4';

            const formData = new FormData();
            formData.append('filename', filename);
            formData.append('direction', direction);

            fetch(rotateUrl, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const currentSrc = imgElement.src.split('?')[0];
                    imgElement.src = currentSrc + '?t=' + new Date().getTime();
                } else {
                    alert('Rotate Error: ' + data.error);
                }
            })
            .catch(err => {
                console.error(err);
                alert('Communication Error');
            })
            .finally(() => {
                if (imgElement) {
                    imgElement.onload = function() {
                        imgElement.style.opacity = '1.0';
                    };
                    setTimeout(() => { imgElement.style.opacity = '1.0'; }, 1000);
                }
            });
        }
    }


    // ==========================================
    // Page: Confirm Save (page-confirm)
    // ==========================================
    if (document.body.classList.contains('page-confirm')) {
        const confirmForm = document.getElementById('confirmForm');
        const btnSave = document.getElementById('btn-save');
        const btnCancel = document.getElementById('btn-cancel');
        const loadingMsg = document.getElementById('save-loading');

        function hideButtons() {
            // Delay to allow form validation
            setTimeout(function() {
                if (confirmForm.checkValidity()) {
                    if (btnSave) btnSave.style.display = 'none';
                    if (btnCancel) btnCancel.style.display = 'none';
                    if (loadingMsg) loadingMsg.style.display = 'block';
                }
            }, 50);
        }

        if (confirmForm) {
            confirmForm.addEventListener('submit', function() {
                hideButtons();
            });
        }

        // Also handle cancel button if it submits a form
        // But here it's in a separate form.
        const cancelForm = document.querySelector('form[action*="index"]'); // rough check
        if (cancelForm) {
             cancelForm.addEventListener('submit', function() {
                 if (btnSave) btnSave.style.display = 'none';
                 if (btnCancel) btnCancel.style.display = 'none';
             });
        }
    }


    // ==========================================
    // Page: View Score (page-view)
    // ==========================================
    if (document.body.classList.contains('page-view')) {
        const printBtn = document.querySelector('.print-box button[type="submit"]');
        if (printBtn) {
            printBtn.addEventListener('click', function() {
                // We use click because it might be inside a form but we want UI feedback
                // Actually form submit event is better but let's stick to the original behavior
                this.innerHTML = '送信中...';
                this.style.opacity = '0.7';
            });
        }
    }

});
