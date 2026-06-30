/**
 * Face Recognition — Web Interface
 *
 * Live mode: MJPEG stream (boxes drawn on frame by server)
 * Photo mode: upload + detect + assign
 * Manage mode: person CRUD + attributes
 */

document.addEventListener("DOMContentLoaded", () => {
    // ── Mode switching ──────────────────────────────────────
    const modeButtons = document.querySelectorAll(".mode-btn");
    const sections = document.querySelectorAll(".mode-section");

    modeButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            const mode = btn.dataset.mode;
            modeButtons.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            sections.forEach((s) => s.classList.remove("active"));
            document.getElementById(`${mode}-mode`).classList.add("active");

            if (mode === "manage") {
                loadPersons();
                loadEncodings();
            }
        });
    });

    // ── Live stream (just an <img>, server draws on frames) ──
    const liveStream = document.getElementById("live-stream");
    liveStream.onerror = () => {
        liveStream.alt = "Camera not available";
    };

    // ── Photo upload ────────────────────────────────────────
    const dropZone = document.getElementById("drop-zone");
    const photoInput = document.getElementById("photo-input");
    const photoPreview = document.getElementById("photo-preview");
    const photoBoxes = document.getElementById("photo-boxes");
    const detectBtn = document.getElementById("detect-btn");
    const clearBtn = document.getElementById("clear-photo-btn");
    const photoResults = document.getElementById("photo-results");

    let currentPhotoId = null;
    let currentFaces = [];
    let selectedFaces = new Set();

    dropZone.addEventListener("click", () => photoInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "#4a9eff";
    });
    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "";
    });
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "";
        if (e.dataTransfer.files.length) {
            handlePhotoUpload(e.dataTransfer.files[0]);
        }
    });

    photoInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
            handlePhotoUpload(e.target.files[0]);
        }
    });

    function handlePhotoUpload(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            photoPreview.src = e.target.result;
            photoPreview.classList.remove("hidden");
            dropZone.querySelector(".upload-prompt").classList.add("hidden");
        };
        reader.readAsDataURL(file);

        const formData = new FormData();
        formData.append("photo", file);

        fetch("/api/upload", { method: "POST", body: formData })
            .then((r) => r.json())
            .then((data) => {
                currentPhotoId = data.photo_id;
                currentFaces = data.faces;
                detectBtn.disabled = false;
                clearBtn.classList.remove("hidden");
                drawPhotoBoxes(data.faces);
            });
    }

    function drawPhotoBoxes(faces) {
        photoBoxes.innerHTML = "";
        if (!photoPreview.naturalWidth) {
            photoPreview.onload = () => drawPhotoBoxes(faces);
            return;
        }
        const rect = photoPreview.getBoundingClientRect();
        const scaleX = rect.width / photoPreview.naturalWidth;
        const scaleY = rect.height / photoPreview.naturalHeight;

        faces.forEach((face, i) => {
            const [x1, y1, x2, y2] = face.bbox;
            const box = document.createElement("div");
            box.className = "face-box";
            if (selectedFaces.has(i)) {
                box.classList.add("selected");
            }
            box.style.left = x1 * scaleX + "px";
            box.style.top = y1 * scaleY + "px";
            box.style.width = (x2 - x1) * scaleX + "px";
            box.style.height = (y2 - y1) * scaleY + "px";

            // Click to toggle selection
            box.addEventListener("click", (e) => {
                if (e.target.classList.contains("face-assign-btn")) return;
                if (selectedFaces.has(i)) {
                    selectedFaces.delete(i);
                    box.classList.remove("selected");
                } else {
                    selectedFaces.add(i);
                    box.classList.add("selected");
                }
                updateAssignSelectedBtn();
            });

            const label = document.createElement("span");
            label.className = "face-label";
            const quality = face.quality !== undefined ? ` Q:${face.quality}` : "";
            label.textContent = `${face.name} (${Math.round(face.confidence * 100)}%)${quality}`;
            box.appendChild(label);

            const assignBtn = document.createElement("button");
            assignBtn.className = "face-assign-btn";
            assignBtn.textContent = "+";
            assignBtn.onclick = (e) => {
                e.stopPropagation();
                openAssignModal(i);
            };
            box.appendChild(assignBtn);

            photoBoxes.appendChild(box);
        });
        updateAssignSelectedBtn();
    }

    function updateAssignSelectedBtn() {
        const btn = document.getElementById("assign-selected-btn");
        if (selectedFaces.size > 0) {
            btn.classList.remove("hidden");
            btn.textContent = `Assign Selected (${selectedFaces.size})`;
        } else {
            btn.classList.add("hidden");
        }
    }

    detectBtn.addEventListener("click", () => {
        if (!currentPhotoId) return;
        photoResults.innerHTML = '<div class="loading">Detecting...</div>';
        // Already detected on upload, results are shown in boxes
    });

    clearBtn.addEventListener("click", () => {
        photoPreview.src = "";
        photoPreview.classList.add("hidden");
        photoBoxes.innerHTML = "";
        photoResults.innerHTML = "";
        dropZone.querySelector(".upload-prompt").classList.remove("hidden");
        detectBtn.disabled = true;
        clearBtn.classList.add("hidden");
        document.getElementById("assign-selected-btn").classList.add("hidden");
        currentPhotoId = null;
        currentFaces = [];
        selectedFaces.clear();
        photoInput.value = "";
    });

    // ── Assign modal ────────────────────────────────────────
    const assignModal = document.getElementById("assign-modal");
    const assignList = document.getElementById("assign-person-list");
    const cancelAssign = document.getElementById("cancel-assign");

    let assigningFaceIndex = null;
    let assigningBatchIndices = null;

    function openAssignModal(faceIndex) {
        assigningFaceIndex = faceIndex;
        assigningBatchIndices = null;
        loadPersons().then((persons) => {
            assignList.innerHTML = "";
            if (persons.length === 0) {
                assignList.innerHTML = '<p class="empty">No people registered yet. Add one in Manage mode.</p>';
            }
            persons.forEach((p) => {
                const row = document.createElement("div");
                row.className = "person-row";
                row.textContent = p.name;
                row.onclick = () => assignFace(p.id, p.name);
                assignList.appendChild(row);
            });
            assignModal.classList.remove("hidden");
        });
    }

    function openBatchAssignModal(indices) {
        assigningFaceIndex = null;
        assigningBatchIndices = indices;
        loadPersons().then((persons) => {
            assignList.innerHTML = "";
            if (persons.length === 0) {
                assignList.innerHTML = '<p class="empty">No people registered yet. Add one in Manage mode.</p>';
            }
            persons.forEach((p) => {
                const row = document.createElement("div");
                row.className = "person-row";
                row.textContent = `${p.name} (${p.encoding_count || 0} enc)`;
                row.onclick = () => batchAssignFaces(p.id, p.name);
                assignList.appendChild(row);
            });
            assignModal.classList.remove("hidden");
        });
    }

    cancelAssign.onclick = () => assignModal.classList.add("hidden");

    function assignFace(personId, personName) {
        if (!currentPhotoId || assigningFaceIndex === null) return;
        const face = currentFaces[assigningFaceIndex];
        fetch("/api/encodings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                person_id: personId,
                photo_id: currentPhotoId,
                face_index: assigningFaceIndex,
            }),
        })
            .then((r) => r.json())
            .then((data) => {
                if (data.status === "added") {
                    face.name = personName;
                    face.confidence = 1.0;
                    drawPhotoBoxes(currentFaces);
                    assignModal.classList.add("hidden");
                    photoResults.innerHTML = `<div class="success">Assigned to ${personName}</div>`;
                } else {
                    photoResults.innerHTML = `<div class="error">${data.error}</div>`;
                }
            });
    }

    function batchAssignFaces(personId, personName) {
        if (!currentPhotoId || !assigningBatchIndices || assigningBatchIndices.length === 0) return;
        const indices = [...assigningBatchIndices];
        fetch("/api/batch-assign", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                photo_id: currentPhotoId,
                face_indices: indices,
                person_id: personId,
            }),
        })
            .then((r) => r.json())
            .then((data) => {
                if (data.status === "ok") {
                    indices.forEach((idx) => {
                        if (currentFaces[idx]) {
                            currentFaces[idx].name = personName;
                            currentFaces[idx].confidence = 1.0;
                        }
                    });
                    selectedFaces.clear();
                    drawPhotoBoxes(currentFaces);
                    assignModal.classList.add("hidden");
                    let msg = `Assigned ${data.added} face(s) to ${personName}`;
                    if (data.errors && data.errors.length > 0) {
                        msg += ` (${data.errors.length} errors)`;
                    }
                    photoResults.innerHTML = `<div class="success">${msg}</div>`;
                } else {
                    photoResults.innerHTML = `<div class="error">${data.error}</div>`;
                }
            });
    }

    document.getElementById("assign-selected-btn").addEventListener("click", () => {
        if (selectedFaces.size > 0) {
            openBatchAssignModal([...selectedFaces]);
        }
    });

    // ── Manage persons ──────────────────────────────────────
    const personCards = document.getElementById("person-cards");
    const encodingCards = document.getElementById("encoding-cards");
    const newPersonName = document.getElementById("new-person-name");
    const addPersonBtn = document.getElementById("add-person-btn");

    addPersonBtn.addEventListener("click", () => {
        const name = newPersonName.value.trim();
        if (!name) return;
        fetch("/api/persons", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name }),
        })
            .then((r) => r.json())
            .then(() => {
                newPersonName.value = "";
                loadPersons();
            });
    });

    function loadPersons() {
        return fetch("/api/persons")
            .then((r) => r.json())
            .then((persons) => {
                personCards.innerHTML = "";
                if (persons.length === 0) {
                    personCards.innerHTML = '<p class="empty">No people registered.</p>';
                    return;
                }
                persons.forEach((p) => {
                    const card = document.createElement("div");
                    card.className = "person-card";

                    const attrs = p.attributes || {};
                    const attrTags = Object.entries(attrs)
                        .map(([k, v]) => `<span class="attr-tag">${k}: ${v}</span>`)
                        .join("");

                    card.innerHTML = `
                        <div class="person-info">
                            <strong>${p.name}</strong>
                            <span class="encoding-count">${p.encoding_count || 0} encodings</span>
                        </div>
                        <div class="person-attrs">${attrTags}</div>
                        <div class="person-actions">
                            <button class="btn btn-small attr-btn" data-id="${p.id}">Edit Attributes</button>
                            <button class="btn btn-small btn-danger del-btn" data-id="${p.id}">Delete</button>
                        </div>
                    `;

                    card.querySelector(".del-btn").addEventListener("click", () => {
                        fetch(`/api/persons/${p.id}`, { method: "DELETE" }).then(() => loadPersons());
                    });

                    card.querySelector(".attr-btn").addEventListener("click", () => {
                        openAttributesModal(p.id, p.name, attrs);
                    });

                    personCards.appendChild(card);
                });
                return persons;
            });
    }

    function loadEncodings() {
        return fetch("/api/encodings")
            .then((r) => r.json())
            .then((encs) => {
                encodingCards.innerHTML = "";
                if (encs.length === 0) {
                    encodingCards.innerHTML = '<p class="empty">No encodings stored.</p>';
                    return;
                }
                encs.forEach((e) => {
                    const card = document.createElement("div");
                    card.className = "encoding-card";
                    card.innerHTML = `
                        <span>${e.person_name || "Unknown"}</span>
                        <button class="btn btn-small btn-danger" data-id="${e.id}">Remove</button>
                    `;
                    card.querySelector(".btn-danger").addEventListener("click", () => {
                        fetch(`/api/encodings/${e.id}`, { method: "DELETE" }).then(() => loadEncodings());
                    });
                    encodingCards.appendChild(card);
                });
            });
    }

    // ── Attributes modal ────────────────────────────────────
    const attrModal = document.getElementById("attributes-modal");
    const attrEditor = document.getElementById("attributes-editor");
    const saveAttrBtn = document.getElementById("save-attributes");
    const cancelAttrBtn = document.getElementById("cancel-attributes");

    let attrPersonId = null;

    function openAttributesModal(personId, personName, attrs) {
        attrPersonId = personId;
        attrEditor.innerHTML = `<p>Attributes for <strong>${personName}</strong></p>`;
        const entries = Object.entries(attrs);
        if (entries.length === 0) {
            entries.push(["", ""]);
        }
        entries.forEach(([key, val]) => {
            const row = document.createElement("div");
            row.className = "attr-row";
            row.innerHTML = `
                <input type="text" class="attr-key" value="${key}" placeholder="Key" />
                <input type="text" class="attr-val" value="${val}" placeholder="Value" />
                <button class="btn btn-small btn-danger remove-attr">X</button>
            `;
            row.querySelector(".remove-attr").addEventListener("click", () => row.remove());
            attrEditor.appendChild(row);
        });

        const addRow = document.createElement("button");
        addRow.className = "btn btn-small";
        addRow.textContent = "+ Add Field";
        addRow.addEventListener("click", () => {
            const row = document.createElement("div");
            row.className = "attr-row";
            row.innerHTML = `
                <input type="text" class="attr-key" placeholder="Key" />
                <input type="text" class="attr-val" placeholder="Value" />
                <button class="btn btn-small btn-danger remove-attr">X</button>
            `;
            row.querySelector(".remove-attr").addEventListener("click", () => row.remove());
            attrEditor.appendChild(row);
        });
        attrEditor.appendChild(addRow);

        attrModal.classList.remove("hidden");
    }

    saveAttrBtn.addEventListener("click", () => {
        const attrs = {};
        attrEditor.querySelectorAll(".attr-row").forEach((row) => {
            const key = row.querySelector(".attr-key").value.trim();
            const val = row.querySelector(".attr-val").value.trim();
            if (key) attrs[key] = val;
        });
        fetch(`/api/persons/${attrPersonId}/attributes`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ attributes: attrs }),
        }).then(() => {
            attrModal.classList.add("hidden");
            loadPersons();
        });
    });

    cancelAttrBtn.addEventListener("click", () => attrModal.classList.add("hidden"));

    // ── Data export / import ──────────────────────────────────
    const exportDataBtn = document.getElementById("export-data-btn");
    const importDataBtn = document.getElementById("import-data-btn");
    const importFileInput = document.getElementById("import-file-input");

    exportDataBtn.addEventListener("click", async () => {
        try {
            // Fetch both persons and encodings for a full export
            const [personsRes, encodingsRes] = await Promise.all([
                fetch("/api/export/persons"),
                fetch("/api/export/encodings"),
            ]);
            const personsData = await personsRes.json();
            const encodingsData = await encodingsRes.json();

            const exportPayload = {
                export_version: 1,
                exported_at: new Date().toISOString(),
                persons: personsData.persons || [],
                encodings: encodingsData.encodings || [],
            };

            const blob = new Blob(
                [JSON.stringify(exportPayload, null, 2)],
                { type: "application/json" }
            );
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `face-data-export-${new Date().toISOString().slice(0, 10)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (err) {
            alert("Export failed: " + err.message);
        }
    });

    importDataBtn.addEventListener("click", () => {
        importFileInput.click();
    });

    importFileInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        try {
            const text = await file.text();
            const data = JSON.parse(text);

            if (!data.persons || !Array.isArray(data.persons)) {
                alert("Invalid import file: missing 'persons' array.");
                return;
            }

            const res = await fetch("/api/import/persons", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ persons: data.persons }),
            });
            const result = await res.json();

            if (result.error) {
                alert("Import error: " + result.error);
            } else {
                const skipped = result.skipped_duplicates || [];
                let msg = `Imported ${result.imported} person(s).`;
                if (skipped.length > 0) {
                    msg += ` Skipped duplicates: ${skipped.join(", ")}`;
                }
                alert(msg);
                loadPersons();
                loadEncodings();
            }
        } catch (err) {
            alert("Import failed: " + err.message);
        } finally {
            importFileInput.value = "";
        }
    });
});
