/* Document Translator frontend: login, upload, progress polling, results. */

/* ---------------- toasts ---------------- */
function toast(message, kind = "error", timeout = 7000) {
  const host = document.getElementById("toasts");
  if (!host) return;
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.setAttribute("role", kind === "error" ? "alert" : "status");

  const text = document.createElement("span");
  text.textContent = message;
  el.appendChild(text);

  const close = document.createElement("button");
  close.textContent = "✕";
  close.setAttribute("aria-label", "Dismiss");
  close.onclick = () => el.remove();
  el.appendChild(close);

  host.appendChild(el);
  if (timeout) setTimeout(() => el.remove(), timeout);
}

/* Pull a readable message out of any response; never surface a stack trace. */
async function errorMessage(response, fallback) {
  try {
    const data = await response.json();
    if (data && typeof data.error === "string") return data.error;
    if (data && typeof data.detail === "string") return data.detail;
  } catch (_) { /* body was not JSON */ }
  return fallback || `Request failed (${response.status}).`;
}

/* ---------------- login ---------------- */
function initLogin() {
  const form = document.getElementById("login-form");
  const errorBox = document.getElementById("login-error");
  const button = document.getElementById("login-button");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorBox.hidden = true;

    const username = document.getElementById("username").value.trim();
    const password = document.getElementById("password").value;
    if (!username || !password) {
      errorBox.textContent = "Enter both a username and a password.";
      errorBox.hidden = false;
      return;
    }

    button.disabled = true;
    button.textContent = "Signing in…";
    try {
      const body = new FormData();
      body.append("username", username);
      body.append("password", password);
      const response = await fetch("/api/login", { method: "POST", body });

      if (response.ok) {
        window.location.href = "/";
        return;
      }
      errorBox.textContent = await errorMessage(
        response, "Incorrect username or password."
      );
      errorBox.hidden = false;
      document.getElementById("password").value = "";
    } catch (_) {
      errorBox.textContent = "Could not reach the server. Check your connection.";
      errorBox.hidden = false;
    } finally {
      button.disabled = false;
      button.textContent = "Sign in";
    }
  });
}

/* ---------------- main app ---------------- */
const STAGES = ["extracting", "translating", "rebuilding", "done"];

function initApp() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const fileChip = document.getElementById("file-chip");
  const fileName = document.getElementById("file-name");
  const translateBtn = document.getElementById("translate-btn");

  let selectedFile = null;
  let direction = "en2ar";
  let pollTimer = null;

  /* --- file selection --- */
  function acceptFile(file) {
    if (!file) return;
    const name = file.name.toLowerCase();
    if (!name.endsWith(".pdf") && !name.endsWith(".docx")) {
      toast("Only PDF and DOCX files can be translated.", "error");
      return;
    }
    if (file.size === 0) {
      toast("That file is empty.", "error");
      return;
    }
    selectedFile = file;
    fileName.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
    fileChip.hidden = false;
    translateBtn.disabled = false;
  }

  function clearFile() {
    selectedFile = null;
    fileInput.value = "";
    fileChip.hidden = true;
    translateBtn.disabled = true;
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => acceptFile(fileInput.files[0]));
  document.getElementById("file-clear").addEventListener("click", clearFile);

  ["dragenter", "dragover"].forEach((type) =>
    dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length > 1) toast("Only the first file was used.", "warning");
    if (files && files.length) acceptFile(files[0]);
  });

  /* --- direction toggle --- */
  document.querySelectorAll(".seg").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".seg").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-checked", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-checked", "true");
      direction = btn.dataset.direction;
    });
  });

  /* --- views --- */
  function show(view) {
    ["upload", "progress", "result"].forEach((name) => {
      document.getElementById(`view-${name}`).hidden = name !== view;
    });
  }

  function setProgress(percent, message, status) {
    const clamped = Math.max(0, Math.min(100, percent));
    document.getElementById("progress-fill").style.width = `${clamped}%`;
    document.getElementById("progress-bar-outer")
            .setAttribute("aria-valuenow", String(clamped));
    document.getElementById("progress-message").textContent = message;

    const reached = STAGES.indexOf(status);
    document.querySelectorAll("#stages li").forEach((li) => {
      const index = STAGES.indexOf(li.dataset.stage);
      li.classList.toggle("active", index === reached);
      li.classList.toggle("complete", reached > index);
    });
  }

  /* --- submit --- */
  translateBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    translateBtn.disabled = true;

    const body = new FormData();
    body.append("file", selectedFile);
    body.append("direction", direction);
    body.append("mirror", document.getElementById("opt-mirror").checked);
    body.append("underline", document.getElementById("opt-underline").checked);
    body.append("flip_images", document.getElementById("opt-flip-images").checked);
    body.append("html_engine", document.getElementById("opt-html").checked);

    try {
      const response = await fetch("/api/jobs", { method: "POST", body });
      if (response.status === 401) return sessionExpired();
      if (!response.ok) {
        toast(await errorMessage(response, "The upload was rejected."), "error");
        translateBtn.disabled = false;
        return;
      }
      const job = await response.json();
      document.getElementById("progress-filename").textContent = job.filename;
      setProgress(job.progress || 2, job.message || "Waiting to start…", job.status);
      show("progress");
      poll(job.id);
    } catch (_) {
      toast("Could not reach the server. Check your connection.", "error");
      translateBtn.disabled = false;
    }
  });

  /* --- polling --- */
  function poll(jobId) {
    clearTimeout(pollTimer);
    let failures = 0;

    async function tick() {
      try {
        const response = await fetch(`/api/jobs/${jobId}`);
        if (response.status === 401) return sessionExpired();
        if (!response.ok) throw new Error("bad status");

        const job = await response.json();
        failures = 0;
        setProgress(job.progress, job.message, job.status);

        if (job.status === "done") return showResult(job);
        if (job.status === "failed") {
          toast(job.error || "The translation failed.", "error", 12000);
          show("upload");
          translateBtn.disabled = false;
          return;
        }
        pollTimer = setTimeout(tick, 1000);
      } catch (_) {
        failures += 1;
        if (failures >= 5) {
          toast("Lost contact with the server while translating.", "error");
          show("upload");
          translateBtn.disabled = false;
          return;
        }
        pollTimer = setTimeout(tick, 2000);
      }
    }
    tick();
  }

  /* --- result --- */
  function showResult(job) {
    document.getElementById("result-filename").textContent = job.filename;
    const link = document.getElementById("download-link");
    link.href = `/api/jobs/${job.id}/download`;
    renderQA(job.qa);
    show("result");
    toast("Your translated document is ready.", "success", 5000);
  }

  function renderQA(qa) {
    const section = document.getElementById("qa-section");
    const summary = document.getElementById("qa-summary");
    const list = document.getElementById("qa-list");
    summary.innerHTML = "";
    list.innerHTML = "";

    const entries = (qa && qa.entries) || [];
    const notable = entries.filter((e) => e.category !== "summary");
    if (!notable.length) { section.hidden = true; return; }

    const counts = (qa && qa.counts) || {};
    const LABELS = {
      error: "need attention",
      warning: "worth a look",
      info: "notes",
    };

    const preserved = entries.filter((e) => e.category === "preserved").length;
    if (preserved) {
      const pill = document.createElement("span");
      pill.className = "pill preserved";
      pill.textContent =
        `${preserved} already in the target language — left unchanged`;
      summary.appendChild(pill);
    }
    ["error", "warning", "info"].forEach((severity) => {
      if (!counts[severity]) return;
      const pill = document.createElement("span");
      pill.className = `pill ${severity}`;
      pill.textContent = `${counts[severity]} ${LABELS[severity]}`;
      summary.appendChild(pill);
    });

    const order = { error: 0, warning: 1, info: 2 };
    notable.sort((a, b) => order[a.severity] - order[b.severity]);

    const LIMIT = 8;
    notable.slice(0, LIMIT).forEach((entry) => list.appendChild(qaItem(entry)));

    if (notable.length > LIMIT) {
      const more = document.createElement("button");
      more.className = "qa-more";
      more.textContent = `Show ${notable.length - LIMIT} more`;
      more.onclick = () => {
        notable.slice(LIMIT).forEach((entry) => list.insertBefore(qaItem(entry), more));
        more.remove();
      };
      list.appendChild(more);
    }
    section.hidden = false;
  }

  function qaItem(entry) {
    const item = document.createElement("div");
    item.className = `qa-item ${entry.severity}`;
    item.textContent = entry.message;
    const excerpt = entry.detail && entry.detail.excerpt;
    if (excerpt) {
      const span = document.createElement("span");
      span.className = "excerpt";
      span.textContent = `“${excerpt}”`;
      item.appendChild(span);
    }
    return item;
  }

  document.getElementById("another").addEventListener("click", () => {
    clearFile();
    show("upload");
  });

  /* --- session --- */
  function sessionExpired() {
    clearTimeout(pollTimer);
    toast("Your session expired. Redirecting to sign in…", "warning", 4000);
    setTimeout(() => (window.location.href = "/login"), 1500);
  }

  document.getElementById("logout").addEventListener("click", async () => {
    try { await fetch("/api/logout", { method: "POST" }); } catch (_) {}
    window.location.href = "/login";
  });

  fetch("/api/me")
    .then((r) => (r.ok ? r.json() : null))
    .then((me) => {
      if (!me) return;
      document.getElementById("user-line").textContent = `Signed in as ${me.username}`;
      if (me.provider === "echo") {
        document.getElementById("provider-line").textContent =
          "English ⇄ Arabic · no translation provider configured";
        toast(
          "No translation provider is configured, so text will be copied " +
          "unchanged. Set ANTHROPIC_API_KEY to enable real translation.",
          "warning", 12000
        );
      }
    })
    .catch(() => {});
}
