(() => {
  const form = document.getElementById("assess-form");
  const inputs = [...form.querySelectorAll("input")];
  const btn = document.getElementById("analyze-btn");
  const spinner = btn.querySelector(".spinner");
  const btnText = btn.querySelector(".btn-text");
  const formError = document.getElementById("form-error");
  const results = document.getElementById("results");
  const $ = (id) => document.getElementById(id);

  // Live value next to each slider
  form.querySelectorAll(".slider input").forEach((r) => {
    r.addEventListener("input", () => (r.nextElementSibling.textContent = r.value));
  });

  function setFieldError(input, message) {
    input.closest(".field").classList.toggle("invalid", Boolean(message));
    $(input.id + "-err").textContent = message || "";
  }

  function validate() {
    let ok = true;
    const values = {};
    inputs.forEach((input) => {
      const label = input.dataset.label;
      const min = Number(input.min);
      const max = Number(input.max);
      const raw = input.value.trim();
      let message = "";
      if (raw === "") message = `${label} is required.`;
      else if (!Number.isFinite(Number(raw))) message = `${label} must be a valid number.`;
      else if (Number(raw) < 0) message = `${label} cannot be negative.`;
      else if (Number(raw) < min || Number(raw) > max) message = `${label} must be between ${min} and ${max}.`;
      setFieldError(input, message);
      if (message) ok = false;
      else values[input.name] = Number(raw);
    });
    if (ok && values.company_tenure > values.experience_years) {
      setFieldError($("company_tenure"), "Years at Current Company cannot exceed Years of Experience.");
      ok = false;
    }
    return ok ? values : null;
  }

  function showError(message) {
    formError.textContent = message;
    formError.hidden = !message;
  }

  function setLoading(loading) {
    btn.disabled = loading;
    spinner.hidden = !loading;
    btnText.textContent = loading ? "Analyzing..." : "Analyze My AI Job Impact";
  }

  function render(data) {
    const level = data.risk_level.toLowerCase();
    const prob = data.layoff_probability * 100;

    $("res-layoff").textContent = data.layoff_prediction.toUpperCase();
    $("res-prob").textContent = prob.toFixed(1) + "%";
    $("res-level").textContent = data.risk_level.toUpperCase();
    $("res-level").className = "badge " + level;
    $("res-message").textContent = data.message;
    $("res-score").textContent = Math.round(data.risk_score);

    const factors = $("res-factors");
    factors.replaceChildren(...data.factors.map((text) => Object.assign(document.createElement("li"), { textContent: text })));

    $("reco").hidden = !(data.layoff_prediction === "Yes" || data.risk_level !== "Low");

    results.hidden = false;
    // Reset bars to 0, then animate to their values
    ["prob-bar", "score-bar"].forEach((id) => { $(id).style.width = "0"; $(id).className = level; });
    requestAnimationFrame(() => requestAnimationFrame(() => {
      $("prob-bar").style.width = prob + "%";
      $("score-bar").style.width = data.risk_score + "%";
      $("prob-bar").parentElement.setAttribute("aria-valuenow", Math.round(prob));
    }));
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    showError("");
    const payload = validate();
    if (!payload) {
      showError("Please fix the highlighted fields.");
      return;
    }
    setLoading(true);
    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.error || "Something went wrong.");
      render(data);
    } catch (err) {
      results.hidden = true;
      showError(err instanceof TypeError ? "Could not reach the server. Check your connection and try again." : err.message);
    } finally {
      setLoading(false);
    }
  });

  $("reset-btn").addEventListener("click", () => {
    form.reset();
    form.querySelectorAll(".slider output").forEach((o) => (o.textContent = "50"));
    inputs.forEach((input) => setFieldError(input, ""));
    showError("");
    results.hidden = true;
  });
})();
