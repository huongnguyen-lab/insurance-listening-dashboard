const charts = {};
const SENTIMENT = {
  tich_cuc: { label: "Positive", color: "#15803d" },
  trung_lap: { label: "Neutral", color: "#64748b" },
  tieu_cuc: { label: "Negative", color: "#dc2626" },
  spam: { label: "Spam", color: "#111827" }
};

function chart(id) {
  if (charts[id]) charts[id].dispose();
  charts[id] = echarts.init(document.getElementById(id));
  return charts[id];
}

function fmt(n) {
  return Number(n || 0).toLocaleString("vi-VN");
}

function setMetric(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function initDates() {
  // Keep date filters empty by default so the dashboard shows the current loaded dataset.
  // The crawl files can be older than today's machine date, especially during model tests.
}

function params() {
  const group = document.getElementById("groupFilter")?.value || "all";
  const sentiment = document.getElementById("sentimentFilter")?.value || "all";
  const start = document.getElementById("startDate")?.value || "";
  const end = document.getElementById("endDate")?.value || "";
  const query = new URLSearchParams({ brand_id: "prudential", groups: group, sentiments: sentiment });
  if (start) query.set("start", start);
  if (end) query.set("end", end);
  return query.toString();
}

async function loadReport() {
  const data = await fetch(`/api/community/prudential_report?${params()}`, { cache: "no-store" }).then(r => r.json());
  renderFilters(data.meta || {});
  renderMetrics(data.metrics || {});
  renderNotice(data.meta || {});
  renderSentiment(data.sentiment_counts || {});
  renderTimeline(data.timeline || []);
  renderGroupCharts(data.group_breakdown || []);
  renderPosts(data.top_posts || []);
}

function renderFilters(meta) {
  const sel = document.getElementById("groupFilter");
  if (!sel || sel.dataset.loaded) return;
  for (const group of meta.available_groups || []) {
    const opt = document.createElement("option");
    opt.value = group.group_id;
    opt.textContent = group.group_name || group.group_id;
    sel.appendChild(opt);
  }
  sel.dataset.loaded = "1";
}

function renderMetrics(m) {
  setMetric("m-posts", fmt(m.posts));
  setMetric("m-comments", fmt(m.comments));
  setMetric("m-labeled", fmt(m.labeled_comments));
  setMetric("m-scoped", fmt(m.scoped_comments));
  setMetric("m-cpp", Number(m.avg_comments_per_post || 0).toFixed(1));
  setMetric("m-post-reactions", fmt(m.post_reactions));
  setMetric("m-comment-reactions", fmt(m.comment_reactions));
  setMetric("m-neg-ratio", `${m.negative_ratio || 0}%`);
  setMetric("m-crisis", fmt(m.crisis_comments));
}

function renderNotice(meta) {
  const el = document.getElementById("scopeNotice");
  if (!el) return;
  el.textContent = meta.brand_detection_ready
    ? "Scope: Prudential-mentioned comments."
    : "Scope: all labeled comments until Prudential brand detection has enough data.";
  el.className = meta.brand_detection_ready ? "notice ok" : "notice warn";
}

function renderSentiment(counts) {
  const rows = ["tich_cuc", "trung_lap", "tieu_cuc", "spam"].map(key => ({
    name: SENTIMENT[key].label,
    value: counts[key] || 0,
    itemStyle: { color: SENTIMENT[key].color }
  }));
  chart("c-sentiment").setOption({
    tooltip: { trigger: "item" },
    legend: { bottom: 0 },
    series: [{
      type: "pie",
      radius: ["45%", "72%"],
      center: ["50%", "43%"],
      data: rows,
      label: { formatter: "{b}\n{d}%" }
    }]
  });
}

function renderTimeline(rows) {
  chart("c-timeline").setOption({
    tooltip: { trigger: "axis" },
    legend: { bottom: 0 },
    grid: { left: 45, right: 20, top: 20, bottom: 55 },
    xAxis: { type: "category", data: rows.map(r => r.period), axisLabel: { hideOverlap: true } },
    yAxis: { type: "value" },
    series: [
      { name: "Positive", type: "bar", stack: "sent", data: rows.map(r => r.positive), itemStyle: { color: SENTIMENT.tich_cuc.color } },
      { name: "Neutral", type: "bar", stack: "sent", data: rows.map(r => r.neutral), itemStyle: { color: SENTIMENT.trung_lap.color } },
      { name: "Negative", type: "bar", stack: "sent", data: rows.map(r => r.negative), itemStyle: { color: SENTIMENT.tieu_cuc.color } }
    ]
  });
}

function topGroups(groups, metric) {
  return [...groups].sort((a, b) => (b[metric] || 0) - (a[metric] || 0)).slice(0, 12).reverse();
}

function renderGroupCharts(groups) {
  const commentGroups = topGroups(groups, "scoped_comments");
  chart("c-group-comments").setOption({
    tooltip: { trigger: "axis" },
    grid: { left: 155, right: 28, top: 16, bottom: 30 },
    xAxis: { type: "value" },
    yAxis: { type: "category", data: commentGroups.map(g => g.group_name || g.group_id) },
    series: [{ type: "bar", data: commentGroups.map(g => g.scoped_comments), itemStyle: { color: "#2563eb", borderRadius: [0, 5, 5, 0] }, label: { show: true, position: "right" } }]
  });

  const reactionGroups = topGroups(groups, "post_reactions");
  chart("c-group-reactions").setOption({
    tooltip: { trigger: "axis" },
    grid: { left: 155, right: 28, top: 16, bottom: 30 },
    xAxis: { type: "value" },
    yAxis: { type: "category", data: reactionGroups.map(g => g.group_name || g.group_id) },
    series: [{ type: "bar", data: reactionGroups.map(g => g.post_reactions), itemStyle: { color: "#d97706", borderRadius: [0, 5, 5, 0] }, label: { show: true, position: "right" } }]
  });

  const sentGroups = topGroups(groups, "scoped_comments");
  chart("c-group-sentiment").setOption({
    tooltip: { trigger: "axis" },
    legend: { bottom: 0 },
    grid: { left: 155, right: 20, top: 20, bottom: 55 },
    xAxis: { type: "value" },
    yAxis: { type: "category", data: sentGroups.map(g => g.group_name || g.group_id) },
    series: [
      { name: "Positive", type: "bar", stack: "sent", data: sentGroups.map(g => g.positive), itemStyle: { color: SENTIMENT.tich_cuc.color } },
      { name: "Neutral", type: "bar", stack: "sent", data: sentGroups.map(g => g.neutral), itemStyle: { color: SENTIMENT.trung_lap.color } },
      { name: "Negative", type: "bar", stack: "sent", data: sentGroups.map(g => g.negative), itemStyle: { color: SENTIMENT.tieu_cuc.color } }
    ]
  });

  const workGroups = topGroups(groups, "comments");
  chart("c-post-workload").setOption({
    tooltip: { trigger: "axis" },
    legend: { bottom: 0 },
    grid: { left: 155, right: 28, top: 20, bottom: 55 },
    xAxis: { type: "value" },
    yAxis: { type: "category", data: workGroups.map(g => g.group_name || g.group_id) },
    series: [
      { name: "Posts", type: "bar", data: workGroups.map(g => g.posts), itemStyle: { color: "#0891b2" } },
      { name: "Comments", type: "bar", data: workGroups.map(g => g.comments), itemStyle: { color: "#2563eb" } }
    ]
  });
}

function renderPosts(posts) {
  const tbody = document.getElementById("postRows");
  if (!tbody) return;
  tbody.innerHTML = posts.length ? posts.map(p => {
    const sent = `<span class="pill pos">${fmt(p.positive)}</span><span class="pill neu">${fmt(p.neutral)}</span><span class="pill neg">${fmt(p.negative)}</span>`;
    const samples = (p.samples || []).length ? `<div class="sample-line">${p.samples.map(s => escapeHtml(s)).join(" | ")}</div>` : "";
    return `<tr>
      <td>${escapeHtml(p.group_name || p.group_id || "-")}</td>
      <td><a href="${escapeAttr(p.post_url || "#")}" target="_blank">${escapeHtml((p.caption || p.post_id || "-").slice(0, 120))}</a>${samples}</td>
      <td>${fmt(p.comments)} <span class="muted">/ ${fmt(p.scoped_comments)}</span></td>
      <td>${fmt(p.post_reactions)} <span class="muted">post</span><br>${fmt(p.comment_reactions)} <span class="muted">comment</span></td>
      <td>${sent}<div class="muted">neg ${p.negative_ratio || 0}%</div></td>
      <td><strong>${fmt(p.risk_score)}</strong><div class="muted">crisis ${fmt(p.crisis)}</div></td>
    </tr>`;
  }).join("") : `<tr><td colspan="6" class="empty-cell">No posts in selected scope</td></tr>`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[c]));
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#96;");
}

window.addEventListener("resize", () => Object.values(charts).forEach(c => c.resize()));
initDates();
loadReport();

