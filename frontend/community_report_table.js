function fmt(n) {
  return Number(n || 0).toLocaleString("vi-VN");
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[c]));
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#96;");
}

let currentReportRows = [];

const POST_EXPORT_FIELDS = [
  ["stt", "STT"], ["post_id", "Post ID"], ["group_id", "Group ID"], ["group_name", "Tên Group"],
  ["link_post", "Link post"], ["caption", "Caption"], ["reaction_positive_group", "Reaction tích cực"],
  ["reaction_sad", "Sad"], ["reaction_angry", "Angry"], ["comment", "Tổng comment"],
  ["comment_reaction", "Comment reactions"], ["sub_loai", "Sub-loại"],
  ["sentiment_positive", "Tích cực"], ["sentiment_neutral", "Trung lập"],
  ["sentiment_negative", "Tiêu cực"], ["sentiment_spam", "Spam"], ["sentiment_total", "Total sentiment"],
  ["prudential_mention_count", "Số comment mention Pru"], ["prudential_mentions", "Comment mention Pru"],
  ["negative_prudential_count", "Số negative comment của Pru"],
  ["negative_prudential_mentions", "Comment tiêu cực mention Pru"],
  ["negative_ratio", "Tỷ lệ tiêu cực/(tích cực + trung lập)"], ["negative_ratio_formula", "Công thức tỷ lệ"],
  ["seeding_recommendation", "Đề xuất seeding"], ["pillar", "Trụ cột"],
  ["crisis_comments", "Crisis comments"], ["risk_score", "Risk score"]
];

const COMMENT_EXPORT_FIELDS = [
  ["post_id", "Post ID"], ["group_id", "Group ID"], ["group_name", "Tên Group"],
  ["link_post", "Link post"], ["caption", "Caption post"], ["comment_id", "Comment ID"],
  ["content", "Comment"], ["author", "Tác giả"], ["date", "Ngày comment"], ["permalink", "Link comment"],
  ["sub_loai", "Sub-loại"], ["positive", "Tích cực"], ["neutral", "Trung lập"],
  ["negative", "Tiêu cực"], ["spam", "Spam"], ["mentions_prudential", "Mention Pru"],
  ["negative_prudential_mention", "Mention Pru tiêu cực"],
  ["post_negative_ratio", "Tỷ lệ tiêu cực level post"],
  ["post_seeding_recommendation", "Đề xuất seeding level post"]
];

const DEFAULT_POST_FIELDS = new Set(["stt", "group_name", "link_post", "caption", "comment", "sub_loai",
  "sentiment_positive", "sentiment_neutral", "sentiment_negative", "sentiment_spam",
  "prudential_mention_count", "negative_ratio", "seeding_recommendation"]);
const DEFAULT_COMMENT_FIELDS = new Set(["post_id", "group_name", "link_post", "comment_id", "content", "author",
  "date", "permalink", "sub_loai", "positive", "neutral", "negative", "spam",
  "mentions_prudential", "negative_prudential_mention", "post_seeding_recommendation"]);

function exportFieldDefinitions() {
  return document.getElementById("exportMode")?.value === "comments" ? COMMENT_EXPORT_FIELDS : POST_EXPORT_FIELDS;
}

function renderExportFields() {
  const mode = document.getElementById("exportMode")?.value || "posts";
  const defaults = mode === "comments" ? DEFAULT_COMMENT_FIELDS : DEFAULT_POST_FIELDS;
  const root = document.getElementById("exportFields");
  if (!root) return;
  root.innerHTML = exportFieldDefinitions().map(([key, label]) =>
    `<label><input type="checkbox" value="${escapeAttr(key)}" ${defaults.has(key) ? "checked" : ""}> ${escapeHtml(label)}</label>`
  ).join("");
  document.getElementById("exportStatus").textContent = "";
}

function toggleAllExportFields(checked) {
  document.querySelectorAll("#exportFields input[type=checkbox]").forEach(el => { el.checked = checked; });
}

function csvCell(value) {
  if (Array.isArray(value)) value = value.join(" | ");
  if (value === true) value = "X";
  if (value === false || value == null) value = "";
  const cell = String(value).replace(/\r?\n/g, " ");
  return `"${cell.replace(/"/g, '""')}"`;
}

function commentExportRows() {
  return currentReportRows.flatMap(post => (post.comment_details || []).map(comment => ({
    post_id: post.post_id, group_id: post.group_id, group_name: post.group_name,
    link_post: post.link_post, caption: post.caption, ...comment,
    post_negative_ratio: post.negative_ratio,
    post_seeding_recommendation: post.seeding_recommendation
  })));
}

function downloadCsv(rows, selected, definitions, filename) {
  const csv = [selected.map(key => csvCell(definitions.get(key) || key)).join(","),
    ...rows.map(row => selected.map(key => csvCell(row[key])).join(","))].join("\r\n");
  const blob = new Blob(["\ufeff", csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function exportSelectedCsv() {
  const status = document.getElementById("exportStatus");
  const selected = [...document.querySelectorAll("#exportFields input:checked")].map(el => el.value);
  if (!selected.length) {
    status.textContent = "Vui lòng chọn ít nhất một trường cần xuất.";
    return;
  }
  const mode = document.getElementById("exportMode")?.value || "posts";
  const labels = new Map(exportFieldDefinitions());
  const rows = mode === "comments" ? commentExportRows() : currentReportRows;
  if (!rows.length) {
    status.textContent = "Không có dữ liệu trong bộ lọc hiện tại để xuất.";
    return;
  }
  downloadCsv(rows, selected, labels,
    `prudential_${mode}_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.csv`);
  status.textContent = `Đã xuất ${fmt(rows.length)} dòng với ${fmt(selected.length)} trường.`;
}

function params() {
  const query = new URLSearchParams({
    brand_id: "prudential",
    groups: document.getElementById("groupFilter")?.value || "all",
    sentiments: document.getElementById("sentimentFilter")?.value || "all",
    limit: document.getElementById("limitRows")?.value || "500"
  });
  const start = document.getElementById("startDate")?.value || "";
  const end = document.getElementById("endDate")?.value || "";
  if (start) query.set("start", start);
  if (end) query.set("end", end);
  return query.toString();
}

async function loadTable() {
  const status = document.getElementById("statusText");
  if (status) status.textContent = "Loading...";
  const data = await fetch(`/api/community/prudential_table_report?${params()}`, { cache: "no-store" }).then(r => r.json());
  currentReportRows = data.rows || [];
  renderFilters(data.meta || {});
  renderSummary(data.rows || []);
  renderRows(data.rows || []);
  if (status) status.textContent = `${fmt((data.rows || []).length)} post rows`;
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

function renderSummary(rows) {
  const urgent = rows.filter(r => r.seeding_recommendation === "Urgent").length;
  const following = rows.filter(r => r.seeding_recommendation === "Following").length;
  const negative = rows.reduce((sum, r) => sum + Number(r.sentiment_negative || 0), 0);
  const negativePru = rows.reduce((sum, r) => sum + Number(r.negative_prudential_count || 0), 0);
  const crisis = rows.reduce((sum, r) => sum + Number(r.crisis_comments || 0), 0);
  document.getElementById("m-rows").textContent = fmt(rows.length);
  document.getElementById("m-urgent").textContent = fmt(urgent);
  document.getElementById("m-following").textContent = fmt(following);
  document.getElementById("m-negative").textContent = fmt(negative);
  document.getElementById("m-negative-pru").textContent = fmt(negativePru);
  document.getElementById("m-crisis").textContent = fmt(crisis);
}

function renderRows(rows) {
  const tbody = document.getElementById("reportRows");
  if (!tbody) return;
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="18" class="empty">No rows in selected scope</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map((row, index) => {
    const link = row.link_post
      ? `<a href="${escapeAttr(row.link_post)}" target="_blank">Open post</a>`
      : "-";
    const pruMentions = (row.prudential_mentions || []).length
      ? `<div><b>${fmt(row.prudential_mention_count)}</b> comment</div><ul>${row.prudential_mentions.slice(0, 3).map(x => `<li>${escapeHtml(x)}</li>`).join("")}</ul>`
      : "-";
    const mentions = (row.negative_prudential_mentions || []).length
      ? `<ul>${row.negative_prudential_mentions.map(x => `<li>${escapeHtml(x)}</li>`).join("")}</ul>`
      : "-";
    const recClass = String(row.seeding_recommendation || "").toLowerCase();
    const details = (row.comment_details || []).map(comment => {
      const commentLink = comment.permalink
        ? `<a href="${escapeAttr(comment.permalink)}" target="_blank">${escapeHtml(comment.content || "-")}</a>`
        : escapeHtml(comment.content || "-");
      return `<tr>
        <td class="comment-copy">${commentLink}<div class="muted">${escapeHtml(comment.author || "-")} · ${escapeHtml(comment.date || "")}</div></td>
        <td>${escapeHtml(comment.sub_loai || "-")}</td>
        <td class="check pos">${comment.positive ? "✓" : ""}</td>
        <td class="check neu">${comment.neutral ? "✓" : ""}</td>
        <td class="check neg">${comment.negative ? "✓" : ""}</td>
        <td class="check spam">${comment.spam ? "✓" : ""}</td>
        <td class="check">${comment.mentions_prudential ? "✓" : ""}</td>
        <td class="check neg">${comment.negative_prudential_mention ? "✓" : ""}</td>
      </tr>`;
    }).join("");
    return `<tr class="post-row">
      <td class="num">${fmt(row.stt)}</td>
      <td>${escapeHtml(row.group_name || row.group_id || "-")}</td>
      <td>${link}</td>
      <td class="caption">${escapeHtml(row.caption || "-")}</td>
      <td class="num">${fmt(row.reaction_positive_group)}</td>
      <td class="num">${fmt(row.reaction_sad)}</td>
      <td class="num angry">${fmt(row.reaction_angry)}</td>
      <td class="num">${fmt(row.comment)}<div class="muted">${fmt(row.comment_reaction)} reactions</div><button class="comment-toggle" type="button" data-comment-target="comments-${index}">Xem comment</button></td>
      <td>${escapeHtml(row.sub_loai || "-")}</td>
      <td class="num pos">${fmt(row.sentiment_positive)}<div class="muted">Total sentiment: ${fmt(row.sentiment_total)}</div></td>
      <td class="num neu">${fmt(row.sentiment_neutral)}</td>
      <td class="num neg">${fmt(row.sentiment_negative)}</td>
      <td class="num spam">${fmt(row.sentiment_spam)}</td>
      <td class="mention">${pruMentions}</td>
      <td class="mention">${mentions}</td>
      <td class="num"><b>${Number(row.negative_ratio || 0).toFixed(2)}</b><div class="muted">${escapeHtml(row.negative_ratio_formula || "")}</div></td>
      <td><span class="badge ${recClass}">${escapeHtml(row.seeding_recommendation || "Normal")}</span></td>
      <td class="manual">${escapeHtml(row.pillar || "")}</td>
    </tr>
    <tr id="comments-${index}" class="comment-detail-row">
      <td colspan="18">
        <div class="comment-detail-wrap">
          <table class="comment-detail-table">
            <thead><tr><th>Comment</th><th>Sub-loại</th><th>Tích cực</th><th>Trung lập</th><th>Tiêu cực</th><th>Spam</th><th>Mention Pru</th><th>Mention Pru tiêu cực</th></tr></thead>
            <tbody>${details || '<tr><td colspan="8" class="empty">Không có comment đã phân loại</td></tr>'}</tbody>
          </table>
        </div>
      </td>
    </tr>`;
  }).join("");
}

document.getElementById("reportRows")?.addEventListener("click", event => {
  const button = event.target.closest("[data-comment-target]");
  if (!button) return;
  const row = document.getElementById(button.dataset.commentTarget);
  if (!row) return;
  const opened = row.classList.toggle("open");
  button.textContent = opened ? "Ẩn comment" : "Xem comment";
});

renderExportFields();
loadTable();
