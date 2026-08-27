"use strict";

const documentState = { documents: [], current: null };
const docEl = {};

document.addEventListener("DOMContentLoaded", bootstrapDocuments);

async function bootstrapDocuments() {
  ["document-list", "document-content", "document-path", "raw-document-link", "document-outline", "document-menu"]
    .forEach((id) => { docEl[toCamelCase(id)] = document.getElementById(id); });
  try {
    const result = await fetchDocumentJson("/api/docs");
    documentState.documents = result.documents;
    if (!documentState.documents.length) throw new Error("No documents found");
    renderDocumentList();
    const requested = new URLSearchParams(location.search).get("doc");
    const initial = documentState.documents.some((item) => item.slug === requested)
      ? requested
      : documentState.documents.find((item) => item.slug === "overview")?.slug || documentState.documents[0].slug;
    await loadDocument(initial, false);
    docEl.documentMenu.addEventListener("click", toggleDocumentMenu);
    addEventListener("popstate", () => {
      const slug = new URLSearchParams(location.search).get("doc") || initial;
      if (slug !== documentState.current) loadDocument(slug, false);
    });
  } catch (error) {
    console.error(error);
    docEl.documentContent.replaceChildren(document.getElementById("document-error-template").content.cloneNode(true));
  }
}

function toCamelCase(value) {
  return value.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

async function fetchDocumentJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

function renderDocumentList() {
  docEl.documentList.replaceChildren();
  documentState.documents.forEach((item) => {
    const link = document.createElement("a");
    link.href = `?doc=${encodeURIComponent(item.slug)}`;
    link.dataset.slug = item.slug;
    link.innerHTML = `<span>${escapeHtml(item.title)}</span><small>${escapeHtml(item.slug)}.md</small>`;
    link.addEventListener("click", (event) => {
      event.preventDefault();
      loadDocument(item.slug, true);
      closeDocumentMenu();
    });
    docEl.documentList.append(link);
  });
}

async function loadDocument(slug, updateHistory) {
  const item = documentState.documents.find((candidate) => candidate.slug === slug);
  if (!item) throw new Error(`Unknown document: ${slug}`);
  docEl.documentContent.innerHTML = '<div class="document-loading">正在载入文档…</div>';
  const result = await fetchDocumentJson(`/api/docs/${encodeURIComponent(slug)}`);
  documentState.current = slug;
  docEl.documentPath.textContent = result.path;
  docEl.rawDocumentLink.href = result.path;
  docEl.documentContent.innerHTML = renderMarkdown(result.content);
  window.scrollTo({ top: 0, behavior: "instant" });
  document.querySelectorAll(".document-list a").forEach((link) => {
    link.classList.toggle("is-active", link.dataset.slug === slug);
  });
  renderOutline();
  document.title = `${item.title} · Qwen3.8 Flash Next Explorer`;
  if (updateHistory) history.pushState({}, "", `?doc=${encodeURIComponent(slug)}`);
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let paragraph = [];
  let listType = null;
  let inCode = false;
  let codeLanguage = "";
  let codeLines = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    html.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph = [];
  };
  const closeList = () => {
    if (!listType) return;
    html.push(`</${listType}>`);
    listType = null;
  };
  const flushCode = () => {
    html.push(`<pre><code${codeLanguage ? ` class="language-${escapeHtml(codeLanguage)}"` : ""}>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
    codeLines = [];
    codeLanguage = "";
  };

  for (const line of lines) {
    const fence = line.match(/^```\s*([\w-]*)/);
    if (fence) {
      if (inCode) {
        flushCode();
        inCode = false;
      } else {
        flushParagraph(); closeList();
        inCode = true;
        codeLanguage = fence[1];
      }
      continue;
    }
    if (inCode) { codeLines.push(line); continue; }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      flushParagraph(); closeList();
      const level = heading[1].length;
      const text = heading[2].trim();
      const id = headingId(text);
      html.push(`<h${level} id="${id}">${inlineMarkdown(text)}<a class="heading-anchor" href="#${id}" aria-label="链接到本节">#</a></h${level}>`);
      continue;
    }

    const unordered = line.match(/^\s*[-*]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unordered || ordered) {
      flushParagraph();
      const nextType = ordered ? "ol" : "ul";
      if (listType !== nextType) { closeList(); listType = nextType; html.push(`<${listType}>`); }
      html.push(`<li>${inlineMarkdown((unordered || ordered)[1])}</li>`);
      continue;
    }

    const quote = line.match(/^>\s?(.*)$/);
    if (quote) {
      flushParagraph(); closeList();
      html.push(`<blockquote>${inlineMarkdown(quote[1])}</blockquote>`);
      continue;
    }
    if (/^---+$/.test(line.trim())) {
      flushParagraph(); closeList(); html.push("<hr>"); continue;
    }
    if (!line.trim()) {
      flushParagraph(); closeList(); continue;
    }
    paragraph.push(line.trim());
  }
  flushParagraph(); closeList();
  if (inCode) flushCode();
  return html.join("\n");
}

function inlineMarkdown(value) {
  let output = escapeHtml(value);
  output = output.replace(/`([^`]+)`/g, "<code>$1</code>");
  output = output.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  output = output.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, href) => {
    let safeHref = /^(https?:\/\/|#|\.\.\/|\.\/|[a-zA-Z0-9_-]+\.md)/.test(href) ? href : "#";
    if (/^[a-zA-Z0-9_-]+\.md$/.test(safeHref)) safeHref = `?doc=${safeHref.slice(0, -3)}`;
    const external = /^https?:\/\//.test(safeHref) ? ' target="_blank" rel="noreferrer"' : "";
    return `<a href="${escapeHtml(safeHref)}"${external}>${label}</a>`;
  });
  return output;
}

function headingId(value) {
  return value.toLowerCase().replace(/`/g, "").replace(/[^a-z0-9\u4e00-\u9fff]+/g, "-").replace(/^-|-$/g, "") || "section";
}

function renderOutline() {
  docEl.documentOutline.replaceChildren();
  docEl.documentContent.querySelectorAll("h2, h3").forEach((heading) => {
    const link = document.createElement("a");
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent.replace(/#$/, "");
    link.className = heading.tagName === "H3" ? "outline-level-3" : "";
    docEl.documentOutline.append(link);
  });
}

function toggleDocumentMenu() {
  const open = document.body.classList.toggle("document-menu-open");
  docEl.documentMenu.setAttribute("aria-expanded", String(open));
}

function closeDocumentMenu() {
  document.body.classList.remove("document-menu-open");
  docEl.documentMenu.setAttribute("aria-expanded", "false");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  })[character]);
}
