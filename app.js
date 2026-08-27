"use strict";

const DATA_URLS = {
  spec: "data/model/qwen3.8-flash-next.model-spec.json",
  weightProfiles: "data/quantization/weight-profiles.json",
  kvProfiles: "data/quantization/kv-cache-profiles.json",
  scenario: "data/scenarios/default.json"
};

const FIXED_BYTES = {
  fixed_fp32: 4,
  fixed_bf16: 2,
  fixed_int32: 4,
  fixed_int64: 8
};

const COMPONENT_WEIGHT_FORMATS = {
  bf16: { type: "fixed_bpw", bpw: 16 },
  fp8: { type: "fp8_block", block_rows: 128, block_cols: 128, scale_bits: 8 },
  "w4-effective-425": { type: "effective_bpw", bpw: 4.25 },
  "w4-raw": { type: "effective_bpw", bpw: 4 }
};

const COMPONENT_QUANTIZATION_IDS = new Set(["inherit", ...Object.keys(COMPONENT_WEIGHT_FORMATS)]);

const EXPANDABLE_MODULES = new Set(["ple", "gr_attn", "gdn", "qsa", "gr_mlp", "moe"]);

const state = {
  spec: null,
  weightProfiles: null,
  kvProfiles: null,
  scenario: null,
  selectedLayerKind: "main",
  selectedLayerIndex: 0,
  selectedNodeId: null,
  expandedModuleId: null,
  pendingClickTimer: null,
  analysis: null
};

const el = {};
const SVG_NS = "http://www.w3.org/2000/svg";

document.addEventListener("DOMContentLoaded", bootstrap);

async function bootstrap() {
  cacheElements();
  try {
    const [spec, weightProfiles, kvProfiles, scenario] = await Promise.all([
      fetchJson(DATA_URLS.spec),
      fetchJson(DATA_URLS.weightProfiles),
      fetchJson(DATA_URLS.kvProfiles),
      fetchJson(DATA_URLS.scenario)
    ]);
    state.spec = spec;
    state.weightProfiles = weightProfiles;
    state.kvProfiles = kvProfiles;
    state.scenario = { ...scenario };
    populateControls();
    bindEvents();
    renderAll();
    el.loadingScreen.hidden = true;
    el.appShell.hidden = false;
    await refreshScenarioList();
  } catch (error) {
    console.error(error);
    el.loadingScreen.remove();
    document.body.append(el.errorTemplate.content.cloneNode(true));
  }
}

function cacheElements() {
  const ids = [
    "loading-screen", "app-shell", "source-chip", "weight-profile", "weight-profile-description",
    "kv-profile", "kv-profile-description", "context-length", "batch-size", "include-ngram", "ngram-profile",
    "include-vision", "vision-profile", "include-mtp", "mtp-profile",
    "scenario-name", "save-local", "export-scenario", "export-analysis", "import-scenario",
    "saved-scenario-select", "load-scenario", "status-message", "metric-params", "metric-params-note",
    "metric-weight", "metric-weight-note", "metric-kv", "metric-kv-note", "metric-total", "model-facts",
    "layer-list", "aux-layer-section", "aux-layer-list", "selected-layer-number", "selected-layer-title",
    "selected-layer-subtitle", "layer-summary", "operator-graph", "graph-viewport", "operator-inspector",
    "collapse-module", "graph-context", "error-template"
  ];
  ids.forEach((id) => {
    el[toCamel(id)] = document.getElementById(id);
  });
}

function toCamel(value) {
  return value.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  if (!response.ok) {
    throw new Error(`${url}: HTTP ${response.status}`);
  }
  return response.json();
}

function populateControls() {
  fillProfileSelect(el.weightProfile, state.weightProfiles.profiles, state.scenario.weight_quantization);
  fillProfileSelect(el.kvProfile, state.kvProfiles.profiles, state.scenario.kv_cache_quantization);
  el.contextLength.value = String(state.scenario.context_length);
  el.batchSize.value = String(state.scenario.batch_size);
  el.includeNgram.checked = state.scenario.include_ngram;
  el.ngramProfile.value = state.scenario.ngram_quantization;
  el.includeVision.checked = state.scenario.include_vision;
  el.visionProfile.value = state.scenario.vision_quantization;
  el.includeMtp.checked = state.scenario.include_mtp;
  el.mtpProfile.value = state.scenario.mtp_quantization;
  el.scenarioName.value = state.scenario.name;
  syncComponentControls();
  updateProfileDescriptions();

  const revision = state.spec.source.revision;
  el.sourceChip.textContent = `official @ ${revision.slice(0, 10)} · schema ${state.spec.schema_version}`;
  el.sourceChip.title = `Hugging Face revision ${revision}`;
  el.sourceChip.href = `${state.spec.source.repository}/tree/${revision}`;
}

function fillProfileSelect(select, profiles, selectedId) {
  select.replaceChildren();
  profiles.forEach((profile) => {
    const option = document.createElement("option");
    option.value = profile.id;
    option.textContent = profile.display_name;
    option.selected = profile.id === selectedId;
    select.append(option);
  });
}

function bindEvents() {
  el.weightProfile.addEventListener("change", () => {
    state.scenario.weight_quantization = el.weightProfile.value;
    updateProfileDescriptions();
    renderAll();
  });
  el.kvProfile.addEventListener("change", () => {
    state.scenario.kv_cache_quantization = el.kvProfile.value;
    updateProfileDescriptions();
    renderAll();
  });
  el.contextLength.addEventListener("change", () => {
    state.scenario.context_length = Number(el.contextLength.value);
    renderAll();
  });
  const updateBatchSize = () => {
    const value = clamp(Math.round(Number(el.batchSize.value) || 1), 1, 1024);
    el.batchSize.value = String(value);
    state.scenario.batch_size = value;
    renderAll();
  };
  el.batchSize.addEventListener("input", updateBatchSize);
  el.batchSize.addEventListener("change", updateBatchSize);
  el.includeNgram.addEventListener("change", () => {
    state.scenario.include_ngram = el.includeNgram.checked;
    syncComponentControls();
    renderAll();
  });
  el.ngramProfile.addEventListener("change", () => {
    state.scenario.ngram_quantization = el.ngramProfile.value;
    renderAll();
  });
  el.includeVision.addEventListener("change", () => {
    state.scenario.include_vision = el.includeVision.checked;
    syncComponentControls();
    renderAll();
  });
  el.visionProfile.addEventListener("change", () => {
    state.scenario.vision_quantization = el.visionProfile.value;
    renderAll();
  });
  el.includeMtp.addEventListener("change", () => {
    state.scenario.include_mtp = el.includeMtp.checked;
    syncComponentControls();
    if (!state.scenario.include_mtp && state.selectedLayerKind === "aux") {
      state.selectedLayerKind = "main";
      state.selectedLayerIndex = 0;
      state.selectedNodeId = null;
    }
    renderAll();
  });
  el.mtpProfile.addEventListener("change", () => {
    state.scenario.mtp_quantization = el.mtpProfile.value;
    renderAll();
  });
  el.collapseModule.addEventListener("click", () => {
    state.expandedModuleId = null;
    state.selectedNodeId = null;
    renderGraph();
    renderInspector();
  });
  el.scenarioName.addEventListener("input", () => {
    state.scenario.name = el.scenarioName.value.trim() || "untitled";
  });
  el.saveLocal.addEventListener("click", saveLocal);
  el.exportScenario.addEventListener("click", exportScenario);
  el.exportAnalysis.addEventListener("click", exportAnalysis);
  el.importScenario.addEventListener("change", importScenario);
  el.loadScenario.addEventListener("click", loadSavedScenario);
}

function updateProfileDescriptions() {
  el.weightProfileDescription.textContent = currentWeightProfile().description;
  el.kvProfileDescription.textContent = currentKvProfile().description;
}

function syncComponentControls() {
  el.ngramProfile.disabled = !state.scenario.include_ngram;
  el.visionProfile.disabled = !state.scenario.include_vision;
  el.mtpProfile.disabled = !state.scenario.include_mtp;
}

function currentWeightProfile() {
  return state.weightProfiles.profiles.find((profile) => profile.id === state.scenario.weight_quantization);
}

function currentKvProfile() {
  return state.kvProfiles.profiles.find((profile) => profile.id === state.scenario.kv_cache_quantization);
}

function renderAll() {
  state.analysis = calculateAnalysis();
  renderMetrics();
  renderModelFacts();
  renderLayers();
  renderSelectedLayerHeader();
  renderGraph();
  renderInspector();
}

function calculateAnalysis() {
  const profile = currentWeightProfile();
  const kvProfile = currentKvProfile();
  const includedLayers = state.spec.layers.map((layer) => ({ ...layer, resultKey: `main:${layer.index}` }));
  if (state.scenario.include_mtp) {
    includedLayers.push(...state.spec.auxiliary_layers.map((layer) => ({ ...layer, resultKey: `aux:${layer.index}` })));
  }

  const globalGroups = state.spec.global_parameter_groups.filter(groupIncluded);
  const globalWeightBytes = globalGroups.reduce(
    (total, group) => total + componentWeightGroupBytes(group),
    0
  );
  const layerResults = includedLayers.map((layer) => {
    const parameterGroups = layer.parameter_groups.filter(groupIncluded);
    const weightBytes = parameterGroups.reduce(
      (total, group) => total + componentWeightGroupBytes(group, layer),
      0
    );
    const kv = layerKvBytes(layer, kvProfile);
    return {
      resultKey: layer.resultKey,
      index: layer.index,
      label: layer.label,
      kind: layer.kind,
      logical_params: parameterGroups.reduce((total, group) => total + group.logical_params, 0),
      weight_bytes: weightBytes,
      kv_cache: kv
    };
  });
  const totalWeightBytes = globalWeightBytes + sum(layerResults, "weight_bytes");
  const totalKvBytes = layerResults.reduce((total, layer) => total + layer.kv_cache.total_bytes, 0);
  const totalParams = globalGroups.reduce((total, group) => total + group.logical_params, 0) + sum(layerResults, "logical_params");
  layerResults.forEach((layer) => {
    layer.weight_ratio = layer.weight_bytes / totalWeightBytes;
  });

  return {
    schema_version: "1.1.0",
    model_id: state.spec.model_id,
    source_revision: state.spec.source.revision,
    generated_at: new Date().toISOString(),
    scenario: { ...state.scenario },
    weight_profile: profile.id,
    kv_cache_profile: kvProfile.id,
    formula_ids: [...state.spec.formula_ids],
    totals: {
      logical_params: totalParams,
      global_weight_bytes: globalWeightBytes,
      weight_bytes: totalWeightBytes,
      kv_cache_bytes: totalKvBytes,
      weight_plus_kv_bytes: totalWeightBytes + totalKvBytes
    },
    layers: layerResults
  };
}

function groupIncluded(group) {
  if (group.optional_component === "ngram") return state.scenario.include_ngram;
  if (group.optional_component === "vision") return state.scenario.include_vision;
  return true;
}

function componentQuantization(group, layer = null) {
  if (group.optional_component === "ngram") return state.scenario.ngram_quantization;
  if (group.optional_component === "vision") return state.scenario.vision_quantization;
  if (layer?.kind === "mtp") return state.scenario.mtp_quantization;
  return "inherit";
}

function componentWeightGroupBytes(group, layer = null) {
  const override = componentQuantization(group, layer);
  if (override === "inherit" || Object.hasOwn(FIXED_BYTES, group.storage_class)) {
    return weightGroupBytes(group, currentWeightProfile());
  }
  const format = COMPONENT_WEIGHT_FORMATS[override];
  if (!format) throw new Error(`Unknown component quantization ${override}`);
  return weightGroupBytes(group, { formats: { [group.policy_group]: format } });
}

function weightGroupBytes(group, profile) {
  if (Object.hasOwn(FIXED_BYTES, group.storage_class)) {
    return group.logical_params * FIXED_BYTES[group.storage_class];
  }
  const format = profile.formats[group.policy_group];
  if (!format) throw new Error(`Missing format for ${group.policy_group}`);
  if (format.type === "fixed_bpw" || format.type === "effective_bpw") {
    return group.logical_params * format.bpw / 8;
  }
  const outFeatures = group.shape[0];
  const inFeatures = group.shape.slice(1).reduce((total, value) => total * value, 1);
  if (format.type === "fp8_block") {
    const weightBytes = outFeatures * inFeatures;
    const scales = Math.ceil(outFeatures / format.block_rows) * Math.ceil(inFeatures / format.block_cols);
    return (weightBytes + scales * format.scale_bits / 8) * group.copies;
  }
  if (format.type === "fp4_block") {
    const weightBytes = outFeatures * inFeatures / 2;
    const scales = outFeatures * Math.ceil(inFeatures / format.block_size);
    return (weightBytes + scales * format.scale_bits / 8) * group.copies;
  }
  throw new Error(`Unsupported format ${format.type}`);
}

function layerKvBytes(layer, kvProfile) {
  const architecture = state.spec.architecture;
  const batch = state.scenario.batch_size;
  const context = state.scenario.context_length;
  const isGdn = layer.attention_type === "linear_attention";
  let mainCacheBytes = 0;
  let indexCacheBytes = 0;
  let recurrentStateBytes = 0;
  let convStateBytes = 0;
  let auxiliaryStateBytes = 0;
  if (isGdn) {
    recurrentStateBytes = batch * architecture.linear_value_heads *
      architecture.linear_head_dim * architecture.linear_head_dim * 4;
    const convDim = 2 * architecture.linear_qk_heads * architecture.linear_head_dim +
      architecture.linear_value_heads * architecture.linear_head_dim;
    convStateBytes = batch * convDim * architecture.linear_conv_kernel * kvProfile.conv_bpw / 8;
  } else {
    const totalKvElements = 2 * architecture.kv_heads * architecture.head_dim;
    const ropeElements = architecture.kv_heads * architecture.rope_head_dim;
    const nopeElements = totalKvElements - ropeElements;
    mainCacheBytes = batch * context *
      (nopeElements * kvProfile.nope_bpw + ropeElements * kvProfile.rope_bpw) / 8;
    indexCacheBytes = batch * context * architecture.index_kv_heads *
      architecture.index_head_dim * kvProfile.index_bpw / 8;
  }
  if (layer.has_ple && state.scenario.include_ngram) {
    auxiliaryStateBytes = batch * architecture.hc_mult * architecture.hidden_size * 9 * kvProfile.conv_bpw / 8 +
      batch * (architecture.ngram_size - 1) * 8;
  }
  return {
    main_cache_bytes: mainCacheBytes,
    index_cache_bytes: indexCacheBytes,
    recurrent_state_bytes: recurrentStateBytes,
    conv_state_bytes: convStateBytes,
    auxiliary_state_bytes: auxiliaryStateBytes,
    total_bytes: mainCacheBytes + indexCacheBytes + recurrentStateBytes + convStateBytes + auxiliaryStateBytes
  };
}

function sum(items, key) {
  return items.reduce((total, item) => total + item[key], 0);
}

function renderMetrics() {
  const totals = state.analysis.totals;
  el.metricParams.textContent = formatParams(totals.logical_params);
  const components = ["48 Blocks"];
  if (state.scenario.include_ngram) components.push("N-gram");
  if (state.scenario.include_vision) components.push("Vision");
  if (state.scenario.include_mtp) components.push("MTP");
  el.metricParamsNote.textContent = components.join(" + ");
  el.metricWeight.textContent = formatBytes(totals.weight_bytes);
  el.metricWeightNote.textContent = `${currentWeightProfile().display_name} · 组件可独立设置`;
  el.metricKv.textContent = formatBytes(totals.kv_cache_bytes);
  el.metricKvNote.textContent = `${formatContext(state.scenario.context_length)} · Batch ${state.scenario.batch_size}`;
  el.metricTotal.textContent = formatBytes(totals.weight_plus_kv_bytes);
}

function renderModelFacts() {
  const arch = state.spec.architecture;
  const facts = [
    ["Hidden", formatInteger(arch.hidden_size)],
    ["Hybrid", `${arch.gdn_layer_count} GDN / ${arch.qsa_layer_count} QSA`],
    ["Experts", `${arch.routed_experts} / Top-${arch.experts_per_token}`],
    ["GR streams", String(arch.hc_mult)],
    ["QSA budget", `${arch.index_token_budget} tok`],
    ["Max context", formatContext(arch.max_context_length)]
  ];
  el.modelFacts.innerHTML = facts.map(([label, value]) => `
    <div class="fact-cell"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>
  `).join("");
}

function renderLayers() {
  const resultMap = new Map(state.analysis.layers.map((result) => [result.resultKey, result]));
  const mainResults = state.spec.layers.map((layer) => resultMap.get(`main:${layer.index}`));
  const maxWeight = Math.max(...mainResults.map((result) => result.weight_bytes));
  el.layerList.replaceChildren();
  state.spec.layers.forEach((layer) => {
    const result = resultMap.get(`main:${layer.index}`);
    el.layerList.append(createLayerButton(layer, result, maxWeight, "main"));
  });

  el.auxLayerSection.hidden = !state.scenario.include_mtp;
  el.auxLayerList.replaceChildren();
  if (state.scenario.include_mtp) {
    state.spec.auxiliary_layers.forEach((layer) => {
      const result = resultMap.get(`aux:${layer.index}`);
      el.auxLayerList.append(createLayerButton(layer, result, maxWeight, "aux"));
    });
  }
}

function createLayerButton(layer, result, maxWeight, kind) {
  const button = document.createElement("button");
  button.type = "button";
  const isGdn = layer.attention_type === "linear_attention";
  button.className = `layer-row ${isGdn ? "is-gdn" : "is-qsa"}${layer.has_ple ? " has-ple" : ""}`;
  const selected = state.selectedLayerKind === kind && state.selectedLayerIndex === layer.index;
  if (selected) button.classList.add("is-selected");
  button.setAttribute("aria-pressed", String(selected));
  const mixerLabel = isGdn ? `GDN${layer.has_ple ? " + PLE" : ""}` : "QSA · 4:1 IDX";
  const indexLabel = kind === "aux" ? "M" : String(layer.index).padStart(2, "0");
  const barWidth = Math.max(4, result.weight_bytes / maxWeight * 100);
  button.innerHTML = `
    <span class="layer-index">${indexLabel}</span>
    <span class="layer-info">
      <span class="layer-topline"><strong>${escapeHtml(mixerLabel)}</strong><span>${formatBytes(result.weight_bytes)}</span></span>
      <span class="layer-bottomline"><span>${isGdn ? "FIXED STATE" : "GLOBAL KV"}</span><span class="mini-track"><i style="width:${barWidth.toFixed(1)}%"></i></span><span>${formatPercent(result.weight_ratio)}</span></span>
    </span>
  `;
  button.addEventListener("click", () => {
    state.selectedLayerKind = kind;
    state.selectedLayerIndex = layer.index;
    state.selectedNodeId = null;
    state.expandedModuleId = null;
    renderLayers();
    renderSelectedLayerHeader();
    renderGraph();
    renderInspector();
  });
  return button;
}

function selectedLayer() {
  const collection = state.selectedLayerKind === "aux" ? state.spec.auxiliary_layers : state.spec.layers;
  return collection.find((layer) => layer.index === state.selectedLayerIndex) || state.spec.layers[0];
}

function selectedLayerResult() {
  const key = `${state.selectedLayerKind}:${state.selectedLayerIndex}`;
  return state.analysis.layers.find((layer) => layer.resultKey === key) || state.analysis.layers[0];
}

function renderSelectedLayerHeader() {
  const layer = selectedLayer();
  const result = selectedLayerResult();
  const isMtp = layer.kind === "mtp";
  el.selectedLayerNumber.textContent = isMtp ? "MTP" : `L${String(layer.index).padStart(2, "0")}`;
  el.selectedLayerTitle.textContent = layer.label;
  const mixerLabel = layer.attention_type === "linear_attention" ? "Gated DeltaNet" : "Qwen Sparse Attention";
  const pleLabel = layer.has_ple ? " · N-gram PLE" : "";
  el.selectedLayerSubtitle.textContent = `${mixerLabel}${pleLabel} · Top-${state.spec.architecture.experts_per_token} MoE · ${layer.cache_mode}`;
  el.layerSummary.innerHTML = `
    <div class="summary-pill"><span>Params</span><strong>${formatParams(result.logical_params)}</strong></div>
    <div class="summary-pill"><span>Weight</span><strong>${formatBytes(result.weight_bytes)}</strong></div>
    <div class="summary-pill"><span>Model share</span><strong>${formatPercent(result.weight_ratio)}</strong></div>
    <div class="summary-pill"><span>State / KV</span><strong>${formatBytes(result.kv_cache.total_bytes)}</strong></div>
  `;
}

function renderGraph() {
  const layer = selectedLayer();
  const graph = state.expandedModuleId ? buildModuleSubgraph(state.expandedModuleId) : state.spec.operator_graphs.module;
  const visibleNodes = graph.nodes.filter((node) => conditionMatches(node.condition, layer));
  const nodeIds = new Set(visibleNodes.map((node) => node.id));
  const visibleEdges = graph.edges.filter((edge) => conditionMatches(edge.condition, layer) && nodeIds.has(edge.from) && nodeIds.has(edge.to));
  if (state.selectedNodeId && !nodeIds.has(state.selectedNodeId)) state.selectedNodeId = null;

  el.collapseModule.hidden = !state.expandedModuleId;
  el.graphContext.textContent = state.expandedModuleId ? `当前：${graph.displayName}` : "单击查看 · 双击展开";

  const layout = state.expandedModuleId
    ? buildSubgraphLayout(state.expandedModuleId, visibleNodes)
    : buildModuleLayout(visibleNodes);
  const svg = el.operatorGraph;
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);
  svg.style.width = `${layout.width}px`;
  svg.style.height = `${layout.height}px`;
  svg.append(createGraphDefs());

  visibleEdges.forEach((edge) => {
    const from = layout.positions[edge.from];
    const to = layout.positions[edge.to];
    if (!from || !to) return;
    const path = svgElement("path", {
      d: edgePath(from, to),
      class: `graph-edge${edge.kind === "residual" ? " residual" : ""}`
    });
    svg.append(path);
  });

  visibleNodes.forEach((node) => {
    const position = layout.positions[node.id];
    const group = svgElement("g", {
      class: `graph-node${state.selectedNodeId === node.id ? " is-selected" : ""}`,
      transform: `translate(${position.x} ${position.y})`,
      tabindex: "0",
      role: "button",
      "aria-label": `${node.name}, ${node.shape || "无参数 Shape"}${EXPANDABLE_MODULES.has(node.id) ? "，单击查看信息，双击展开" : "，单击查看信息"}`,
      "data-node-id": node.id,
      "data-group": node.group
    });
    group.append(svgElement("rect", { width: position.width, height: position.height, rx: 10, ry: 10 }));
    const name = svgElement("text", { x: position.width / 2, y: 27, "text-anchor": "middle", class: "node-name" });
    name.textContent = node.name;
    group.append(name);
    const shape = svgElement("text", { x: position.width / 2, y: 47, "text-anchor": "middle", class: "node-shape" });
    shape.textContent = shorten(node.shape || "tensor flow", 31);
    group.append(shape);
    const selectNode = () => {
      state.selectedNodeId = node.id;
      renderGraph();
      renderInspector();
    };
    const expandNode = () => {
      state.expandedModuleId = node.id;
      state.selectedNodeId = null;
      renderGraph();
      renderInspector();
    };
    group.addEventListener("click", () => {
      if (!EXPANDABLE_MODULES.has(node.id)) {
        selectNode();
        return;
      }
      if (state.pendingClickTimer) window.clearTimeout(state.pendingClickTimer);
      state.pendingClickTimer = window.setTimeout(() => {
        state.pendingClickTimer = null;
        selectNode();
      }, 230);
    });
    group.addEventListener("dblclick", (event) => {
      if (!EXPANDABLE_MODULES.has(node.id)) return;
      event.preventDefault();
      if (state.pendingClickTimer) window.clearTimeout(state.pendingClickTimer);
      state.pendingClickTimer = null;
      expandNode();
    });
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        selectNode();
      } else if (event.key === " " && EXPANDABLE_MODULES.has(node.id)) {
        event.preventDefault();
        expandNode();
      }
    });
    svg.append(group);
    renderNodeMetrics(svg, node, position, layer);
  });
}

function renderNodeMetrics(svg, node, position, layer) {
  if (node.group === "tensor") return;
  const groups = parameterGroupsForNode(node, layer);
  const nodeWeightBytes = groups.reduce((total, group) => total + componentWeightGroupBytes(group, layer), 0);
  const layerWeightBytes = selectedLayerResult().weight_bytes;
  const layerShare = layerWeightBytes ? nodeWeightBytes / layerWeightBytes : 0;
  const kvLoadBytes = kvLoadBytesForNode(node.id);
  const labels = [
    { text: `Layer ${formatPercent(layerShare)}`, kind: "share" },
    { text: `Logical ${formatParams(groups.reduce((total, group) => total + group.logical_params, 0))}`, kind: "logical" },
    { text: `Weight ${formatBytes(nodeWeightBytes)}`, kind: "weight" }
  ];
  if (kvLoadBytes > 0) labels.push({ text: `KV Load ${formatBytes(kvLoadBytes)}`, kind: "kv" });

  const panelHeight = labels.length * 15 + 10;
  const meta = svgElement("g", {
    class: "node-meta",
    transform: `translate(${position.x + position.width + 8} ${position.y + 6})`,
    "data-meta-for": node.id,
    "aria-hidden": "true"
  });
  meta.append(svgElement("rect", {
    x: 0,
    y: 0,
    width: 132,
    height: panelHeight,
    rx: 7,
    ry: 7,
    class: `node-meta-panel${kvLoadBytes > 0 ? " has-kv" : ""}`
  }));
  labels.forEach((label, index) => {
    const textNode = svgElement("text", {
      x: 9,
      y: 17 + index * 15,
      "text-anchor": "start",
      class: `node-meta-text ${label.kind}`
    });
    textNode.textContent = label.text;
    meta.append(textNode);
  });
  svg.append(meta);
}

function kvLoadBytesForNode(nodeId) {
  const kv = selectedLayerResult().kv_cache;
  if (nodeId === "gdn") return kv.recurrent_state_bytes + kv.conv_state_bytes;
  if (nodeId === "gdn_recur") return kv.recurrent_state_bytes;
  if (nodeId === "gdn_conv") return kv.conv_state_bytes;
  if (nodeId === "qsa") return kv.main_cache_bytes + kv.index_cache_bytes;
  if (nodeId === "qsa_attn" || nodeId === "qsa_kv") return kv.main_cache_bytes;
  if (nodeId === "qsa_index" || nodeId === "qsa_pool" || nodeId === "qsa_topk") return kv.index_cache_bytes;
  if (nodeId === "ple" || nodeId === "ple_conv") return kv.auxiliary_state_bytes;
  return 0;
}

function createGraphDefs() {
  const defs = svgElement("defs");
  const marker = svgElement("marker", {
    id: "arrowhead",
    markerWidth: 8,
    markerHeight: 8,
    refX: 7,
    refY: 4,
    orient: "auto",
    markerUnits: "strokeWidth"
  });
  marker.append(svgElement("path", { d: "M 0 0 L 8 4 L 0 8 z", fill: "rgba(137,165,183,.58)" }));
  defs.append(marker);
  return defs;
}

function buildModuleLayout(nodes) {
  const width = 820;
  const nodeWidth = 290;
  const nodeHeight = 66;
  const positions = {};
  nodes.forEach((node, index) => {
    positions[node.id] = {
      x: (width - nodeWidth) / 2,
      y: 28 + index * 102,
      width: nodeWidth,
      height: nodeHeight
    };
  });
  return { width, height: Math.max(620, 42 + nodes.length * 102), positions };
}

// Qwen4-Exp module drill-downs share the generic SVG renderer above.
function buildModuleSubgraph(moduleId) {
  const expanded = new Map(state.spec.operator_graphs.expanded.nodes.map((node) => [node.id, node]));
  const take = (id) => ({ ...expanded.get(id) });
  const virtual = (id, name, shape, description) => ({ id, name, shape, description, group: "tensor" });
  const chain = (ids) => ids.slice(0, -1).map((id, index) => ({ from: id, to: ids[index + 1] }));

  if (moduleId === "ple") {
    const ids = ["ple_input", "ple_lookup", "ple_gate", "ple_conv", "ple_output"];
    return {
      displayName: "N-gram PLE",
      nodes: [
        virtual("ple_input", "Token history", "current + previous 2 ids", "Bigram and trigram ids are hashed deterministically."),
        take("ple_lookup"), take("ple_gate"), take("ple_conv"),
        virtual("ple_output", "4-stream injection", "[B,S,4,2560]", "Lexical pattern memory is injected before the second decoder block.")
      ],
      edges: chain(ids)
    };
  }
  if (moduleId === "gdn") {
    const ids = ["gdn_input", "gdn_qkv", "gdn_conv", "gdn_recur", "gdn_gate", "gdn_out", "gdn_output"];
    return {
      displayName: "Gated DeltaNet",
      nodes: [
        virtual("gdn_input", "GR mixed input", "[B,S,2560]", "Read-gated mixture of four residual streams."),
        take("gdn_qkv"), take("gdn_conv"), take("gdn_recur"), take("gdn_gate"), take("gdn_out"),
        virtual("gdn_output", "GDN output", "[B,S,2560]", "The write gate injects this result into four streams.")
      ],
      edges: chain(ids)
    };
  }
  if (moduleId === "qsa") {
    return {
      displayName: "Qwen Sparse Attention",
      nodes: [
        virtual("qsa_input", "GR mixed input", "[B,S,2560]", "Read-gated mixture of four residual streams."),
        take("qsa_q"), take("qsa_kv"), take("qsa_index"), take("qsa_pool"), take("qsa_topk"), take("qsa_attn"), take("qsa_out"),
        virtual("qsa_output", "QSA output", "[B,S,2560]", "Sparse global retrieval followed by a learned output gate.")
      ],
      edges: [
        { from: "qsa_input", to: "qsa_q" }, { from: "qsa_input", to: "qsa_kv" },
        { from: "qsa_input", to: "qsa_index" }, { from: "qsa_index", to: "qsa_pool" },
        { from: "qsa_pool", to: "qsa_topk" }, { from: "qsa_topk", to: "qsa_attn" },
        { from: "qsa_q", to: "qsa_attn" }, { from: "qsa_kv", to: "qsa_attn" },
        { from: "qsa_attn", to: "qsa_out" }, { from: "qsa_out", to: "qsa_output" }
      ]
    };
  }
  if (moduleId === "gr_attn" || moduleId === "gr_mlp") {
    const isAttention = moduleId === "gr_attn";
    const prefix = isAttention ? "gr_attn" : "gr_mlp";
    const readId = `${prefix}_read`;
    const writeId = `${prefix}_write`;
    const opName = isAttention ? "GDN / QSA" : "Sparse MoE";
    return {
      displayName: isAttention ? "Sequence-mixer Gated Residual" : "MoE Gated Residual",
      nodes: [
        virtual(`${prefix}_input`, "4-stream input", "[B,S,4,2560]", "Widened residual stream."),
        take(readId),
        virtual(`${prefix}_op`, opName, "[B,S,2560]", "The wrapped block consumes the dynamically mixed input."),
        take(writeId),
        virtual(`${prefix}_output`, "4-stream output", "[B,S,4,2560]", "Per-branch write gates inject the block output.")
      ],
      edges: [
        { from: `${prefix}_input`, to: readId }, { from: readId, to: `${prefix}_op` },
        { from: `${prefix}_op`, to: writeId }, { from: `${prefix}_input`, to: writeId, kind: "residual" },
        { from: writeId, to: `${prefix}_output` }
      ]
    };
  }
  return {
    displayName: "Sparse MoE",
    nodes: [
      virtual("moe_input", "GR mixed input", "[B,S,2560]", "The router and shared path consume the same normalized input."),
      take("router"), take("routed"), take("shared"), take("combine"),
      virtual("moe_output", "MoE output", "[B,S,2560]", "Top-10 routed results are combined with the gated shared expert.")
    ],
    edges: [
      { from: "moe_input", to: "router" }, { from: "router", to: "routed" },
      { from: "moe_input", to: "shared" }, { from: "routed", to: "combine" },
      { from: "shared", to: "combine" }, { from: "combine", to: "moe_output" }
    ]
  };
}

function buildSubgraphLayout(moduleId, nodes) {
  const width = 1020;
  const nodeWidth = 188;
  const nodeHeight = 64;
  let coordinates = {};
  let height = 650;
  if (moduleId === "qsa") {
    coordinates = {
      qsa_input: [356, 24], qsa_q: [20, 130], qsa_kv: [356, 130], qsa_index: [692, 130],
      qsa_pool: [692, 230], qsa_topk: [692, 330], qsa_attn: [356, 450],
      qsa_out: [356, 550], qsa_output: [356, 650]
    };
    height = 745;
  } else if (moduleId === "moe") {
    coordinates = {
      moe_input: [356, 24], router: [20, 130], routed: [20, 230], shared: [692, 130],
      combine: [356, 340], moe_output: [356, 440]
    };
    height = 535;
  } else {
    nodes.forEach((node, index) => {
      coordinates[node.id] = [356, 24 + index * 100];
    });
    height = Math.max(535, 44 + nodes.length * 100);
  }
  const positions = {};
  nodes.forEach((node) => {
    const [x, y] = coordinates[node.id] || [356, 24];
    positions[node.id] = { x, y, width: nodeWidth, height: nodeHeight };
  });
  return { width, height, positions };
}

function edgePath(from, to) {
  const startX = from.x + from.width / 2;
  const startY = from.y + from.height;
  const endX = to.x + to.width / 2;
  const endY = to.y;
  const distance = Math.max(28, Math.abs(endY - startY) * 0.45);
  return `M ${startX} ${startY} C ${startX} ${startY + distance}, ${endX} ${endY - distance}, ${endX} ${endY}`;
}

function conditionMatches(condition, layer) {
  if (!condition) return true;
  if (condition === "linear_attention") return layer.attention_type === "linear_attention";
  if (condition === "qsa") return layer.attention_type !== "linear_attention";
  if (condition === "has_ple") return layer.has_ple && state.scenario.include_ngram;
  if (condition === "no_ple") return !layer.has_ple || !state.scenario.include_ngram;
  return false;
}

function svgElement(tag, attributes = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attributes).forEach(([name, value]) => node.setAttribute(name, String(value)));
  return node;
}

function renderInspector() {
  const layer = selectedLayer();
  const currentGraph = state.expandedModuleId ? buildModuleSubgraph(state.expandedModuleId) : state.spec.operator_graphs.module;
  const node = currentGraph.nodes.find((item) => item.id === state.selectedNodeId)
    || state.spec.operator_graphs.expanded.nodes.find((item) => item.id === state.selectedNodeId)
    || state.spec.operator_graphs.module.nodes.find((item) => item.id === state.selectedNodeId);
  if (!node) {
    el.operatorInspector.innerHTML = `
      <div class="inspector-placeholder">
        <span class="inspector-icon" aria-hidden="true">◎</span>
        <p>选择图中的算子，查看张量 Shape、参数与量化后 Weight。</p>
      </div>`;
    return;
  }

  const groups = parameterGroupsForNode(node, layer);
  const weightBytes = groups.reduce((total, group) => total + componentWeightGroupBytes(group, layer), 0);
  const params = groups.reduce((total, group) => total + group.logical_params, 0);
  const layerResult = selectedLayerResult();
  const layerShare = layerResult.weight_bytes ? weightBytes / layerResult.weight_bytes : 0;
  const modelShare = state.analysis.totals.weight_bytes ? weightBytes / state.analysis.totals.weight_bytes : 0;
  const sourceLines = node.source_lines || groups[0]?.source.lines || "";
  const sourceUrl = sourceLink(sourceLines);

  const rows = groups.length ? groups.map((group) => `
    <tr>
      <td>${escapeHtml(group.name)}</td>
      <td>${escapeHtml(`[${group.shape.join(" × ")}]`)}</td>
      <td>${formatInteger(group.copies)}</td>
      <td>${formatParams(group.logical_params)}</td>
      <td>${formatBytes(componentWeightGroupBytes(group, layer))}</td>
    </tr>`).join("") : `
    <tr><td colspan="5">该节点为无权重 Tensor 操作。</td></tr>`;

  el.operatorInspector.innerHTML = `
    <div class="inspector-header">
      <div>
        <h3>${escapeHtml(node.name)}</h3>
        <p>${escapeHtml(node.description || `${node.group} operator`)} · Shape ${escapeHtml(node.shape || "—")}</p>
      </div>
      ${sourceLines ? `<a class="source-link" href="${sourceUrl}" target="_blank" rel="noreferrer">modeling_qwen4_exp.py:${escapeHtml(sourceLines)}</a>` : ""}
    </div>
    <div class="inspector-stats">
      <div class="inspector-stat"><span>Logical params</span><strong>${formatParams(params)}</strong></div>
      <div class="inspector-stat"><span>Weight</span><strong>${formatBytes(weightBytes)}</strong></div>
      <div class="inspector-stat"><span>Layer share</span><strong>${formatPercent(layerShare)}</strong></div>
      <div class="inspector-stat"><span>Model share</span><strong>${formatPercent(modelShare)}</strong></div>
    </div>
    <div class="tensor-table-wrap">
      <table class="tensor-table">
        <thead><tr><th>Tensor</th><th>Shape</th><th>Copies</th><th>Params</th><th>Weight</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

function parameterGroupsForNode(node, layer) {
  const ids = new Set(node.parameter_ids || []);
  const prefixes = node.parameter_prefixes || [];
  return layer.parameter_groups.filter(groupIncluded).filter((group) => ids.has(group.id) || prefixes.some((prefix) => group.id.startsWith(prefix)));
}

function sourceLink(lines) {
  const firstLine = String(lines).match(/\d+/)?.[0] || "1";
  return `https://github.com/huggingface/transformers/blob/${state.spec.source.transformers_revision}/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L${firstLine}`;
}

async function saveLocal() {
  state.scenario.name = el.scenarioName.value.trim() || "untitled";
  state.analysis = calculateAnalysis();
  setStatus("正在保存…");
  try {
    const response = await fetchJson("/api/scenarios", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: state.scenario.name,
        scenario: state.scenario,
        analysis: serializableAnalysis()
      })
    });
    setStatus(`已保存 ${response.scenario_path} 和 ${response.analysis_path}`);
    await refreshScenarioList(state.scenario.name);
  } catch (error) {
    setStatus("本地服务不可用；仍可使用导出按钮保存 JSON。", true);
  }
}

async function refreshScenarioList(selectedName = "") {
  try {
    const result = await fetchJson("/api/scenarios");
    el.savedScenarioSelect.replaceChildren();
    if (!result.scenarios.length) {
      el.savedScenarioSelect.append(new Option("暂无已保存场景", ""));
      return;
    }
    result.scenarios.forEach((item) => {
      el.savedScenarioSelect.append(new Option(item.name, item.name, false, item.name === selectedName));
    });
  } catch {
    el.savedScenarioSelect.replaceChildren(new Option("静态模式：使用导入 / 导出", ""));
  }
}

async function loadSavedScenario() {
  const name = el.savedScenarioSelect.value;
  if (!name) return;
  setStatus("正在载入场景…");
  try {
    const result = await fetchJson(`/api/scenarios/${encodeURIComponent(name)}`);
    applyScenario(result.scenario);
    setStatus(`已载入场景 ${name}`);
  } catch (error) {
    setStatus("场景载入失败。", true);
  }
}

async function importScenario(event) {
  const [file] = event.target.files;
  if (!file) return;
  try {
    const scenario = JSON.parse(await file.text());
    applyScenario(scenario);
    setStatus(`已导入 ${file.name}`);
  } catch (error) {
    setStatus("场景 JSON 无效或字段不完整。", true);
  } finally {
    event.target.value = "";
  }
}

function applyScenario(scenario) {
  const normalized = {
    ngram_quantization: "inherit",
    vision_quantization: "inherit",
    mtp_quantization: "inherit",
    ...scenario
  };
  validateScenario(normalized);
  state.scenario = normalized;
  el.weightProfile.value = normalized.weight_quantization;
  el.kvProfile.value = normalized.kv_cache_quantization;
  el.contextLength.value = String(normalized.context_length);
  el.batchSize.value = String(normalized.batch_size);
  el.includeNgram.checked = normalized.include_ngram;
  el.ngramProfile.value = normalized.ngram_quantization;
  el.includeVision.checked = normalized.include_vision;
  el.visionProfile.value = normalized.vision_quantization;
  el.includeMtp.checked = normalized.include_mtp;
  el.mtpProfile.value = normalized.mtp_quantization;
  el.scenarioName.value = normalized.name;
  syncComponentControls();
  if (!normalized.include_mtp && state.selectedLayerKind === "aux") {
    state.selectedLayerKind = "main";
    state.selectedLayerIndex = 0;
  }
  state.selectedNodeId = null;
  state.expandedModuleId = null;
  updateProfileDescriptions();
  renderAll();
}

function validateScenario(scenario) {
  const required = [
    "name", "model_id", "weight_quantization", "kv_cache_quantization", "context_length",
    "batch_size", "include_ngram", "ngram_quantization", "include_vision", "vision_quantization",
    "include_mtp", "mtp_quantization", "operator_detail"
  ];
  required.forEach((key) => {
    if (!Object.hasOwn(scenario, key)) throw new Error(`Missing ${key}`);
  });
  if (scenario.model_id !== state.spec.model_id) throw new Error("Wrong model");
  if (!state.weightProfiles.profiles.some((item) => item.id === scenario.weight_quantization)) throw new Error("Unknown weight profile");
  if (!state.kvProfiles.profiles.some((item) => item.id === scenario.kv_cache_quantization)) throw new Error("Unknown KV profile");
  for (const key of ["ngram_quantization", "vision_quantization", "mtp_quantization"]) {
    if (!COMPONENT_QUANTIZATION_IDS.has(scenario[key])) throw new Error(`Unknown component profile: ${key}`);
  }
  if (!["module", "expanded"].includes(scenario.operator_detail)) throw new Error("Wrong detail mode");
  if (scenario.context_length < 1 || scenario.context_length > state.spec.architecture.extended_context_length) throw new Error("Context out of range");
  if (scenario.batch_size < 1 || scenario.batch_size > 1024) throw new Error("Batch out of range");
}

function exportScenario() {
  state.scenario.name = el.scenarioName.value.trim() || "untitled";
  downloadJson(`${safeFileName(state.scenario.name)}.scenario.json`, state.scenario);
  setStatus("场景 JSON 已导出。请在浏览器下载目录中查看。 ");
}

function exportAnalysis() {
  state.analysis = calculateAnalysis();
  downloadJson(`${safeFileName(state.scenario.name)}.analysis.json`, serializableAnalysis());
  setStatus("分析结果 JSON 已导出。 ");
}

function serializableAnalysis() {
  return JSON.parse(JSON.stringify(state.analysis));
}

function downloadJson(filename, value) {
  const blob = new Blob([JSON.stringify(value, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function setStatus(message, error = false) {
  el.statusMessage.textContent = message;
  el.statusMessage.classList.toggle("is-error", error);
}

function safeFileName(value) {
  return value.trim().replace(/[^a-zA-Z0-9\u4e00-\u9fff._-]+/g, "-").replace(/^-+|-+$/g, "") || "scenario";
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = bytes;
  let unitIndex = 0;
  while (Math.abs(value) >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const digits = value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[unitIndex]}`;
}

function formatParams(value) {
  if (value >= 1e9) return `${(value / 1e9).toFixed(value >= 100e9 ? 2 : 3)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return formatInteger(value);
}

function formatPercent(value) {
  if (!Number.isFinite(value)) return "—";
  if (value > 0 && value < 0.0001) return "<0.01%";
  return `${(value * 100).toFixed(value < 0.01 ? 2 : 1)}%`;
}

function formatInteger(value) {
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 0 }).format(value);
}

function formatContext(value) {
  if (value === 1000000 || value === 1048576) return "1M";
  if (value % 1024 === 0) return `${value / 1024}K`;
  return formatInteger(value);
}

function shorten(value, maxLength) {
  return value.length > maxLength ? `${value.slice(0, maxLength - 1)}…` : value;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
