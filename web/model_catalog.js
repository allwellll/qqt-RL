'use strict';

(function modelCatalogFactory(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.QQTModelCatalog = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function buildCatalog() {
  const SHA256 = /^[0-9a-f]{64}$/;
  const FILE = /^[a-z0-9][a-z0-9._-]*\.json$/;

  function validateManifest(document) {
    if (!document || document.schema !== 'qqt.web-models/v1') throw new Error('invalid model manifest schema');
    if (!Array.isArray(document.models)) throw new Error('invalid model list');
    return document.models.map((row) => {
      if (!row || typeof row.id !== 'string' || !row.id) throw new Error('invalid model id');
      if (!FILE.test(row.file || '')) throw new Error('invalid model file');
      if (!SHA256.test(row.sha256 || '')) throw new Error('invalid model sha256');
      if (!Number.isInteger(row.bytes) || row.bytes <= 0) throw new Error('invalid model bytes');
      if (!Number.isInteger(row.cycle) || row.cycle < 0) throw new Error('invalid model cycle');
      if (row.specialist) {
        const card = row.specialist, evidence = card.evidence;
        const text = value => typeof value === 'string' && value.trim().length > 0;
        const list = value => Array.isArray(value) && value.length > 0 && value.every(text);
        if (row.rule_version !== 'bun-native-v3-items' || !text(card.label)
            || !list(card.strengths) || !list(card.weaknesses) || !evidence
            || evidence.ruleset !== row.rule_version || !Number.isInteger(evidence.games) || evidence.games <= 0
            || !text(evidence.summary)
            || !/^specialists\/[a-z0-9][a-z0-9._-]*\.json(?:#[a-z0-9-]+)?$/.test(evidence.url || '')) {
          throw new Error('invalid specialist evidence card');
        }
        if (card.compatibility && (card.compatibility.ruleset !== 'bun-native-v4-facing'
            || !Number.isInteger(card.compatibility.games) || card.compatibility.games <= 0
            || !text(card.compatibility.summary))) throw new Error('invalid specialist compatibility scope');
      }
      if (row.onnx && (!/^[a-z0-9][a-z0-9._-]*\.onnx$/.test(row.onnx.file || '')
          || !SHA256.test(row.onnx.sha256 || '') || !Number.isInteger(row.onnx.bytes) || row.onnx.bytes <= 0)) {
        throw new Error('invalid ONNX asset');
      }
      if (row.onnx && (row.onnx.meta || row.onnx.reference_zero)) {
        const m=row.onnx.meta, r=row.onnx.reference_zero;
        if (!m || !r || m.arch!=='transformer'
            || m.checkpoint_sha256!==row.source_sha256
            || !Array.isArray(m.obs_shape) || m.obs_shape.length!==3
            || !m.obs_shape.every(n=>Number.isInteger(n)&&n>0)
            || !Number.isInteger(m.state_dim) || m.state_dim<=0
            || !Array.isArray(r.move) || r.move.length!==m.move_actions
            || !Array.isArray(r.bomb) || r.bomb.length!==m.ability_actions
            || ![...r.move,...r.bomb,r.value].every(Number.isFinite)) {
          throw new Error('invalid ONNX metadata/reference');
        }
        if(m.ability_actions>3 && (m.schema_version!=='five-item-slots-v1'
            || m.action_schema!=='move-v1+five-item-slots-v1' || m.ability_actions!==7
            || m.state_dim!==52 || m.global_dim!==52
            || JSON.stringify(m.obs_shape)!=='[25,13,15]'
            || JSON.stringify(m.ability_encoding)!=='["none","bomb","use_slot1","use_slot2","use_slot3","use_slot4","use_slot5"]')) {
          throw new Error('invalid five-slot model schema');
        }
      }
      for (const asset of [row, ...(row.onnx ? [row.onnx] : [])]) if (asset.oss_url) {
        if (!/^https:\/\/qqt-model\.oss-cn-beijing\.aliyuncs\.com\/web-models\/[a-z0-9._-]+$/.test(asset.oss_url)
            || !asset.oss_url.includes(asset.sha256)
            || !asset.oss_url.endsWith(asset.file.endsWith('.onnx') ? '.onnx' : '.json')
            || !/^https:\/\/github\.com\/[^/]+\/[^/]+\/releases\/download\/[^/]+\/[^/?#]+$/.test(asset.url || '')
            || !asset.url.endsWith('/' + asset.file)) throw new Error('invalid OSS/Release asset sources');
      }
      return Object.freeze({ ...row });
    });
  }

  function modelUrl(row) { return `models/${row.file}`; }
  function optionLabel(row) {
    return row.menu_label || (row.specialist ? `${row.display_name} · 专项：${row.specialist.label}` : row.display_name);
  }
  function resolveSelection(value, rows, defaultModelId) {
    if (value && !value.startsWith('model:')) return value;
    const id = value?.slice('model:'.length);
    const selected = rows.find(row => row.id === id) || rows.find(row => row.id === defaultModelId);
    return selected ? `model:${selected.id}` : '';
  }
  function specialistLines(row) {
    const card = row?.specialist;
    return card ? [`专项：${card.label}`, `长处：${card.strengths.join('；')}`,
      `短板：${card.weaknesses.join('；')}`,
      `证据范围：原评测 ${card.evidence.ruleset} · ${card.evidence.games} 局；${card.evidence.summary}`,
      ...(card.compatibility ? [`当前规则快筛：${card.compatibility.ruleset} · ${card.compatibility.games} 局；${card.compatibility.summary}`] : [])] : [];
  }
  function matchLabel(mode) { return mode === 'model-vs-rule' ? '模型 vs 规则Bot' : '真人 vs 对手'; }
  return { validateManifest, modelUrl, optionLabel, resolveSelection, specialistLines, matchLabel };
});
