/* 智慧笔迹 · NLP 新闻分类系统 前端逻辑 */

// 14 类图标（按 label 0~13 顺序）
const LABEL_EMOJI = [
  '💻', '📈', '⚽', '🎬', '🏛', '👥', '🎓',
  '💰', '🏠', '🎮', '🏢', '👗', '🎰', '✨',
];

const state = {
  tab: 'sample',
  labels: [],
  trueLabel: null,      // 验证演示模式下的真实标签
  verifyTotal: 0,
  verifyCorrect: 0,
  chart: null,
};

const $ = (id) => document.getElementById(id);

// ---------- 初始化 ----------
async function init() {
  try {
    const meta = await fetch('/api/meta').then((r) => r.json());
    state.labels = meta.labels;
    $('modelBadge').textContent =
      '模型：' + (meta.models[0] ? meta.models[0].display_name : '未知');
    $('footerMeta').textContent =
      `验证池 ${meta.demo_pool_size} 条 · 离线 macro F1 ${meta.demo_pool_f1} · 准确率 ${meta.demo_pool_acc}`;
  } catch (e) {
    $('modelBadge').textContent = '模型：加载失败';
    $('footerMeta').textContent = '后端未连接，请先运行 python run.py';
  }
  initChart();
  bindEvents();
}

// ---------- 事件绑定 ----------
function bindEvents() {
  // Tab 切换
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });

  $('btnSample').addEventListener('click', async () => {
    const d = await fetch('/api/sample').then((r) => r.json());
    setText(d.text);
    state.trueLabel = null;
    resetResult();
  });

  $('btnDemo').addEventListener('click', async () => {
    const d = await fetch('/api/demo').then((r) => r.json());
    setText(d.text);
    state.trueLabel = d.true_label;
    $('trueLabelBadge').textContent = '真实标签（预测后揭晓）';
    $('trueLabelBadge').classList.remove('hidden');
    resetResult();
  });

  $('btnPredict').addEventListener('click', predict);
  $('btnClear').addEventListener('click', () => {
    setText('');
    state.trueLabel = null;
    $('trueLabelBadge').classList.add('hidden');
    resetResult();
  });

  $('textInput').addEventListener('input', onTextChange);
}

function switchTab(tab) {
  state.tab = tab;
  document.querySelectorAll('.tab').forEach((t) =>
    t.classList.toggle('active', t.dataset.tab === tab));
  document.querySelectorAll('.tab-body').forEach((b) =>
    b.classList.toggle('active', b.id === 'tab-' + tab));
  // 手动输入模式清空，其余模式清掉旧结果
  state.trueLabel = null;
  $('trueLabelBadge').classList.add('hidden');
  resetResult();
  if (tab !== 'manual') setText('');
}

function setText(text) {
  $('textInput').value = text;
  onTextChange();
}

function onTextChange() {
  const n = $('textInput').value.trim().split(/\s+/).filter(Boolean).length;
  $('charCount').textContent = n + ' token';
  $('btnPredict').disabled = n === 0;
}

// ---------- 预测 ----------
async function predict() {
  const text = $('textInput').value.trim();
  if (!text) return;

  const btn = $('btnPredict');
  btn.disabled = true;
  btn.textContent = '预测中…';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    renderResult(data);
  } catch (e) {
    alert('预测失败，请确认后端已启动（python run.py）');
  } finally {
    btn.disabled = false;
    btn.textContent = '🔮 开始预测';
  }
}

// ---------- 结果渲染 ----------
function renderResult(data) {
  $('resultEmpty').classList.add('hidden');
  $('resultBody').classList.remove('hidden');

  // 预测类别 + 置信度
  $('predLabel').textContent = LABEL_EMOJI[data.label] + ' ' + data.name;
  $('predConf').textContent = '置信度 ' + (data.prob * 100).toFixed(2) + '%';

  // 概率条形图
  renderChart(data);

  // Top-3
  $('top3').innerHTML = data.top3
    .map((c, i) =>
      `<div class="cand">${i === 0 ? '🥇' : i === 1 ? '🥈' : '🥉'} <b>${c.name}</b>` +
      `<span class="pct">${(c.prob * 100).toFixed(1)}%</span></div>`)
    .join('');

  // 关键词
  if (data.keywords && data.keywords.length) {
    $('keywordsBlock').classList.remove('hidden');
    $('keywords').innerHTML = data.keywords
      .map((k) => `<span class="kw">${k.term}<span class="score">${k.score.toFixed(2)}</span></span>`)
      .join('');
  } else {
    $('keywordsBlock').classList.add('hidden');
  }

  // 验证演示：揭晓真实标签 + 累计正确率
  if (state.trueLabel !== null) {
    $('verifyResult').classList.remove('hidden');
    const correct = data.label === state.trueLabel;
    $('trueLabel').textContent = state.labels[state.trueLabel];
    $('trueLabel').className = 'tag';
    $('predLabelSmall').textContent = data.name;
    $('predLabelSmall').className = 'tag';
    $('verdict').textContent = correct ? '✓ 预测正确' : '✗ 预测错误';
    $('verdict').className = 'tag ' + (correct ? 'correct' : 'wrong');

    state.verifyTotal += 1;
    if (correct) state.verifyCorrect += 1;
    const pct = state.verifyTotal ? (state.verifyCorrect / state.verifyTotal * 100) : 0;
    $('scoreText').textContent = `${state.verifyCorrect} / ${state.verifyTotal}`;
    $('scorePct').textContent = `（${pct.toFixed(1)}%）`;
  } else {
    $('verifyResult').classList.add('hidden');
  }
}

function resetResult() {
  $('resultEmpty').classList.remove('hidden');
  $('resultBody').classList.add('hidden');
  $('verifyResult').classList.add('hidden');
  if (state.chart) state.chart.clear();
}

// ---------- ECharts 概率分布图 ----------
function initChart() {
  state.chart = echarts.init($('probChart'));
  window.addEventListener('resize', () => state.chart && state.chart.resize());
}

function renderChart(data) {
  // 按概率降序排列
  const items = state.labels
    .map((name, i) => ({ name, emoji: LABEL_EMOJI[i], prob: data.probs[i], i }))
    .sort((a, b) => b.prob - a.prob);

  const names = items.map((it) => it.emoji + ' ' + it.name);
  const vals = items.map((it) => +(it.prob * 100).toFixed(2));
  // 预测类高亮为蓝色，其余灰色
  const colors = items.map((it) =>
    it.i === data.label ? '#4C78A8' : '#c3ccd6');

  state.chart.setOption({
    grid: { left: 10, right: 40, top: 10, bottom: 10, containLabel: true },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' },
      formatter: (p) => `${p[0].name}<br/>概率：${p[0].value}%` },
    xAxis: { type: 'value', max: 100, axisLabel: { formatter: '{value}%', fontSize: 11 } },
    yAxis: {
      type: 'category', data: names.reverse(),
      axisLabel: { fontSize: 12, color: '#444' }, axisTick: { show: false },
    },
    series: [{
      type: 'bar', data: vals.reverse(),
      barMaxWidth: 18, itemStyle: { color: (p) => colors[colors.length - 1 - p.dataIndex], borderRadius: [0, 4, 4, 0] },
      label: { show: true, position: 'right', formatter: '{c}%', fontSize: 10, color: '#555' },
    }],
  });
}

init();
