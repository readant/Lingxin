/**
 * Lingxin Chart Module - ECharts Integration
 * Handles confidence bar chart and word frequency pie chart
 */

let confidenceChart = null;
let wordFreqChart = null;

function initCharts() {
    if (typeof echarts === 'undefined') {
        console.warn('ECharts not loaded');
        return;
    }

    initConfidenceChart();
    initWordFreqChart();
    window.updateCharts = updateCharts;
    window.updateWordFreqChart = updateWordFreqChart;
    window.clearCharts = clearCharts;
}

function initConfidenceChart() {
    const el = document.getElementById('confidenceChart');
    if (!el) return;

    confidenceChart = echarts.init(el, 'dark');
    confidenceChart.setOption({
        backgroundColor: 'transparent',
        grid: { top: 10, right: 20, bottom: 30, left: 60 },
        xAxis: {
            type: 'category',
            data: [],
            axisLabel: { color: '#9B8B7A', fontSize: 12 },
            axisLine: { lineStyle: { color: '#E8E0D5' } }
        },
        yAxis: {
            type: 'value',
            max: 100,
            axisLabel: { color: '#9B8B7A', formatter: '{value}%' },
            splitLine: { lineStyle: { color: '#F5F0E8' } }
        },
        series: [{
            type: 'bar',
            data: [],
            barWidth: '50%',
            itemStyle: {
                borderRadius: [4, 4, 0, 0],
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: '#E8C9A0' },
                    { offset: 1, color: '#D4A574' }
                ])
            },
            animationDuration: 300,
            animationEasing: 'cubicOut'
        }],
        tooltip: {
            trigger: 'axis',
            backgroundColor: 'rgba(250, 247, 242, 0.95)',
            borderColor: '#E8E0D5',
            textStyle: { color: '#2C2420' },
            formatter: (params) => {
                const p = params[0];
                return `${p.name}<br/>置信度: <b>${p.value}%</b>`;
            }
        }
    });

    window.addEventListener('resize', () => {
        if (confidenceChart) confidenceChart.resize();
    });
}

function initWordFreqChart() {
    const el = document.getElementById('wordFreqChart');
    if (!el) return;

    wordFreqChart = echarts.init(el, 'dark');
    wordFreqChart.setOption({
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'item',
            backgroundColor: 'rgba(250, 247, 242, 0.95)',
            borderColor: '#E8E0D5',
            textStyle: { color: '#2C2420' },
            formatter: '{b}: {c}次 ({d}%)'
        },
        legend: {
            orient: 'vertical',
            right: 10,
            top: 'center',
            textStyle: { color: '#9B8B7A', fontSize: 11 }
        },
        series: [{
            type: 'pie',
            radius: ['45%', '70%'],
            center: ['40%', '50%'],
            avoidLabelOverlap: false,
            label: { show: false },
            emphasis: {
                label: { show: true, fontSize: 14, fontWeight: 'bold' }
            },
            labelLine: { show: false },
            data: [],
            itemStyle: {
                borderRadius: 6,
                borderColor: '#FFFFFF',
                borderWidth: 2
            }
        }],
        color: [
            '#D4A574', '#8B9A7D', '#C17F59', '#5A6B7C', '#E8C9A0',
            '#A8B89A', '#D49A7A', '#7A8B6D', '#B8A090', '#9B8B7A'
        ]
    });

    window.addEventListener('resize', () => {
        if (wordFreqChart) wordFreqChart.resize();
    });
}

function updateCharts(predictions) {
    if (!confidenceChart || !predictions || predictions.length === 0) return;

    const top5 = predictions.slice(0, 5);
    confidenceChart.setOption({
        xAxis: { data: top5.map(p => p.word) },
        series: [{ data: top5.map(p => (p.confidence * 100).toFixed(1)) }]
    });
}

function updateWordFreqChart(wordFreq) {
    if (!wordFreqChart) return;

    const data = Object.entries(wordFreq)
        .map(([name, value]) => ({ name, value }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 10);

    wordFreqChart.setOption({
        series: [{ data }]
    });
}

function clearCharts() {
    if (confidenceChart) {
        confidenceChart.setOption({
            xAxis: { data: [] },
            series: [{ data: [] }]
        });
    }
    if (wordFreqChart) {
        wordFreqChart.setOption({
            series: [{ data: [] }]
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    setTimeout(initCharts, 100);
});
