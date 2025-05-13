import React from 'react';
import ReactECharts from 'echarts-for-react';

interface VisualizationCardProps {
  viz: any;
  index: number;
  plotData: any;
  layout: any;
}

function shortCurrency(value: number) {
  if (value >= 1e9) return '$' + (value / 1e9).toFixed(1) + 'B';
  if (value >= 1e6) return '$' + (value / 1e6).toFixed(1) + 'M';
  if (value >= 1e3) return '$' + (value / 1e3).toFixed(1) + 'K';
  return '$' + value;
}

function getFormatter(displayType: string | undefined) {
  if (displayType === 'currency') return shortCurrency;
  if (displayType === 'correlation') return (v: number) => v?.toFixed(2);
  if (displayType === 'percentage') return (v: number) => (v * 100).toFixed(1) + '%';
  if (displayType === 'count') return (v: number) => v;
  return (v: number) => v;
}

function getEChartsOption(viz: any, plotData: any, layout: any) {
  // Defensive: always use arrays
  const xArr = Array.isArray(plotData?.x) ? plotData.x : [];
  const yArr = Array.isArray(plotData?.y) ? plotData.y : [];
  const labelsArr = Array.isArray(plotData?.labels) ? plotData.labels : [];
  const valuesArr = Array.isArray(plotData?.values) ? plotData.values : [];
  const textArr = Array.isArray(plotData?.text) ? plotData.text : [];
  const type = viz?.type;

  // Get display types for formatting
  const xDisplayType = plotData?.x_display_type;
  const yDisplayType = plotData?.y_display_type;
  const valuesDisplayType = plotData?.values_display_type;

  const yFormatter = getFormatter(yDisplayType);
  const xFormatter = getFormatter(xDisplayType);
  const valuesFormatter = getFormatter(valuesDisplayType);

  // Debug: log chart type and data arrays
  console.log('[ECharts] Rendering chart', {
    chartType: type,
    title: viz?.title,
    x: xArr,
    y: yArr,
    labels: labelsArr,
    values: valuesArr,
    text: textArr,
    plotData,
    layout,
    viz
  });
  if (!xArr.length && !yArr.length) {
    console.warn(`[ECharts] Chart "${viz?.title}" of type "${type}" has empty data arrays.`);
  }

  let option: any = {
    title: {
      text: viz?.title || '',
      left: 'center',
      top: 20,
      textStyle: { fontSize: 22, fontWeight: 700 }
    },
    tooltip: { trigger: 'item' },
    legend: { show: layout?.showlegend ?? true, bottom: 0 },
    grid: { left: 60, right: 40, top: 60, bottom: 60 },
    xAxis: {},
    yAxis: {},
    series: []
  };
  if (type === 'bar') {
    option.xAxis = { type: 'category', data: xArr };
    option.yAxis = { type: 'value', name: layout?.yaxis?.title, axisLabel: { formatter: yFormatter } };
    option.series = [{
      type: 'bar',
      data: yArr,
      name: viz?.title,
      itemStyle: { color: viz?.style?.color_scheme || '#5470C6' },
      label: {
        show: viz?.style?.show_values ?? false,
        position: 'top',
        formatter: (params: any) => yFormatter(params.value)
      }
    }];
  } else if (type === 'line') {
    option.xAxis = { type: 'category', data: xArr, name: layout?.xaxis?.title };
    option.yAxis = { type: 'value', name: layout?.yaxis?.title, axisLabel: { formatter: yFormatter } };
    option.series = [{
      type: 'line',
      data: yArr,
      name: viz?.title,
      smooth: true,
      lineStyle: { width: viz?.style?.line_width || 2, color: viz?.style?.color_scheme || '#5470C6' },
      symbolSize: viz?.style?.marker_size || 10,
      label: {
        show: viz?.style?.show_values ?? false,
        position: 'top',
        formatter: (params: any) => yFormatter(params.value)
      }
    }];
  } else if (type === 'scatter') {
    option.xAxis = { type: 'value', name: layout?.xaxis?.title, axisLabel: { formatter: xFormatter } };
    option.yAxis = { type: 'value', name: layout?.yaxis?.title, axisLabel: { formatter: yFormatter } };
    option.series = [{
      type: 'scatter',
      data: xArr.map((x: any, i: number) => [x, yArr[i]]),
      name: viz?.title,
      symbolSize: viz?.style?.marker_size || 10,
      itemStyle: { color: viz?.style?.color_scheme || '#5470C6' },
      label: {
        show: !!(labelsArr && labelsArr.length),
        formatter: (params: any) => labelsArr ? labelsArr[params.dataIndex] : '',
        position: 'top'
      }
    }];
    option.tooltip = {
      trigger: 'item',
      formatter: (params: any) => textArr ? textArr[params.dataIndex] : ''
    };
  } else if (type === 'pie') {
    option.series = [{
      type: 'pie',
      data: (labelsArr.length ? labelsArr : xArr).map((label: any, i: number) => ({
        name: label,
        value: (valuesArr.length ? valuesArr : yArr)[i]
      })),
      radius: '60%',
      label: {
        show: true,
        formatter: (params: any) => `${params.name}: ${valuesFormatter(params.value)} (${params.percent}%)`
      }
    }];
    option.legend = { show: true, bottom: 0 };
    option.tooltip = {
      trigger: 'item',
      formatter: (params: any) => `${params.name}: ${valuesFormatter(params.value)} (${params.percent}%)`
    };
    option.xAxis = undefined;
    option.yAxis = undefined;
  } else if (type === 'box' || type === 'boxplot') {
    // Box plot logic: ECharts expects a 2D array for boxplot data
    // If only x or y is present, use that as the data
    // If both are present and non-empty, use [xArr, yArr] as two boxes
    let boxData: any[] = [];
    let categories: string[] = [];
    if (xArr.length && yArr.length) {
      // Both x and y present: treat as two groups
      boxData = [xArr, yArr];
      categories = ['x', 'y'];
    } else if (xArr.length) {
      // Only x present
      boxData = [xArr];
      categories = [viz?.layout?.xaxis_title || ''];
    } else if (yArr.length) {
      // Only y present
      boxData = [yArr];
      categories = [viz?.layout?.yaxis_title || ''];
    } else {
      // No data
      boxData = [[]];
      categories = [''];
    }
    // ECharts expects each item in boxData to be an array of numbers
    // Defensive: filter out non-numeric values
    boxData = boxData.map(arr => (Array.isArray(arr) ? arr.filter(v => typeof v === 'number' && !isNaN(v)) : []));
    // Debug log
    console.log('[ECharts][BoxPlot] boxData:', boxData, 'categories:', categories);
    option.xAxis = { type: 'category', data: categories, name: viz?.layout?.xaxis_title };
    option.yAxis = { type: 'value', name: viz?.layout?.yaxis_title };
    option.series = [{
      type: 'boxplot',
      data: boxData,
      name: viz?.title,
      itemStyle: { color: viz?.style?.color_scheme || '#91cc75' },
      tooltip: {
        formatter: function(param: any) {
          // param.data is [min, Q1, median, Q3, max] if using ECharts' boxplot transform
          if (Array.isArray(param.data)) {
            return `Min: ${param.data[0]}<br>Q1: ${param.data[1]}<br>Median: ${param.data[2]}<br>Q3: ${param.data[3]}<br>Max: ${param.data[4]}`;
          }
          return '';
        }
      }
    }];
    // Remove legend if not needed
    option.legend = { show: viz?.style?.show_legend ?? false };
    // Remove grid if not needed
    option.grid = option.grid || { left: 60, right: 40, top: 60, bottom: 60 };
  }
  // Debug: log the final ECharts option
  console.log('[ECharts] Final option for chart', viz?.title, option);
  return option;
}

export default function VisualizationCard({ viz, index, plotData, layout }: VisualizationCardProps) {
  // Use the new top-level explanation field
  const explanation = viz?.explanation;

  return (
    <section
      className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8 flex flex-col justify-between min-h-[480px]"
      aria-label={viz?.title || `Chart ${index + 1}`}
    >
      <div className="flex items-start justify-between mb-2">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-1 leading-tight">{viz?.title || `Chart ${index + 1}`}</h2>
          {viz?.description && viz.description.trim() && (
            <p className="text-md text-blue-700 mb-2 font-medium">{viz.description}</p>
          )}
        </div>
      </div>
      <div className="flex-1 flex items-center justify-center min-h-[320px]">
        <ReactECharts
          option={getEChartsOption(viz, plotData, layout)}
          style={{ width: '100%', height: 400 }}
        />
      </div>
      {explanation && (
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded text-blue-900 text-base">
          <strong>What this chart shows:</strong> {explanation}
        </div>
      )}
    </section>
  );
} 