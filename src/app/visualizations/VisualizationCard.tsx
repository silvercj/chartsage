import React from 'react';
import ReactECharts from 'echarts-for-react';

interface VisualizationCardProps {
  viz: any;
  index: number;
  plotData: any;
  handleDownload: (index: number, title: string) => void;
  layout: any;
  config: any;
  plotRef?: (el: any) => void;
}

function getEChartsOption(viz: any, plotData: any, layout: any) {
  // Defensive: always use arrays
  const xArr = Array.isArray(plotData?.x) ? plotData.x : [];
  const yArr = Array.isArray(plotData?.y) ? plotData.y : [];
  const labelsArr = Array.isArray(plotData?.labels) ? plotData.labels : [];
  const valuesArr = Array.isArray(plotData?.values) ? plotData.values : [];
  const textArr = Array.isArray(plotData?.text) ? plotData.text : [];
  const type = viz?.type;

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
    option.yAxis = { type: 'value', name: layout?.yaxis?.title };
    option.series = [{
      type: 'bar',
      data: yArr,
      name: viz?.title,
      itemStyle: { color: viz?.style?.color_scheme || '#5470C6' },
      label: { show: viz?.style?.show_values ?? false, position: 'top' }
    }];
  } else if (type === 'line') {
    option.xAxis = { type: 'category', data: xArr, name: layout?.xaxis?.title };
    option.yAxis = { type: 'value', name: layout?.yaxis?.title };
    option.series = [{
      type: 'line',
      data: yArr,
      name: viz?.title,
      smooth: true,
      lineStyle: { width: viz?.style?.line_width || 2, color: viz?.style?.color_scheme || '#5470C6' },
      symbolSize: viz?.style?.marker_size || 10
    }];
  } else if (type === 'scatter') {
    option.xAxis = { type: 'value', name: layout?.xaxis?.title };
    option.yAxis = { type: 'value', name: layout?.yaxis?.title };
    option.series = [{
      type: 'scatter',
      data: xArr.map((x: any, i: number) => [x, yArr[i]]),
      name: viz?.title,
      symbolSize: viz?.style?.marker_size || 10,
      itemStyle: { color: viz?.style?.color_scheme || '#5470C6' },
      label: {
        show: !!(textArr && textArr.length),
        formatter: (params: any) => textArr ? textArr[params.dataIndex] : '',
        position: 'top'
      }
    }];
    option.tooltip = {
      trigger: 'item',
      formatter: (params: any) => {
        const label = textArr ? textArr[params.dataIndex] : '';
        return `${label ? label + '<br/>' : ''}X: ${params.value[0]}<br/>Y: ${params.value[1]}`;
      }
    };
  } else if (type === 'pie') {
    option.series = [{
      type: 'pie',
      data: (labelsArr.length ? labelsArr : xArr).map((label: any, i: number) => ({
        name: label,
        value: (valuesArr.length ? valuesArr : yArr)[i]
      })),
      radius: '60%',
      label: { show: true, formatter: '{b}: {d}%'}
    }];
    option.legend = { show: true, bottom: 0 };
    option.tooltip = { trigger: 'item', formatter: '{b}: {c} ({d}%)' };
    option.xAxis = undefined;
    option.yAxis = undefined;
  }
  // Debug: log the final ECharts option
  console.log('[ECharts] Final option for chart', viz?.title, option);
  return option;
}

export default function VisualizationCard({ viz, index, plotData, handleDownload, layout, config, plotRef }: VisualizationCardProps) {
  return (
    <section
      className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8 flex flex-col justify-between min-h-[480px]"
      aria-label={viz?.title || `Chart ${index + 1}`}
    >
      <div className="flex items-start justify-between mb-2">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-1 leading-tight">{viz?.title || `Chart ${index + 1}`}</h2>
          <p className="text-md text-blue-700 mb-2 font-medium">{viz?.description || 'No description available'}</p>
        </div>
        <button
          className="ml-2 px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg shadow transition"
          onClick={() => handleDownload(index, viz?.title || `chart_${index + 1}`)}
          aria-label={`Download ${viz?.title || `Chart ${index + 1}`} as PNG`}
        >
          Download
        </button>
      </div>
      <div className="flex-1 flex items-center justify-center min-h-[320px]">
        <ReactECharts
          option={getEChartsOption(viz, plotData, layout)}
          style={{ width: '100%', height: 400 }}
        />
      </div>
    </section>
  );
} 