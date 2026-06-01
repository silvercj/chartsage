import type { Report } from '../../report/[id]/useReportLayout';

// Synthetic sales report used only for the marketing example embed.
// No real user data — every number here is made up.
export const SAMPLE_REPORT: Report = {
  generated_at: '2024-12-31T00:00:00Z',
  summary:
    'Revenue grew 22% year over year, led by the West region at 31% of total. Order volume rose alongside average order value, and Q4 closed above target. One data note: 8% of rows have a blank discount_code.',
  data_quality: [
    '8% of rows in "discount_code" are blank — those orders are counted as "none".',
  ],
  charts: [
    {
      chart_id: 'c1',
      caption: 'Revenue rose steadily through the year and closed December above target.',
      spec: {
        kind: 'line',
        title: 'Revenue by month',
        x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y: [320, 340, 360, 355, 390, 410, 430, 425, 470, 510, 560, 620],
        x_label: 'Month',
        y_label: 'Revenue ($k)',
        y_display_type: 'number',
      },
    },
    {
      chart_id: 'c2',
      caption: 'The West region leads at roughly a third of total revenue.',
      spec: {
        kind: 'bar',
        title: 'Revenue by region',
        x: ['West', 'East', 'North', 'South', 'Intl'],
        y: [1490, 1180, 940, 760, 430],
        x_label: 'Region',
        y_label: 'Revenue ($k)',
        y_display_type: 'number',
      },
    },
    {
      chart_id: 'c3',
      caption: 'Most orders come through Direct and Online, with Partner and Reseller behind.',
      spec: {
        kind: 'pie',
        title: 'Orders by channel',
        x: ['Direct', 'Online', 'Partner', 'Reseller', 'Other'],
        y: [4200, 3650, 1980, 1240, 430],
        x_label: 'Channel',
        y_label: 'Orders',
        y_display_type: 'number',
      },
    },
  ],
  layout: [
    { chart_id: 'c1', position: 'main', order: 0 },
    { chart_id: 'c2', position: 'main', order: 1 },
    { chart_id: 'c3', position: 'main', order: 2 },
  ],
  metadata: {},
};
