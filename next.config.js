/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
    return [
      {
        source: '/report/:id/embed',
        headers: [{ key: 'Content-Security-Policy', value: 'frame-ancestors *' }],
      },
    ];
  },
};

module.exports = nextConfig;
