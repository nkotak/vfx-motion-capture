/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
      {
        source: '/outputs/:path*',
        destination: 'http://localhost:8000/outputs/:path*',
      },
    ]
  },
  images: {
    domains: ['localhost'],
  },
}

module.exports = nextConfig
