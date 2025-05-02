# ChartSage

Turn any Excel file into a beautiful, interactive dashboard with AI-generated insights in seconds.

## Features

- Upload Excel/CSV files and get instant visualizations
- AI-powered insights and recommendations
- Interactive dashboards with Plotly.js
- Export to PDF/PowerPoint
- Shareable web links

## Tech Stack

- Frontend: Next.js 14, React, TypeScript, Tailwind CSS
- Backend: FastAPI, Python
- AI: Claude API
- Data Processing: pandas, numpy
- Visualization: Plotly.js
- Authentication: NextAuth.js
- Payments: Stripe

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env.local
   ```
4. Start the development server:
   ```bash
   npm run dev
   ```

## Environment Variables

- `NEXT_PUBLIC_API_URL`: Backend API URL
- `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY`: Stripe publishable key
- `STRIPE_SECRET_KEY`: Stripe secret key
- `ANTHROPIC_API_KEY`: Claude API key
- `NEXTAUTH_SECRET`: NextAuth secret
- `NEXTAUTH_URL`: NextAuth URL

## Project Structure

```
src/
├── app/              # Next.js app directory
├── components/       # React components
├── lib/             # Utility functions
├── styles/          # Global styles
└── types/           # TypeScript types
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 