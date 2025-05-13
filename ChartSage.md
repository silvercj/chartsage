# ChartSage Project Documentation

## 1. Overview

ChartSage is a web application designed to ingest user-uploaded data files (CSV/Excel), leverage AI to generate insightful analysis and chart specifications, and display these as interactive visualizations in a dynamic dashboard.

This document serves as the central knowledge base for the ChartSage project, covering its architecture, setup, API, database schema, and ongoing development.

## 2. Features

- **File Upload:** Supports uploading data files (e.g., CSV, Excel).
- **AI-Powered Insights:** Leverages AI (Claude) to analyze data and generate insights.
- **Interactive Visualizations:** Displays insights through various chart types (bar, scatter, pie, box) using ECharts.
- **Dynamic Dashboard:** Presents visualizations in a user-friendly dashboard.
- **Data Preview:** Allows users to preview uploaded data.
- **Persistent Sessions:** Uses Redis to store dashboard data for user sessions.
- **Robust CSV Parsing:** Employs PapaParse for handling complex CSV files.
- **Lowercase Column Name Normalization:** Ensures consistency by converting all column names to lowercase.
- **Context-Aware Data Formatting:** Applies appropriate formatting (currency, percentage, etc.) in charts.
- **AI-Generated Chart Explanations:** Provides business-friendly explanations for each chart.
- **[Add other key features as they are developed]**

## 3. Tech Stack

### 3.1. Frontend
- **Framework:** Next.js (React)
- **Language:** TypeScript
- **Charting Library:** ECharts (via `echarts-for-react`)
- **Styling:** Tailwind CSS
- **CSV Parsing:** PapaParse

### 3.2. Backend
- **Framework:** FastAPI
- **Language:** Python
- **AI Integration:** Anthropic Claude API
- **Data Handling:** Pandas, NumPy
- **Session Store:** Redis

### 3.3. Database
- **[Specify database if used, e.g., PostgreSQL, SQLite]**
No traditional relational database is currently used. Redis is employed for session data storage.

### 3.4. Other Tools
- **Version Control:** Git
- **Package Managers:** npm (frontend), pip (backend)
- **[Add any other relevant tools or services]**

## 4. Project Structure

```
ChartSage/
├── ChartSage.md        # This file
├── src/
│   ├── api/            # Backend (FastAPI)
│   │   ├── main.py     # Main FastAPI application
│   │   ├── insight_prompt.txt # Prompt for AI insight generation
│   │   ├── derived_fields.py  # Logic for computing derived data fields
│   │   ├── chart_processing.py # Logic for processing chart specifications
│   │   └── ...         # Other backend modules
│   └── app/            # Frontend (Next.js)
│       ├── visualizations/ # Visualization display components
│       │   ├── VisualizationCard.tsx
│       │   └── page.tsx
│       ├── page.tsx    # Main landing/upload page
│       └── ...         # Other frontend components and pages
├── .gitignore
├── package.json
├── tsconfig.json
├── next.config.js
├── requirements.txt    # Or Pipfile/pyproject.toml
└── ...                 # Other configuration files
```

## 5. Setup and Installation

### 5.1. Prerequisites
- Node.js (version [specify version, e.g., 18.x or higher])
- Python (version [specify version, e.g., 3.9 or higher])
- pip (Python package installer)
- Redis (running locally or accessible via network)
- [Add any other prerequisites, e.g., Git]

### 5.2. Backend Setup
1.  Navigate to the `src/api` directory:
    ```bash
    cd src/api
    ```
2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Set up environment variables:
    - Create a `.env` file in `src/api` based on `.env.example` (if one exists).
    - Ensure `ANTHROPIC_API_KEY` is set.
    - Ensure Redis connection details are correct (defaults to `localhost:6379`).
5.  Start the FastAPI server:
    ```bash
    python main.py
    ```
    The backend API should now be running, typically at `http://localhost:8000`.

### 5.3. Frontend Setup
1.  Navigate to the project root directory.
2.  Install Node.js dependencies:
    ```bash
    npm install
    ```
3.  Set up environment variables:
    - Create a `.env.local` file in the project root if needed (e.g., for `NEXT_PUBLIC_API_URL`).
    - By default, `NEXT_PUBLIC_API_URL` should point to your backend (e.g., `http://localhost:8000`).
4.  Start the Next.js development server:
    ```bash
    npm run dev
    ```
    The frontend should now be accessible, typically at `http://localhost:3000`.

### 5.4. Redis Setup
- Ensure Redis is installed and running.
- On macOS with Homebrew:
    ```bash
    brew install redis
    brew services start redis
    ```
- Verify Redis is running:
    ```bash
    redis-cli ping
    ```
  (Should return `PONG`)

## 6. API Documentation

*(This section should detail all API endpoints, request/response formats, and authentication methods.)*

### 6.1. `/upload`
- **Method:** POST
- **Description:** Uploads a data file (CSV/Excel) for processing.
- **Request Body:** `multipart/form-data` with a `file` field.
- **Response:** JSON object containing a data preview (`columns`, `data`).
- **Example:**
  ```json
  // Response
  {
    "status": "success",
    "preview": {
      "columns": ["column1", "column2"],
      "data": [{"column1": "value1", "column2": "value2"}]
    }
  }
  ```

### 6.2. `/generate-dashboard`
- **Method:** POST
- **Description:** Generates AI insights and chart visualizations from an uploaded file.
- **Request Body:** `multipart/form-data` with a `file` field.
- **Response:** JSON object containing `status` and `visualizations` array.
- **Example:**
  ```json
  // Response
  {
    "status": "success",
    "visualizations": [ /* ...array of chart objects... */ ]
  }
  ```

### 6.3. `/session-dashboard`
- **Method:** POST
- **Description:** Stores generated dashboard visualizations in a server-side session (Redis).
- **Request Body:** JSON array of visualization objects.
- **Response:** JSON object with a `session_id`.
- **Example:**
  ```json
  // Request
  [ /* ...array of chart objects... */ ]
  // Response
  { "session_id": "unique-session-identifier" }
  ```

### 6.4. `/session-dashboard/{session_id}`
- **Method:** GET
- **Description:** Retrieves dashboard visualizations from a server-side session using the `session_id`.
- **Response:** JSON array of visualization objects.

### 6.5. Authentication
Currently, there is no specific user authentication implemented for accessing ChartSage's own API endpoints. The backend API endpoints are open.

The Anthropic API key is required for the backend to communicate with the Claude AI service. This key is configured as an environment variable on the backend server and is not exposed to the client.

## 7. Database Schema

*(This section should document the structure of your database tables, columns, relationships, and constraints if a database is used.)*

**[If not using a relational database, describe data storage strategy, e.g., for Redis session data.]**
ChartSage does not use a traditional relational database. Session data related to dashboards is stored in Redis.

### 7.1. Redis Session Data for Dashboards
- **Key Format:** `dashboard:<session_id>`
- **Value:** JSON string representing an array of visualization objects.
- **Expiration:** Typically 24 hours.

## 8. Migrations Log

*(This section should log all database schema migrations, including the date, version, and a brief description of changes. If not using a formal migration tool, manually log schema changes here.)*

- **[Date] - [Version/Description]:** Initial schema setup for [Feature X].
- **[Date] - [Version/Description]:** Added `new_column` to `table_name`.
Not applicable. ChartSage currently uses Redis for session storage, which is schema-less. If a relational database is added in the future, this section will track its schema migrations.

## 9. AI Prompt Design (`insight_prompt.txt`)

The `src/api/insight_prompt.txt` file contains the master prompt used to instruct the Claude AI model for generating visualizations. Key aspects of the prompt include:
- **Role Definition:** Senior Business Analyst.
- **Supported Chart Types:** bar, scatter, pie, box.
- **Derived Field Rules:** Specifies allowed patterns for derived fields (ratios, histograms, sums, etc.) and the requirement for a `derived_from` key.
- **JSON Schema:** Defines the expected structure for each visualization object.
- **Data Formatting:** Includes rules for lowercase column names, field type hints (`<field>_type`), display type hints (`<field>_display_type`), and chart explanations.
- **Output Constraints:** Requires exactly 10 JSON objects in an array, no prose.
- **Context Injection:** Placeholders for `data_sample`, `numeric_columns`, `categorical_columns`, `row_count`, and `summary_stats`.

**Note:** When modifying the prompt, ensure all JSON examples are valid and placeholders are correctly formatted (e.g., `{{` and `}}` for literal braces in JSON examples if using Python's `.format()` method).

## 10. Key Decisions & Architecture Notes

- **Session Management:** Dashboard data is passed via server-side Redis sessions to avoid large URL query strings (HTTP 431 error).
- **Column Name Handling:** All column names are converted to lowercase on file upload (frontend preview and backend processing) to ensure consistency and prevent case-sensitivity issues.
- **Logging:** Backend logs to both console and `chartsage.log` to capture all output. Frontend logs to browser console.
- **PDF Generation Removed:** Initial attempts to include PDF report generation faced significant challenges with layout and chart rendering. The feature was subsequently removed to focus on core dashboard functionality and improve development velocity.
- **CSV Parsing:** Switched to PapaParse on the frontend for robust handling of various CSV file complexities, including quoted fields, embedded commas, and newlines.
- **AI Prompt Engineering:** The prompt used for the Claude AI (`src/api/insight_prompt.txt`) has undergone several iterations. This refinement process is crucial for improving the quality, consistency, and structure of AI-generated chart specifications and textual insights. Key aspects include defining the AI's role, specifying allowed derived fields, mandating field type and display type hints, and ensuring lowercase column names.
- **Error Handling:** Emphasis has been placed on improving backend error handling, particularly for external API calls (e.g., Claude API overload errors returning 529/503). These are now propagated more clearly to the frontend to provide better user feedback (e.g., "AI is busy" messages) rather than generic server errors.
- **[Add other important architectural decisions or notes here]**

## 11. Contribution Guidelines

*(Optional: Add guidelines if others will contribute to the project.)*
*(This section can be expanded with specific project conventions if collaborators join or the project grows.)*
- **Branching Strategy:** (e.g., Gitflow, feature branches)
- **Code Style:** (e.g., PEP 8 for Python, Prettier for TypeScript)
- **Testing:** (e.g., preferred testing frameworks, coverage expectations)
- **Commit Messages:** (e.g., Conventional Commits format)
- **Pull Requests:** (e.g., review process, required checks)

## 12. Future Enhancements / Roadmap

*(List potential future features, improvements, or areas for refactoring.)*
- [ ] Feature X: [Description]
- [ ] Improvement Y: [Description]
- [ ] Refactor Z: [Description]

---

*This document should be kept up-to-date as the project evolves.* 