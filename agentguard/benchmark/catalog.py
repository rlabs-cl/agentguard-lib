"""Benchmark spec catalog — predefined development requests per category and complexity.

Each archetype category has specs at 5 complexity levels:
  trivial → low → medium → high → enterprise

Authors must select at least one spec per complexity level when running benchmarks.
They can also provide custom specs.
"""

from __future__ import annotations

from agentguard.benchmark.types import BenchmarkSpec, Complexity

# ══════════════════════════════════════════════════════════════════
#  CATALOG: category → complexity → list[BenchmarkSpec]
# ══════════════════════════════════════════════════════════════════

BENCHMARK_CATALOG: dict[str, dict[Complexity, list[BenchmarkSpec]]] = {
    # ── Backend ───────────────────────────────────────────────
    "backend": {
        Complexity.TRIVIAL: [
            BenchmarkSpec(
                Complexity.TRIVIAL,
                "Create a REST API with a single GET /hello endpoint that returns {'message': 'Hello, World!'}.",
                "backend",
            ),
            BenchmarkSpec(
                Complexity.TRIVIAL,
                "Create an API with a single POST /echo endpoint that returns the JSON body it receives.",
                "backend",
            ),
        ],
        Complexity.LOW: [
            BenchmarkSpec(
                Complexity.LOW,
                "Create a REST API for a todo list with GET /todos, POST /todos, and DELETE /todos/{id} using in-memory storage.",
                "backend",
            ),
            BenchmarkSpec(
                Complexity.LOW,
                "Create a REST API that converts temperatures between Celsius, Fahrenheit, and Kelvin with input validation.",
                "backend",
            ),
        ],
        Complexity.MEDIUM: [
            BenchmarkSpec(
                Complexity.MEDIUM,
                "Build a REST API for a bookstore with CRUD operations, search by title/author, pagination, and SQLite persistence.",
                "backend",
            ),
            BenchmarkSpec(
                Complexity.MEDIUM,
                "Build a URL shortener API with POST /shorten, GET /{code} redirect, GET /stats/{code} analytics, and persistent storage.",
                "backend",
            ),
        ],
        Complexity.HIGH: [
            BenchmarkSpec(
                Complexity.HIGH,
                "Build a REST API for a task management system with user authentication (JWT), PostgreSQL database, role-based access control (admin/user), task assignment, due dates, status transitions, and input validation.",
                "backend",
            ),
            BenchmarkSpec(
                Complexity.HIGH,
                "Build a REST API for an e-commerce product catalog with categories, product search with filters, inventory management, image upload references, and comprehensive error handling.",
                "backend",
            ),
        ],
        Complexity.ENTERPRISE: [
            BenchmarkSpec(
                Complexity.ENTERPRISE,
                "Build a microservice for order processing with JWT authentication, PostgreSQL, Redis caching, rate limiting, background task queue for email notifications, health checks, structured logging, OpenTelemetry tracing, graceful shutdown, and database migrations.",
                "backend",
            ),
        ],
    },
    # ── CLI ────────────────────────────────────────────────────
    "cli": {
        Complexity.TRIVIAL: [
            BenchmarkSpec(
                Complexity.TRIVIAL,
                "Create a CLI tool that takes a name as argument and prints 'Hello, {name}!'.",
                "cli",
            ),
        ],
        Complexity.LOW: [
            BenchmarkSpec(
                Complexity.LOW,
                "Create a CLI tool that converts temperatures between Celsius, Fahrenheit, and Kelvin with --from and --to flags.",
                "cli",
            ),
            BenchmarkSpec(
                Complexity.LOW,
                "Create a CLI tool that counts words, lines, and characters in a given file, similar to wc.",
                "cli",
            ),
        ],
        Complexity.MEDIUM: [
            BenchmarkSpec(
                Complexity.MEDIUM,
                "Build a CLI tool that parses Excel files (.xlsx) to JSON with column filtering, sheet selection, and output to file or stdout.",
                "cli",
            ),
            BenchmarkSpec(
                Complexity.MEDIUM,
                "Build a CLI tool for managing a local JSON-based task list with add, remove, list, complete, and search subcommands.",
                "cli",
            ),
        ],
        Complexity.HIGH: [
            BenchmarkSpec(
                Complexity.HIGH,
                "Build a multi-format log analyzer CLI that reads log files (JSON, syslog, Apache), supports streaming with --follow, regex filtering, time-range queries, and outputs aggregation reports (error counts, top endpoints, p95 latency).",
                "cli",
            ),
        ],
        Complexity.ENTERPRISE: [
            BenchmarkSpec(
                Complexity.ENTERPRISE,
                "Build a CLI tool for database migration management with init, create, up, down, status, and rollback subcommands. Support PostgreSQL and SQLite, track migration history, generate timestamped migration files, handle concurrent migrations with advisory locks, dry-run mode, and structured logging.",
                "cli",
            ),
        ],
    },
    # ── Frontend ──────────────────────────────────────────────
    "frontend": {
        Complexity.TRIVIAL: [
            BenchmarkSpec(
                Complexity.TRIVIAL,
                "Create a counter component with increment and decrement buttons and a display showing the current count.",
                "frontend",
            ),
        ],
        Complexity.LOW: [
            BenchmarkSpec(
                Complexity.LOW,
                "Create a Pong game with two paddles, a ball, score tracking, and keyboard controls.",
                "frontend",
            ),
            BenchmarkSpec(
                Complexity.LOW,
                "Create a tip calculator with bill amount input, tip percentage slider, number of people split, and formatted results.",
                "frontend",
            ),
        ],
        Complexity.MEDIUM: [
            BenchmarkSpec(
                Complexity.MEDIUM,
                "Build a todo application with add, edit, delete, mark complete, filter by status, and localStorage persistence.",
                "frontend",
            ),
            BenchmarkSpec(
                Complexity.MEDIUM,
                "Build a weather dashboard that fetches data from a mock API, displays current conditions and 5-day forecast, with city search and unit toggle.",
                "frontend",
            ),
        ],
        Complexity.HIGH: [
            BenchmarkSpec(
                Complexity.HIGH,
                "Build a Kanban board with drag-and-drop between columns (To Do, In Progress, Done), card creation/editing, labels, due dates, search, and localStorage persistence.",
                "frontend",
            ),
        ],
        Complexity.ENTERPRISE: [
            BenchmarkSpec(
                Complexity.ENTERPRISE,
                "Build an admin dashboard with authentication, role-based access, real-time data charts (line, bar, pie), data tables with sorting/filtering/pagination, form validation, dark mode, keyboard navigation, ARIA compliance, error boundaries, centralized strings, and responsive layout with 3 breakpoints.",
                "frontend",
            ),
        ],
    },
    # ── Fullstack ─────────────────────────────────────────────
    "fullstack": {
        Complexity.TRIVIAL: [
            BenchmarkSpec(
                Complexity.TRIVIAL,
                "Create a guestbook app with a form to submit name+message and a list of all messages.",
                "fullstack",
            ),
        ],
        Complexity.LOW: [
            BenchmarkSpec(
                Complexity.LOW,
                "Create a simple notes app with a backend API and frontend UI for creating, listing, and deleting notes.",
                "fullstack",
            ),
        ],
        Complexity.MEDIUM: [
            BenchmarkSpec(
                Complexity.MEDIUM,
                "Build a blog platform with user registration, post creation with markdown, comment system, and a responsive frontend.",
                "fullstack",
            ),
        ],
        Complexity.HIGH: [
            BenchmarkSpec(
                Complexity.HIGH,
                "Build a project management tool with user auth, team workspaces, task boards, file attachments, notifications, and a real-time dashboard.",
                "fullstack",
            ),
        ],
        Complexity.ENTERPRISE: [
            BenchmarkSpec(
                Complexity.ENTERPRISE,
                "Build a multi-tenant SaaS platform with JWT auth, role-based access, Stripe billing integration, usage metering, admin panel, audit logging, rate limiting, and API versioning.",
                "fullstack",
            ),
        ],
    },
    # ── Library ───────────────────────────────────────────────
    "library": {
        Complexity.TRIVIAL: [
            BenchmarkSpec(
                Complexity.TRIVIAL,
                "Create a library with a single function that validates email addresses using regex.",
                "library",
            ),
        ],
        Complexity.LOW: [
            BenchmarkSpec(
                Complexity.LOW,
                "Create a library for parsing and formatting dates with timezone support.",
                "library",
            ),
        ],
        Complexity.MEDIUM: [
            BenchmarkSpec(
                Complexity.MEDIUM,
                "Build a retry library with exponential backoff, jitter, max retries, timeout, and configurable exception filtering.",
                "library",
            ),
        ],
        Complexity.HIGH: [
            BenchmarkSpec(
                Complexity.HIGH,
                "Build a schema validation library supporting nested objects, arrays, optional fields, custom validators, error aggregation, and type coercion.",
                "library",
            ),
        ],
        Complexity.ENTERPRISE: [
            BenchmarkSpec(
                Complexity.ENTERPRISE,
                "Build a plugin system library with plugin discovery, dependency resolution, lifecycle hooks (init/start/stop), configuration injection, sandboxed execution, and hot-reload support.",
                "library",
            ),
        ],
    },
    # ── Script ────────────────────────────────────────────────
    "script": {
        Complexity.TRIVIAL: [
            BenchmarkSpec(
                Complexity.TRIVIAL,
                "Write a script that reads a CSV file and prints the number of rows and columns.",
                "script",
            ),
        ],
        Complexity.LOW: [
            BenchmarkSpec(
                Complexity.LOW,
                "Write a script that reads a CSV file, calculates descriptive statistics (mean, median, std dev) for numeric columns, and prints a summary table.",
                "script",
            ),
        ],
        Complexity.MEDIUM: [
            BenchmarkSpec(
                Complexity.MEDIUM,
                "Write a script that scrapes product data from a mock HTML page, cleans the data, and exports to both CSV and JSON formats with error handling.",
                "script",
            ),
        ],
        Complexity.HIGH: [
            BenchmarkSpec(
                Complexity.HIGH,
                "Write a data pipeline script that reads from multiple CSV sources, joins on common keys, handles missing values, computes aggregations, generates a matplotlib visualization, and exports a PDF report.",
                "script",
            ),
        ],
        Complexity.ENTERPRISE: [
            BenchmarkSpec(
                Complexity.ENTERPRISE,
                "Write a data processing pipeline with configurable sources (CSV, JSON, API), transformation chains, validation rules, error recovery, progress reporting, structured logging, and parallel processing for large datasets.",
                "script",
            ),
        ],
    },
    # ── Data / ML ─────────────────────────────────────────────
    "data": {
        Complexity.TRIVIAL: [
            BenchmarkSpec(
                Complexity.TRIVIAL,
                "Create a Jupyter notebook that loads the Iris dataset and prints basic statistics.",
                "data",
            ),
        ],
        Complexity.LOW: [
            BenchmarkSpec(
                Complexity.LOW,
                "Create a Jupyter notebook that loads the Titanic dataset, performs EDA with visualizations, and identifies missing value patterns.",
                "data",
            ),
        ],
        Complexity.MEDIUM: [
            BenchmarkSpec(
                Complexity.MEDIUM,
                "Create a data analysis pipeline that loads a dataset, cleans missing values, engineers features, trains a classification model, and reports accuracy metrics with cross-validation.",
                "data",
            ),
        ],
        Complexity.HIGH: [
            BenchmarkSpec(
                Complexity.HIGH,
                "Build a recommendation engine with collaborative filtering, content-based fallback, evaluation metrics (precision, recall, NDCG), A/B test framework, and a REST API serving layer.",
                "data",
            ),
        ],
        Complexity.ENTERPRISE: [
            BenchmarkSpec(
                Complexity.ENTERPRISE,
                "Build an ML pipeline with data versioning, feature store, model training with hyperparameter tuning, model registry, A/B deployment, monitoring for data drift, and automated retraining triggers.",
                "data",
            ),
        ],
    },
    # ── DevOps / Infra ────────────────────────────────────────
    "devops": {
        Complexity.TRIVIAL: [
            BenchmarkSpec(
                Complexity.TRIVIAL,
                "Create a Dockerfile for a Python Flask application.",
                "devops",
            ),
        ],
        Complexity.LOW: [
            BenchmarkSpec(
                Complexity.LOW,
                "Create a Docker Compose setup for a Python web app with PostgreSQL and Redis.",
                "devops",
            ),
        ],
        Complexity.MEDIUM: [
            BenchmarkSpec(
                Complexity.MEDIUM,
                "Build a CI/CD pipeline configuration (GitHub Actions) for a Python project with linting, testing, Docker build, and deployment to staging.",
                "devops",
            ),
        ],
        Complexity.HIGH: [
            BenchmarkSpec(
                Complexity.HIGH,
                "Build Terraform modules for deploying a web service to AWS with VPC, ECS Fargate, RDS, ALB, ACM, and CloudWatch monitoring.",
                "devops",
            ),
        ],
        Complexity.ENTERPRISE: [
            BenchmarkSpec(
                Complexity.ENTERPRISE,
                "Build a complete infrastructure-as-code setup with Terraform modules for multi-environment (dev/staging/prod), Kubernetes manifests with Helm charts, GitOps with ArgoCD, secrets management, monitoring stack (Prometheus + Grafana), and disaster recovery runbooks.",
                "devops",
            ),
        ],
    },
}

# ── Fallback: general category ────────────────────────────────────

BENCHMARK_CATALOG["general"] = {
    Complexity.TRIVIAL: [
        BenchmarkSpec(Complexity.TRIVIAL, "Create a calculator with add, subtract, multiply, and divide operations.", "general"),
    ],
    Complexity.LOW: [
        BenchmarkSpec(Complexity.LOW, "Create a Pong game with keyboard controls, score tracking, and basic collision physics.", "general"),
    ],
    Complexity.MEDIUM: [
        BenchmarkSpec(Complexity.MEDIUM, "Build a personal finance tracker with income/expense entries, categories, monthly summaries, and data persistence.", "general"),
    ],
    Complexity.HIGH: [
        BenchmarkSpec(Complexity.HIGH, "Build a real-time chat application with user registration, rooms, message history, typing indicators, and WebSocket communication.", "general"),
    ],
    Complexity.ENTERPRISE: [
        BenchmarkSpec(Complexity.ENTERPRISE, "Build a workflow automation engine with a visual DAG editor, conditional branching, retry policies, webhook triggers, scheduling, audit trail, and REST API.", "general"),
    ],
}

# Alias categories that share specs
for _alias, _source in [("ml", "data"), ("infra", "devops"), ("mobile", "frontend")]:
    if _alias not in BENCHMARK_CATALOG:
        BENCHMARK_CATALOG[_alias] = BENCHMARK_CATALOG[_source]


def get_specs_for_category(category: str) -> dict[Complexity, list[BenchmarkSpec]]:
    """Get all benchmark specs for a category, falling back to 'general'."""
    return BENCHMARK_CATALOG.get(category, BENCHMARK_CATALOG["general"])


def get_default_specs(category: str) -> list[BenchmarkSpec]:
    """Get one default spec per complexity level for a category."""
    specs_by_complexity = get_specs_for_category(category)
    return [specs[0] for specs in specs_by_complexity.values()]
