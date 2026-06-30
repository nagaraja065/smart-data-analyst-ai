"""
Application Constants — Centralized magic values, labels, and UI tokens.

Everything that would otherwise be a magic string or number lives here.
Organized by domain for easy discovery.
"""

# ─── Application ─────────────────────────────────────────────────────────────

APP_TITLE = "📊 AI Data Analytics Agent"
APP_ICON = "📊"
APP_DESCRIPTION = "Enterprise AI-powered data analytics platform"

# ─── Supported File Types ────────────────────────────────────────────────────

SUPPORTED_FILE_TYPES = {
    "csv": {"extensions": [".csv", ".tsv"], "mime": "text/csv"},
    "excel": {"extensions": [".xlsx", ".xls"], "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
    "json": {"extensions": [".json"], "mime": "application/json"},
}

UPLOAD_EXTENSIONS = ["csv", "tsv", "xlsx", "xls", "json"]

# ─── Data Quality Thresholds ────────────────────────────────────────────────

QUALITY_THRESHOLDS = {
    "excellent": 90,
    "good": 75,
    "fair": 50,
    "poor": 25,
}

MAX_MISSING_PCT_WARNING = 5.0      # Warn if >5% missing
MAX_MISSING_PCT_CRITICAL = 30.0    # Critical if >30% missing
MAX_CARDINALITY_RATIO = 0.95       # Flag if >95% unique (possible ID column)
MIN_ROWS_FOR_ML = 50               # Minimum rows for ML models
MIN_ROWS_FOR_FORECAST = 12         # Minimum data points for time-series

# ─── Column Detection Keywords ──────────────────────────────────────────────

COLUMN_KEYWORDS = {
    "sales": ["sales", "revenue", "amount", "total_sales", "gross_sales"],
    "profit": ["profit", "margin", "income", "net_income", "earnings"],
    "quantity": ["quantity", "qty", "units", "count", "volume"],
    "date": ["date", "time", "timestamp", "order_date", "created_at", "datetime"],
    "category": ["category", "type", "segment", "group", "class", "department"],
    "region": ["region", "state", "location", "city", "area", "country", "territory"],
    "product": ["product", "item", "name", "sku", "product_name"],
    "customer": ["customer", "client", "buyer", "user", "customer_id", "account"],
    "price": ["price", "cost", "unit_price", "rate", "fee"],
    "id": ["id", "order_id", "transaction_id", "invoice"],
}

# ─── Chart Configuration ────────────────────────────────────────────────────

CHART_TYPES = [
    "Bar Chart", "Line Chart", "Pie Chart", "Donut Chart",
    "Scatter Plot", "Box Plot", "Histogram", "Heatmap",
    "Treemap", "Funnel", "Waterfall", "Sunburst",
]

PLOTLY_TEMPLATE = "plotly_white"

# Enterprise color palette — tested for accessibility (WCAG AA)
COLOR_PALETTE = {
    "primary": "#6366F1",     # Indigo
    "secondary": "#8B5CF6",   # Violet
    "success": "#10B981",     # Emerald
    "warning": "#F59E0B",     # Amber
    "danger": "#EF4444",      # Red
    "info": "#3B82F6",        # Blue
    "neutral": "#6B7280",     # Gray
}

CHART_COLORS = [
    "#6366F1", "#8B5CF6", "#EC4899", "#F43F5E",
    "#F97316", "#EAB308", "#22C55E", "#14B8A6",
    "#06B6D4", "#3B82F6", "#A855F7", "#D946EF",
]

# ─── ML Configuration ───────────────────────────────────────────────────────

ML_MODELS = {
    "regression": ["Linear Regression", "Ridge", "Lasso", "Random Forest", "Gradient Boosting"],
    "classification": ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting", "KNN"],
    "clustering": ["KMeans", "DBSCAN", "Hierarchical"],
    "anomaly": ["Isolation Forest", "Local Outlier Factor", "Z-Score", "IQR"],
    "forecasting": ["Linear Trend", "Moving Average", "Random Forest Regressor"],
}

TEST_SIZE_DEFAULT = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# ─── AI Agent ────────────────────────────────────────────────────────────────

AGENT_MAX_STEPS = 5
AGENT_SYSTEM_PROMPT = """You are an expert data analyst AI agent. You analyze datasets, 
generate insights, create visualizations, and answer business questions with precision.
Always cite specific numbers. Be concise but thorough. Use markdown formatting."""

# ─── UI Labels ───────────────────────────────────────────────────────────────

PAGE_LABELS = {
    "upload": "📁 Data Upload",
    "profiling": "🔍 Data Profiling",
    "eda": "📊 Exploratory Analysis",
    "visualizations": "📈 Visualizations",
    "ai_agent": "🤖 AI Agent",
    "ml_studio": "🧠 ML Studio",
    "forecasting": "🔮 Forecasting",
    "reports": "📄 Reports",
    "settings": "⚙️ Settings",
}

# ─── Error Messages ─────────────────────────────────────────────────────────

MSG_NO_DATA = "👈 Please upload a dataset first."
MSG_UPLOAD_SUCCESS = "✅ Dataset loaded successfully!"
MSG_CLEANING_DONE = "✅ Data cleaned and ready for analysis."
MSG_NO_NUMERIC = "⚠️ No numeric columns found for this analysis."
MSG_NO_CATEGORICAL = "⚠️ No categorical columns found."
MSG_NO_DATE = "⚠️ No date/time column detected."
MSG_INSUFFICIENT_DATA = "⚠️ Not enough data points for this analysis."
MSG_API_KEY_MISSING = "💡 Add a Claude API key in Settings for AI-powered analysis."
MSG_API_ERROR = "❌ API error occurred. Falling back to local analysis."
