# smart-data-analyst-ai
AI-powered data analytics platform that automates data cleaning, EDA, and visualization, with Claude AI-driven insights, a natural-language chatbot, and ML-based forecasting.


# 📊 AI Data Analytics Agent

An end-to-end AI-powered data analytics platform built with Python, Streamlit, and Claude AI. Upload any CSV or Excel file to automatically clean, analyze, visualize, and generate business insights from your data.

## 🚀 Features

| Module | Description |
|--------|-------------|
| 📁 Dataset Upload | CSV & Excel support with instant preview |
| 🔍 Data Validation | Missing values, duplicates, type detection |
| 🧹 Auto Cleaning | Fill nulls, remove duplicates, parse dates |
| 📊 EDA | Statistical summaries, KPIs, business metrics |
| 📈 Visualizations | Bar, Line, Pie, Scatter, Box, Heatmap charts |
| 🤖 AI Insights | Claude-powered business insight generation |
| 💬 Chat Interface | Natural language Q&A on your dataset |
| 🔮 Forecasting | Linear Regression & Random Forest predictions |
| 📄 Report Export | Excel & CSV download |

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly
- **Machine Learning:** Scikit-learn
- **AI Integration:** Anthropic Claude API
- **Deployment:** Streamlit Community Cloud

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/AI_Data_Analytics_Agent.git
cd AI_Data_Analytics_Agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

## 📁 Project Structure

```
AI_Data_Analytics_Agent/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── README.md
├── data/
│   └── sample_sales.csv    # Sample dataset for testing
└── modules/
    ├── upload.py           # File upload & preview
    ├── cleaning.py         # Data validation & cleaning
    ├── analysis.py         # EDA & KPI calculations
    ├── visualization.py    # Interactive charts
    ├── insights.py         # AI insight generation
    ├── chatbot.py          # Natural language Q&A
    ├── forecasting.py      # ML-based forecasting
    └── report.py           # Excel/CSV export
```

## 📸 Screenshots

Upload any dataset → Auto-analysis → AI-powered insights → Forecasting → Export

## 🎯 Use Cases

- Business analysts needing quick data exploration
- Sales teams analyzing performance data
- Managers wanting automated reports
- Students learning end-to-end data analytics

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

## 📄 License

MIT License
