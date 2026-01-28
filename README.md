# Capital_pulse_submission

**Capital Pulse** is a financial analytics platform that combines **Machine Learning (XGBoost)** for quantitative forecasting with **Generative AI (RAG + LLM)** for qualitative market analysis. It allows users to upload stock data, visualize future trends with confidence intervals, and chat with an AI analyst about market conditions.

## 🚀 Features

### 1. 📊 Advanced Forecasting
* **Algorithm:** Uses **XGBoost Regressor** to predict future stock returns based on historical lags and volatility features.
* **Recursive Strategy:** Employs a recursive forecasting method to generate realistic multi-day price paths (preventing "flatline" predictions).
* **Confidence Intervals:** Calculates and visualizes 95% confidence regions based on the model's historical error variance (sigma) and time horizon.
* **Dual Visualization:**
    * **Historical Context:** View the forecast in the context of the last 90 days.
    * **Zoomed Forecast:** A dedicated 7-day view with precise price labels and uncertainty bands.

### 2. 🤖 AI Market Analyst (RAG Chatbot)
* **Retrieval Augmented Generation (RAG):** Contextualizes the AI's answers using your specific uploaded CSV data.
* **Vector Search:** Uses `SKLearnVectorStore` (Scikit-Learn) and `FAISS` principles to find relevant market days (trends, volume, price action) similar to user queries.
* **LLM Integration:** Powered by **Google Flan-T5-Small** (via Hugging Face) for lightweight, offline-capable text generation.
* **Memory:** Maintains conversational history for a continuous chat experience.

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** [XGBoost](https://xgboost.readthedocs.io/), [Scikit-Learn](https://scikit-learn.org/)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib
* **AI & LLM:**
    * [LangChain](https://www.langchain.com/) (Orchestration)
    * [Hugging Face Transformers](https://huggingface.co/) (Model & Embeddings)
    * `sentence-transformers/all-MiniLM-L6-v2` (Embeddings)
    * `google/flan-t5-small` (Text Generation)

## ⚙️ Installation

### Prerequisites
* Python 3.9 or higher (Tested on Python 3.13)

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/capital-pulse.git](https://github.com/yourusername/capital-pulse.git)
cd capital-pulse
