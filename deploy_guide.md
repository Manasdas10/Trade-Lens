# TradeLens Platform Deployment Guide

This guide describes how to deploy the TradeLens interactive dashboard and forecasting REST API server to public live servers.

---

## Method 1: Streamlit Community Cloud (Recommended & Easiest)

Streamlit Community Cloud is **free**, fully hosted, and connects directly to your GitHub repository, redeploying automatically whenever you push updates.

### Prerequisites
1. Push your updated code to your GitHub repository: `https://github.com/Manasdas10/Trade-Lens.git`
2. Create a free account at [share.streamlit.io](https://share.streamlit.io) using your GitHub login.

### Steps to Deploy
1. Log in to [Streamlit Community Cloud](https://share.streamlit.io).
2. Click the **"New app"** button.
3. In the deployment page, enter your repository details:
   - **Repository**: `Manasdas10/Trade-Lens`
   - **Branch**: `main`
   - **Main file path**: `market-engine/ai-python/dashboard.py`
4. Click **"Deploy!"**.
5. Your application will be live in ~2 minutes with a public URL you can share!

---

## Method 2: Render (Docker & REST API Server)

If you want both the **FastAPI server (port 8000)** and the **Streamlit Dashboard (port 8501)** running together in a unified cloud environment, use [Render](https://render.com).

We have configured a `Dockerfile` and a `render.yaml` Blueprint to make this a one-click deployment.

### Steps to Deploy
1. Sign up/log in to [Render](https://render.com).
2. Go to the dashboard and click **"New"** -> **"Blueprint"** (or select **"Web Service"**).
3. Connect your GitHub repository (`Manasdas10/Trade-Lens`).
4. Render will automatically detect the `render.yaml` file:
   - It will launch a Web Service named `trade-lens-platform` using the `Dockerfile`.
   - It builds the environment and runs `/app/start.sh` to fire up both uvicorn (FastAPI) and Streamlit.
5. Click **"Apply"** to approve the blueprint.
6. Once the build completes, Render will provide a public URL (e.g. `https://trade-lens-platform.onrender.com`) pointing to your Streamlit dashboard!

---

## Method 3: Local Docker Execution

If you want to run the containerized application on a server or locally using Docker:

### Build the Image
Navigate to the root directory containing the `Dockerfile` and run:
```bash
docker build -t tradelens-app .
```

### Run the Container
Expose both ports so you can access the REST API and the Dashboard:
```bash
docker run -p 8000:8000 -p 8501:8501 tradelens-app
```

Once running:
- **FastAPI Endpoint**: `http://localhost:8000/docs`
- **Streamlit Dashboard**: `http://localhost:8501`
