# Contributing to Socio Eco Stress Inference

We welcome contributions from the community, especially from students, researchers, and developers interested in AI for Social Good. This project aims to bridge the gap between theoretical math and real world software engineering.

## 🚀 Roadmap and Future Improvements

We have identified 5 key areas where this project needs to evolve. We prioritize Pull Requests (PRs) that address these specific technical goals.

### 1. Architecture Shift: Encoders over Decoders
* **Goal:** Move classification tasks away from generative models (GPT style) to specialized **Encoder models**.
* **Action:** Implement BERT or RoBERTa based classifiers. These are computationally cheaper and statistically superior for classification and regression tasks compared to generating text.

### 2. Decoupled Embedding Spaces
* **Goal:** Use domain specific embeddings rather than a generic one.
* **Action:** Implement two distinct vector spaces:
    * **Financial:** Use `ProsusAI/finbert` embeddings to capture market specific nuances.
    * **Socio Political:** Use models trained on political discourse (like `politics-distilbert`) to better separate "protest" from "celebration."

### 3. Bias Mitigation via Data Sources
* **Goal:** Reduce reliance on corporate or state sanitized news feeds.
* **Action:** Shift from traditional News APIs to **Social Media Listening** (X/Twitter, Facebook, Reddit). This captures raw, unfiltered public sentiment and detects unrest before it hits the mainstream news cycle.

### 4. Sampling Strategy Overhaul
* **Goal:** Improve how headlines are selected for the model.
* **Action:** Replace random batching with **Weighted Importance Sampling**. Prioritize headlines based on engagement metrics (retweets, shares) rather than just recency to filter out noise.

### 5. Moroccan Localization (The "Darija" Layer)
* **Goal:** Adapt the tool for the MENA region and Morocco specifically.
* **Action:**
    * Fine tune models on **Moroccan Darija, Arabic, and French**.
    * Ingest data from local sources (e.g., Hespress, local Facebook groups).
    * Align the "Social Stress" metric with the reality of the Moroccan street.

## 🛠 How to Contribute

1.  **Fork** the repository.
2.  **Clone** your fork locally.
3.  **Create a Branch** for your feature (`git checkout -b feature/amazing_upgrade`).
4.  **Commit** your changes.
5.  **Push** to the branch.
6.  **Open a Pull Request**.

## 📝 Code Style & Guidelines

* **Educational Focus:** Please comment your code heavily. This repository is used for teaching, so clarity is as important as performance.
* **Dependencies:** Update `reqs.txt` if you add new libraries.
* **Testing:** Verify that the `app.py` dashboard still launches correctly after your changes.

Thank you for helping us build a robust tool for socio economic analysis!