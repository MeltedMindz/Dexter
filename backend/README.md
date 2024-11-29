# Backend Directory

![1](https://github.com/user-attachments/assets/9feeee0d-03e1-4741-8bb3-9e2a3e522ed4)


This directory contains the backend logic, database schemas, and AI agent integration for the **DexBrain** system. The backend handles data storage, processing, and interaction with the knowledge database, as well as AI-driven decision-making.

---

## **Structure**

### **1. `db/`**
This folder contains scripts and files related to database initialization and schema design.
- **`schema.sql`**: Defines the structure of the PostgreSQL database, including tables for:
  - Strategies
  - Performance metrics
  - Machine learning datasets
- **`initialize.py`**: A Python script to initialize the database by creating the required tables.

### **2. `dexbrain/`**
This folder contains the core AI agent logic and database management tools.
- **`db_manager.py`**: Handles queries and interactions with the PostgreSQL database.
- **`ai_agent.py`**: Contains the Dexter AI agent logic, which queries the database and suggests liquidity strategies.
- **`datasets/`**
  - **`add_dataset.py`**: A script to add new datasets to the database for machine learning purposes.

### **3. `tests/`**
This folder includes test scripts for ensuring the functionality of the backend components.
- **`test_db_manager.py`**: Tests the database manager methods, including querying and inserting strategies.

---

## **Features**

1. **DexBrain Database**:
   - A centralized knowledge base for storing strategies, performance metrics, and datasets.
   - Built on PostgreSQL for robust and scalable data management.

2. **Dexter AI Agent**:
   - Suggests liquidity strategies based on historical data or fresh market analysis.
   - Interacts with the knowledge database to retrieve and store performance insights.

3. **Machine Learning Dataset Management**:
   - Enables adding datasets for training and improving AI models.

4. **Test Coverage**:
   - Includes unit tests to verify database interactions and agent logic.

---

## **Setup**

### **1. Install Dependencies**
\`\`\`bash
pip install -r ../requirements.txt
\`\`\`

### **2. Initialize the Database**
Run the database initialization script:
\`\`\`bash
python db/initialize.py
\`\`\`

### **3. Add Datasets**
Add a new dataset to the database using:
\`\`\`bash
python dexbrain/datasets/add_dataset.py
\`\`\`

### **4. Run Tests**
Ensure the backend is functioning correctly:
\`\`\`bash
pytest tests/
\`\`\`

---

## **Usage**

### **AI Agent**
The Dexter AI agent can:
- Query historical strategies for token pairs.
- Suggest new liquidity strategies based on market data and user-defined risk tolerance.
- Store new strategies and performance metrics into the DexBrain database.

Example usage:
\`\`\`python
from dexbrain.ai_agent import DexterAgent

agent = DexterAgent()
suggestion = agent.suggest_strategy("SOL/USDC", "balanced")
print(suggestion)
\`\`\`

### **Add a Dataset**
Use the dataset script to add a machine learning dataset to the DexBrain database:
\`\`\`bash
python dexbrain/datasets/add_dataset.py
\`\`\`

---

## **Contributing**

Contributions are welcome! If you'd like to add new features or improve the codebase:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## **Future Enhancements**

- **Auto-compounding Logic**: Automate liquidity adjustments and compounding.
- **Advanced Machine Learning**: Integrate TensorFlow/PyTorch models for real-time decision-making.
- **API Integration**: Expose endpoints for external applications to interact with the backend.

---

## **License**

This project is licensed under the [MIT License](../LICENSE).
