# Auto Feature Selector Tool 🔍

This project is a **Streamlit-based web application** that allows users to perform **automatic feature selection** on tabular data using different statistical techniques. It simplifies data preprocessing, feature engineering, and allows exporting the refined dataset easily.

---

## 🚀 Features

- 📁 Upload CSV dataset
- 🧹 Clean dataset:
  - Remove unnecessary columns
  - Handle missing values (mean, median, mode, custom, iterative imputation)
  - Remove duplicate rows
- 🔠 Encode categorical columns (Label Encoding / One-Hot Encoding)
- 🎯 Select the target column for prediction
- 🔍 Perform feature selection using:
  - Correlation Coefficient
  - Chi-Square Test
  - ANOVA (f_classif)
  - Mutual Information
  - Variance Threshold
- 📥 Download the selected features along with the target column as a CSV file

---

## 🛠 Tech Stack

- Python
- [Streamlit](https://streamlit.io/)
- Pandas, NumPy
- scikit-learn
- fancyimpute

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/auto-feature-selector.git
cd auto-feature-selector
2. Create Virtual Environment
bash
Copy code
python -m venv myenv
3. Activate Virtual Environment
On Windows:

bash
Copy code
myenv\Scripts\activate
On Mac/Linux:

bash
Copy code
source myenv/bin/activate
4. Install Dependencies
bash
Copy code
pip install -r requirements.txt
▶️ Run the Application
bash
Copy code
streamlit run app.py
📁 Project Structure
bash
Copy code
auto-feature-selector/
├── app.py               # Main Streamlit app
├── helpers.py           # All helper functions for data cleaning and feature selection
├── requirements.txt     # Python dependencies
├── .gitignore           # Files/folders to ignore in Git
└── README.md            # Project documentation
💡 Example Use Case
Upload a dataset, select your target variable (like Price, Survived, Outcome, etc.), choose a feature selection method (e.g., ANOVA), and get a filtered dataset with the most important features. Perfect for ML preprocessing!

📸 Screenshots
Add screenshots of your UI here if available:

mathematica
Copy code
📊 Upload → Clean → Encode → Select Target → Feature Selection → Download!
📄 License
This project is licensed under the MIT License. See LICENSE for more information.

👨‍💻 Author
Prabhujot Deshwal
GitHub: @your-username

🙏 Acknowledgements
Streamlit Docs

scikit-learn Team

OpenAI (for feature inspiration 😉)
