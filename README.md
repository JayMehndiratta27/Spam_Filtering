📩 Spam Message Classifier
This project builds a spam message filtering system using Machine Learning and NLP. It trains models to classify text messages as Spam or Ham (Not Spam) with high accuracy.

🚀 Features
✅ Text Preprocessing & Vectorization using TfidfVectorizer
✅ Machine Learning Models:

Multinomial Naïve Bayes (MNB) – 96.50% accuracy
Complement Naïve Bayes (CNB) – 98.30% accuracy
Linear SVC – 99.19% accuracy (Best Model ✅)
✅ Real-Time Prediction – Classifies new messages instantly
📂 Dataset
Uses the Spam Text Message 2017 Dataset from Kaggle, which contains:

📩 5572 messages (4825 ham, 747 spam)
📊 Cleaned and split into train (80%) and test (20%)
⚙️ How It Works
1️⃣ Loads the dataset and splits it into training & testing sets.
2️⃣ Converts text to numerical features using TfidfVectorizer.
3️⃣ Trains multiple models (Naïve Bayes & SVC).
4️⃣ Evaluates performance using accuracy & classification reports.
5️⃣ Predicts new messages (e.g., "Call 927363663 to receive your prize" → Spam ✅).



