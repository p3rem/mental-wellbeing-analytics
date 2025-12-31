# 🎓 Placement & Interview Resources

This document contains ready-to-use bullet points for your resume and structured answers for technical interviews.

## 📄 Resume Bullet Points

**Project Name:** Mental Wellbeing Analytics (Group Project)

**Option 1: Technical Focus (Best for SDE/Data Science Roles)**
*   Developed an end-to-end analytics dashboard to study the correlation between digital habits (Screen Time, Social Media) and mental health indicators using **Python** and **Streamlit**.
*   Implemented **Random Forest** (90% accuracy) for stress prediction and **XGBoost** for mood classification, integrating models into a real-time web application.
*   Applied **K-Means Clustering** to segment user lifestyle patterns and **Isolation Forest** for detecting anomalous digital behaviors.
*   Collaborated with a team to build a scalable ML pipeline using **Scikit-learn** and **Pandas**, conducting comprehensive EDA on a dataset of 100k+ records.

**Option 2: Product/Impact Focus (Best for Analyst/Consultant Roles)**
*   Led the development of "Mental Wellbeing Analytics," a data-driven tool to quantify the impact of digital consumption on user mental health.
*   Designed an interactive dashboard using **Streamlit** and **Plotly**, enabling users to visualize lifestyle trends and providing personalized, ML-driven wellness recommendations.
*   Analyzed complex datasets to identify key stressors, achieving 85%+ predictive accuracy in flagging high-stress digital patterns.

---

## 🎤 Interview Prep (STAR Method)

### Q: "Tell me about this project."
**Situation:** "As part of our placement training, my team identified a need to understand how increasing digital consumption affects mental health."
**Task:** "We aimed to build a full-stack data application that not only analyzes this correlation but also uses Machine Learning to provide actionable insights for users."
**Action:** "I worked on [Your Main Contribution, e.g., the ML pipeline/frontend]. We used Random Forest for classification because of its robustness against overfitting on our dataset. We also implemented K-Means clustering to categorize varied user lifestyles rather than just labeling them 'good' or 'bad'. We wrapped this in a Streamlit app to make it accessible."
**Result:** "The final application successfully identifies high-stress patterns with 90% accuracy and provides a user-friendly interface for self-assessment, which we successfully demonstrated during our college evaluation."

### Q: "Why did you choose Random Forest over other models?"
"We started with Logistic Regression, but it struggled with the non-linear relationships between screen time and mood. Random Forest provided better accuracy (~90%) and feature importance visibility, which was crucial for explaining *why* a user was predicted as 'High Stress' (e.g., seeing that Sleep Quality was the top factor)."

### Q: "What was the most challenging part?"
"Cleaning and feature engineering the data was tricky. Raw screen time data isn't enough, so we created 'derived features' like `Digital Wellness Score` and `Screen-to-Sleep Ratio`. These engineered features significantly improved our model's F1-score compared to using just raw inputs."
