📰 Fake News Detection System

This project leverages a fine-tuned BERT model to detect whether an article is fake or real. Trained on a popular fake news detection dataset, the system achieves an impressive accuracy of 97%. With a user-friendly interface, you can simply paste an article, and the model will determine its authenticity, providing insights with a pie chart that shows the confidence of the classification. The system also utilizes the AWS Language API to filter and analyze only English news articles, ensuring accuracy in the sentiment analysis.

📜 Features
* BERT-Based Fake News Detection: Uses a fine-tuned BERT model trained specifically to detect fake news with high precision.
* User-Friendly Input: Easily paste any article into the system to check its authenticity.
* Real-Time Classification: The system instantly provides results indicating whether the article is fake or real.
* Confidence Pie Chart: Visualizes how confident the model is in its decision, displaying a pie chart that shows the percentage of fake vs. real.

🚀 How It Works
* Input an article (or a portion of text).
* Submit the text for classification.
* The BERT model analyzes the content and predicts whether it's fake or real.
* The result is presented, along with a pie chart showing the confidence level for both "fake" and "real" labels.

🛠️ Technologies Used
* Python
* Django
* Transformers
* PyTorch
* Pandas
* Numpy
* AWS Language API
* HTML/CSS
* Matplotlib
* JavaScript
* Matplotlib

🖼️ Screenshots

 🧠 Fake News Detection Results
  ![image](https://github.com/user-attachments/assets/775c0680-2b7f-44fa-a2f3-f075aa6e4634)
 📊 Confidence Pie Chart
  ![image](https://github.com/user-attachments/assets/7d8f84d3-ac35-41d1-a6df-003e49019c9c)

🌟 Future Enhancements
* Expand the model to detect news bias and misinformation.
* Include multilingual support for detecting fake news in multiple languages.




