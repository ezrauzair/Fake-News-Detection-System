from django.shortcuts import render
from transformers import BertTokenizer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import base64
import io
import boto3
from django.contrib import messages
from django.contrib.messages import get_messages

def home(request):
    return render(request, 'home.html')

def bulletchart(request):
    if request.method == 'POST' and 'process' in request.POST:
        # Clear existing messages
        storage = messages.get_messages(request)
        for _ in storage:
            pass
        access_key = 'Paste Your Own AWS Access Key Here'
        secret_key = 'Paste Your Own AWS Secret Key Here'
        region_name = 'Paste Your Own AWS Region Name Here'
        comprehend = boto3.client(
            'comprehend',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )

        news = request.POST.get('news')

        if not news:
            messages.error(request, 'Please paste an article')
            return render(request, 'home.html')
        else:
            words = news.split()
            length = len(words)

        if length < 50:
            messages.error(request, 'The length is too short')
            return render(request, 'home.html')

        response = comprehend.detect_dominant_language(Text=news)
        detected_language_code = response['Languages'][0]['LanguageCode']
        confidence_score = response['Languages'][0]['Score']

        if detected_language_code != 'en' or confidence_score < 0.99:
            messages.error(request, 'An English article only')
            return render(request, 'home.html')
        

        device = torch.device('cpu')
        model = torch.load(r"C:\Users\PMYLS\Desktop\All projects\FNDS without Menue\New folder (2)\fakenews\modelandtok\fakenewsdetectionmodel.pt", map_location=device)
        tokenizer = BertTokenizer.from_pretrained(r"C:\Users\PMYLS\Desktop\All projects\FNDS without Menue\New folder (2)\fakenews\modelandtok\tok")

        model.eval().to('cpu')

        encoded_article = tokenizer.encode_plus(
            news,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=500,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )

        input_ids_real = encoded_article['input_ids']
        attention_masks_real = encoded_article['attention_mask']

        with torch.no_grad():
            input_ids_real = input_ids_real.to(device)
            attention_masks_real = attention_masks_real.to(device)

            outputs = model(input_ids_real, attention_mask=attention_masks_real)
            _, predicted_labels = torch.max(outputs.logits, 1)
            predictions = predicted_labels.cpu().numpy()
            probabilities = F.softmax(outputs.logits, dim=1).cpu().numpy()

        real_prob = probabilities[0][0]
        fake_prob = probabilities[0][1]

        labels = 'Real', 'Fake'
        sizes = [real_prob, fake_prob]
        colors = ['#0EB48B', 'lightcoral']
        explode = (0.1, 0)
        fig = plt.figure(figsize=(5, 4), dpi=100)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.title('Real vs Fake Probability')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        image_base64 = base64.b64encode(image_png).decode('utf-8')
        chart = f'data:image/png;base64,{image_base64}'

        label = "Fake" if predictions[0] == 1 else "Real"
        
        
        if label == 'Real':
                return render(request, 'home.html', {'Real': label, 'realp': real_prob, 'fakep': fake_prob, 'chart': chart, 'ok':'ok'})
        elif label == "Fake":
                return render(request, 'home.html', {'Fake': label, 'realp': real_prob, 'fakep': fake_prob, 'chart': chart, 'ok':'ok'})
        else:
            return render(request, 'home.html')
