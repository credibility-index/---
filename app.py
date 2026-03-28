import gradio as gr
from nir_model import nir_phishing_predict, model, tokenizer, device, load_model
import os

# Загружаем модель при старте
if os.path.exists('NIR_phishing_v3.2_best_F1_0.849.pth'):
    model = load_model('NIR_phishing_v3.2_best_F1_0.849.pth')

def gradio_predict(text):
    if not text.strip():
        return "📝 Введите SMS для анализа!"
    
    result = nir_phishing_predict(text)
    prediction = max(result, key=result.get)
    confidence = max(result.values())
    
    return f"""🎯 **РЕЗУЛЬТАТ NIR_PHISHING v3.2 (F1=0.849)**

📱 **SMS**: `{text[:60]}{'...' if len(text) > 60 else ''}`

🚨 **Предсказание**: {prediction}  
💯 **Уверенность**: {confidence:.1%}

📊 **Вероятности**:  
{chr(10).join([f'{k}: {v}' for k, v in result.items()])}"""

demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(
        label="📱 SMS для проверки", 
        placeholder="BTC x10! Депозит 1000$ → 10k! t.me/wallet",
        lines=2
    ),
    outputs=gr.Markdown(),
    title="🚨 NIR_PHISHING Detector v3.2",
    description="""**НИР МИСиС 2026 | Ana Bee**

F1_macro=**0.849** | RuBERT-tiny + BiLSTM | Детектор фишинга/фейков/реальных SMS

**Точность по классам:**
- PHISHING: **99.5%**
- REAL/FAKE: **~84%**""",
    examples=[
        ["BTC x10! Депозит 1000$ → 10k! t.me/wallet"],
        ["Сбербанк: подтвердите данные для вывода 5000₽"],
        ["нато и сербия достигли соглашения по урану"],
        ["Привет! Как дела?"]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
