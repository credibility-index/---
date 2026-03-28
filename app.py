import gradio as gr
from nir_model import nir_predict, model, tokenizer, device

def gradio_predict(text):
    return nir_predict(text)

demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(label="SMS для проверки", placeholder="BTC x10! Депозит 1000$..."),
    outputs=gr.Label(label="Результат NIR_fishing v3.2 (F1=0.849)"),
    title="🚨 NIR_fishing Detector v3.2",
    description="F1_macro=0.849 | RuBERT-tiny + BiLSTM | НИР МИСиС 2026"
)

if __name__ == "__main__":
    demo.launch(share=True)  
