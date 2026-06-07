import gradio as gr
import pickle

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

labels = {0: "😡 Negative", 1: "😐 Neutral", 2: "🙂 Positive"}

def predict(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    return labels[pred[0]]

with gr.Blocks() as demo:
    gr.Markdown("# 💬 Sentiment Analysis AI")
    gr.Markdown("Enter text and detect sentiment (Positive / Neutral / Negative)")

    input_box = gr.Textbox(label="Input Text", lines=3)
    output_box = gr.Textbox(label="Result")

    btn = gr.Button("Analyze")

    btn.click(predict, input_box, output_box)

    gr.Examples(
        examples=[
            "I love this product, it is amazing",
            "This is the worst experience ever",
            "It is okay not bad"
        ],
        inputs=input_box
    )

demo.launch()