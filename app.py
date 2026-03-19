
    import gradio as gr
    import random
    def predict(text):
        return f"Processed: {text}"
    with gr.Blocks() as demo:
        gr.Markdown("## AI Tool")
        inp = gr.Textbox(label="Input")
        out = gr.Textbox(label="Output")
        inp.change(predict, inp, out)
    demo.launch()
    