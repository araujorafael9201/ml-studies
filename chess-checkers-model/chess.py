# Export to huggingface.co 
# Live Demo at https://rafaelaraujo13-chess-checkers.hf.space/

import gradio as gr
from fastai.vision.all import *

def clasify_image(img):
    categories = ['checkers', 'chess']

    pred, idx, probs = learner.predict(img)
    return dict(zip(categories, map(float, probs)))

learner = load_learner('model.pkl')

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['chess.jpg', 'checkers.jpg']

intf = gr.Interface(fn=clasify_image, inputs=image, outputs=label, examples=examples)

intf.launch(inline=False)
