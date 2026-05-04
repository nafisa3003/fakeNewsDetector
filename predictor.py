import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from preprocessor import wordopt


def output_lable(n):
    # converts numeric prediction to text label
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not Fake News"


def manual_testing(news, vectorization, LR, DT, GB, RF):
    # predicts news class using trained models and majority vote
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) # cleans the input news text
    new_x_test  = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    pred_LR = int(LR.predict(new_xv_test)[0])
    pred_DT = int(DT.predict(new_xv_test)[0])
    pred_GB = int(GB.predict(new_xv_test)[0])
    pred_RF = int(RF.predict(new_xv_test)[0])

    print("LR Prediction: {}".format(output_lable(pred_LR)))
    print("DT Prediction: {}".format(output_lable(pred_DT)))
    print("GB Prediction: {}".format(output_lable(pred_GB)))
    print("RF Prediction: {}".format(output_lable(pred_RF)))

    # calculates final verdict based on the consensus of four models
    votes = pred_LR + pred_DT + pred_GB + pred_RF
    majority = 1 if votes >= 2 else 0
    print("\nFinal Verdict: {}".format(output_lable(majority)))


def launch_widget(vectorization, LR, DT, GB, RF):
    # initializes and displays interactive Jupyter widget for testing
    text_input = widgets.Textarea(
        placeholder='Paste your news article here...',
        description='News:',
        layout=widgets.Layout(width='80%', height='100px')
    )

    button = widgets.Button(description='Check News', button_style='primary')
    output = widgets.Output()

    def on_button_click(b):
        with output:
            output.clear_output()
            news = text_input.value
            if news.strip() == '':
                print("Please enter some news text!")
            else:
                manual_testing(news, vectorization, LR, DT, GB, RF)

    button.on_click(on_button_click)
    display(text_input, button, output)