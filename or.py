# OR GGATE

from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import pandas as  pd

def main(data, model_name, plotName, eta, epochs):
    df_OR = pd.DataFrame(data)
    X, y = prepare_data(df_OR)

    model_OR = Perceptron(eta=eta, epochs=epochs)
    model_OR.fit(X, y)

    _ = model_OR.total_loss()    # Dummy Variable

    model_OR.save(filename=model_name, model_dir='model')
    save_plot(df_OR, model_OR, filename=plotName)

if __name__ == "__main__":
    OR = {
    'x1': [0,0,1,1],
    'x2': [0,1,0,1],
    'y' : [0,1,1,1]
    }
    ETA = 0.3      # Learning Rate
    Epochs = 10
    main(data=OR, model_name='or.model', plotName='or.png', eta=ETA, epochs=Epochs)






