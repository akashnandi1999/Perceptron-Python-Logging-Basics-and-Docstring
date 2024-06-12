# AND GATE

from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import pandas as  pd

def main(data, model_name, plotName, eta, epochs):
    df_AND = pd.DataFrame(data)
    X, y = prepare_data(df_AND)

    model_AND = Perceptron(eta=eta, epochs=epochs)
    model_AND.fit(X, y)

    _ = model_AND.total_loss()    # Dummy Variable

    model_AND.save(filename=model_name, model_dir='model')
    save_plot(df_AND, model_AND, filename=plotName)

if __name__ == "__main__":
    AND = {
    'x1': [0,0,1,1],
    'x2': [0,1,0,1],
    'y' : [0,0,0,1]
    }
    ETA = 0.3      # Learning Rate
    Epochs = 10
    main(data=AND, model_name='and.model', plotName='and.png', eta=ETA, epochs=Epochs)






