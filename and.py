# AND GATE

from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import pandas as  pd
import logging
import os

gate = 'AND Gate'

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'running_logs.logs'),
    level=logging.INFO,
    format='[%(asctime)s: %(levelname)s: %(module)s]: %(message)s',
    filemode='a')

def main(data, model_name, plotName, eta, epochs):
    df = pd.DataFrame(data)
    logging.info(f"This is the raw dataset: \n {df}")
    X, y = prepare_data(df)

    model_AND = Perceptron(eta=eta, epochs=epochs)
    model_AND.fit(X, y)

    _ = model_AND.total_loss()    # Dummy Variable

    model_AND.save(filename=model_name, model_dir='model')
    save_plot(df, model_AND, filename=plotName)

if __name__ == "__main__":
    AND = {
    'x1': [0,0,1,1],
    'x2': [0,1,0,1],
    'y' : [0,0,0,1]
    }
    ETA = 0.3      # Learning Rate
    Epochs = 10

    try:
        logging.info(f'<<<<<<<< Starting Training for {gate} >>>>>>>>')
        main(data=AND, model_name='and.model', plotName='and.png', eta=ETA, epochs=Epochs)
        logging.info(f'<<<<<<<< Training Completed for {gate} >>>>>>>>\n\n')
    except Exception as e:
        logging.exception(e)
        raise e 








