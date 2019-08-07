from utils import vocabulary_extractor
from model.rnnmodel import Model
import yaml
import sys
from utils.arg_parser import parse_input_args
from data_processing.graph_features import  get_used_edges_type, get_used_nodes_type

def train(task_id):

  with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

  checkpoint_path = "saved_models/rnnmodel"
  train_path = cfg['train_path']
  val_path   = cfg['val_path']
  token_path = cfg['token_path']


  vocabulary = vocabulary_extractor.create_vocabulary_from_corpus(train_path, token_path)
  print("Constructed vocabulary...")

  m = Model(mode='train', task_id=task_id, vocabulary=vocabulary)
  n_train_epochs = 50

  m.train(train_path=train_path, val_path=val_path, n_epochs=n_train_epochs, checkpoint_path=checkpoint_path)
  print("Model trained successfully...")



args = sys.argv[1:]
task_id = parse_input_args(args)


train(task_id)
print(get_used_nodes_type())
