"""
项目配置
"""
import pickle
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = ("cpu")

################# classify 相关的配置 ###############
predict_ratio = 0.98  #预测可能性的阈值

################# chatbot相关的配置 #################
chatbot_train_batch_size = 400
chatbot_test_batch_size = 500

input_ws = pickle.load(open("../chatbot/models/ws_input.pkl","rb"))
target_ws = pickle.load(open("../chatbot/models/ws_target.pkl","rb"))
chatbot_input_max_len = 20
chatbot_target_max_len = 30

chatbot_encoder_embedding_dim = 300
chatbot_encoder_hidden_size = 128
chatbot_encoder_number_layer = 2
chatbot_encoder_bidirectional = True
chatbot_encoder_dropout = 0.3

chatbot_decoder_embedding_dim = 300
chatbot_decoder_hidden_size = 128*2
chatbot_decoder_number_layer = 1
chatbot_decoder_dropout = 0

beam_width = 15