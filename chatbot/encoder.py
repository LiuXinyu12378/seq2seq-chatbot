"""
进行编码
"""

import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
import config
import torch


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.input_ws),
                                     embedding_dim=config.chatbot_encoder_embedding_dim,
                                     padding_idx=config.input_ws.PAD
                                     )
        # 2层双向，每层hidden_size 128
        self.gru = nn.GRU(input_size=config.chatbot_encoder_embedding_dim,
                          hidden_size=config.chatbot_encoder_hidden_size,
                          num_layers=config.chatbot_encoder_number_layer,
                          batch_first=True,
                          bidirectional=config.chatbot_encoder_bidirectional,
                          dropout=config.chatbot_encoder_dropout)


    def forward(self, input,input_len):
        input_embeded = self.embedding(input)

        #对输入进行打包
        input_packed = pack_padded_sequence(input_embeded,input_len,batch_first=True)
        #经过GRU处理
        output,hidden = self.gru(input_packed)
        # print("encoder gru hidden:",hidden.size())
        #进行解包
        output_paded,seq_len = pad_packed_sequence(output,batch_first=True,padding_value=config.input_ws.PAD)
        #获取最上层的正向和反向最后一个时间步的输出，表示整个句子
        encoder_hidden = torch.cat([hidden[-2],hidden[-1]],dim=-1).unsqueeze(0) #[1,batch_size,128*2]
        return output_paded,encoder_hidden  #[1,batch_size,128*2]
