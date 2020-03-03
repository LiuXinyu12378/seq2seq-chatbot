"""
实现attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class Attention(nn.Module):
    def __init__(self,method="general"):
        super(Attention,self).__init__()
        assert method in ["dot","general","concat"],"attention method error"
        self.method = method
        if method == "general":
            self.W = nn.Linear(config.chatbot_encoder_hidden_size*2,config.chatbot_encoder_hidden_size*2,bias=False)

        if method == "concat":
            self.W = nn.Linear(config.chatbot_decoder_hidden_size*4,config.chatbot_decoder_hidden_size*2,bias=False)
            self.V = nn.Linear(config.chatbot_decoder_hidden_size*2,1,bias=False)



    def forward(self,decoder_hidden,encoder_outputs):
        if self.method == "dot":
            return self.dot_score(decoder_hidden,encoder_outputs)

        elif self.method == "general":
            return self.general_socre(decoder_hidden,encoder_outputs)

        elif self.method == "concat":
            return self.concat_socre(decoder_hidden,encoder_outputs)

    def dot_score(self,decoder_hidden,encoder_outputs):
        """H_t^T * H_s
        :param decoder_hidden:[1,batch_size,128*2] --->[batch_size,128*2,1]
        :param encoder_outputs:[batch_size,encoder_max_len,128*2] --->[batch_size,encoder_max_len,128*2]
        :return:attention_weight:[batch_size,encoder_max_len]
        """
        decoder_hidden_viewed = decoder_hidden.squeeze(0).unsqueeze(-1) #[batch_size,128*2,1]
        attention_weight = torch.bmm(encoder_outputs,decoder_hidden_viewed).squeeze(-1)
        return F.softmax(attention_weight,dim=-1)

    def general_socre(self,decoder_hidden,encoder_outputs):
        """
        H_t^T *W* H_s
        :param decoder_hidden:[1,batch_size,128*2]-->[batch_size,decode_hidden_size] *[decoder_hidden_size,encoder_hidden_size]--->[batch_size,encoder_hidden_size]
        :param encoder_outputs:[batch_size,encoder_max_len,128*2]
        :return:[batch_size,encoder_max_len]
        """
        decoder_hidden_processed =self.W(decoder_hidden.squeeze(0)).unsqueeze(-1) #[batch_size,encoder_hidden_size*2,1]
        attention_weight = torch.bmm(encoder_outputs, decoder_hidden_processed).squeeze(-1)
        return F.softmax(attention_weight, dim=-1)

    def concat_socre(self,decoder_hidden,encoder_outputs):
        """
        V*tanh(W[H_t,H_s])
        :param decoder_hidden:[1,batch_size,128*2]
        :param encoder_outputs:[batch_size,encoder_max_len,128*2]
        :return:[batch_size,encoder_max_len]
        """
        #1. decoder_hidden:[batch_size,128*2] ----> [batch_size,encoder_max_len,128*2]
        # encoder_max_len 个[batch_size,128*2] -->[encoder_max_len,bathc_size,128*2] -->transpose--->[]
        encoder_max_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        decoder_hidden_repeated = decoder_hidden.squeeze(0).repeat(encoder_max_len,1,1).transpose(0,1) #[batch_size,max_len,128*2]
        h_cated = torch.cat([decoder_hidden_repeated,encoder_outputs],dim=-1).view([batch_size*encoder_max_len,-1]) #[batch_size*max_len,128*4]
        attention_weight = self.V(F.tanh(self.W(h_cated))).view([batch_size,encoder_max_len]) #[batch_size*max_len,1]
        return F.softmax(attention_weight,dim=-1)







