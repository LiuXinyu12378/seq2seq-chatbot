"""
完成seq2seq模型
"""
import torch.nn as nn
from chatbot.encoder import Encoder
from chatbot.decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input,input_len,target):
        encoder_outputs,encoder_hidden = self.encoder(input,input_len)
        decoder_outputs,decoder_hidden = self.decoder(encoder_hidden,target,encoder_outputs)
        return decoder_outputs

    def evaluate(self,input,input_len):
        encoder_outputs, encoder_hidden = self.encoder(input, input_len)
        decoder_outputs, predict_result = self.decoder.evaluate(encoder_hidden,encoder_outputs)
        return decoder_outputs,predict_result


    def evaluate_with_beam_search(self,input,input_len):
        encoder_outputs, encoder_hidden = self.encoder(input, input_len)
        best_seq = self.decoder.evaluate_with_beam_search(encoder_hidden, encoder_outputs)
        return best_seq
