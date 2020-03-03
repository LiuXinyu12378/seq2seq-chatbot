"""
实现解码器
"""
import heapq

import torch.nn as nn
import config
import torch
import torch.nn.functional as F
import numpy as np
import random
from chatbot.attention import Attention


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

        self.embedding = nn.Embedding(num_embeddings=len(config.target_ws),
                                      embedding_dim=config.chatbot_decoder_embedding_dim,
                                      padding_idx=config.target_ws.PAD)

        #需要的hidden_state形状：[1,batch_size,64]
        self.gru = nn.GRU(input_size=config.chatbot_decoder_embedding_dim,
                          hidden_size=config.chatbot_decoder_hidden_size,
                          num_layers=config.chatbot_decoder_number_layer,
                          bidirectional=False,
                          batch_first=True,
                          dropout=config.chatbot_decoder_dropout)

        #假如encoder的hidden_size=64，num_layer=1 encoder_hidden :[2,batch_sizee,64]

        self.fc = nn.Linear(config.chatbot_decoder_hidden_size,len(config.target_ws))
        self.attn = Attention(method="general")
        self.fc_attn = nn.Linear(config.chatbot_decoder_hidden_size * 2, config.chatbot_decoder_hidden_size, bias=False)

    def forward(self, encoder_hidden,target,encoder_outputs):
        # print("target size:",target.size())
        #第一个时间步的输入的hidden_state
        decoder_hidden = encoder_hidden  #[1,batch_size,128*2]
        #第一个时间步的输入的input
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor([[config.target_ws.SOS]]*batch_size).to(config.device)         #[batch_size,1]
        # print("decoder_input:",decoder_input.size())


        #使用全为0的数组保存数据，[batch_size,max_len,vocab_size]
        decoder_outputs = torch.zeros([batch_size,config.chatbot_target_max_len,len(config.target_ws)]).to(config.device)

        if random.random()>0.5:    #teacher_forcing机制

            for t in range(config.chatbot_target_max_len):
                decoder_output_t,decoder_hidden = self.forward_step(decoder_input,decoder_hidden,encoder_outputs)
                decoder_outputs[:,t,:] = decoder_output_t


                #获取当前时间步的预测值
                value,index = decoder_output_t.max(dim=-1)
                decoder_input = index.unsqueeze(-1)  #[batch_size,1]
                # print("decoder_input:",decoder_input.size())
        else:
            for t in range(config.chatbot_target_max_len):
                decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden,encoder_outputs)
                decoder_outputs[:, t, :] = decoder_output_t
                #把真实值作为下一步的输入
                decoder_input = target[:,t].unsqueeze(-1)
                # print("decoder_input size:",decoder_input.size())
        return decoder_outputs,decoder_hidden


    def forward_step(self,decoder_input,decoder_hidden,encoder_outputs):
        '''
        计算一个时间步的结果
        :param decoder_input: [batch_size,1]
        :param decoder_hidden: [1,batch_size,128*2]
        :return:
        '''

        decoder_input_embeded = self.embedding(decoder_input)
        # print("decoder_input_embeded:",decoder_input_embeded.size())

        #out:[batch_size,1,128*2]
        #decoder_hidden :[1,bathc_size,128*2]
        # print(decoder_hidden.size())
        out,decoder_hidden = self.gru(decoder_input_embeded,decoder_hidden)

        ##### 开始attention ############
        ### 1. 计算attention weight
        attn_weight = self.attn(decoder_hidden,encoder_outputs)  #[batch_size,1,encoder_max_len]
        ### 2. 计算context vector
        #encoder_ouputs :[batch_size,encoder_max_len,128*2]
        context_vector = torch.bmm(attn_weight.unsqueeze(1),encoder_outputs).squeeze(1) #[batch_szie,128*2]
        ### 3. 计算 attention的结果
        #[batch_size,128*2]  #context_vector:[batch_size,128*2] --> 128*4
        #attention_result = [batch_size,128*4] --->[batch_size,128*2]
        attention_result = torch.tanh(self.fc_attn(torch.cat([context_vector,out.squeeze(1)],dim=-1)))
        # attention_result = torch.tanh(torch.cat([context_vector,out.squeeze(1)],dim=-1))
        #### attenion 结束

        # print("decoder_hidden size:",decoder_hidden.size())
        #out ：【batch_size,1,hidden_size】

        # out_squeezed = out.squeeze(dim=1) #去掉为1的维度
        out_fc = F.log_softmax(self.fc(attention_result),dim=-1) #[bathc_size,vocab_size]
        # print("out_fc:",out_fc.size())
        return out_fc,decoder_hidden

    def evaluate(self,encoder_hidden,encoder_outputs):

        # 第一个时间步的输入的hidden_state
        decoder_hidden = encoder_hidden  # [1,batch_size,128*2]
        # 第一个时间步的输入的input
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor([[config.target_ws.SOS]] * batch_size).to(config.device)  # [batch_size,1]
        # print("decoder_input:",decoder_input.size())

        # 使用全为0的数组保存数据，[batch_size,max_len,vocab_size]
        decoder_outputs = torch.zeros([batch_size, config.chatbot_target_max_len, len(config.target_ws)]).to(
            config.device)

        predict_result = []
        for t in range(config.chatbot_target_max_len):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden,encoder_outputs)
            decoder_outputs[:, t, :] = decoder_output_t

            # 获取当前时间步的预测值
            value, index = decoder_output_t.max(dim=-1)
            predict_result.append(index.cpu().detach().numpy()) #[[batch],[batch]...]
            decoder_input = index.unsqueeze(-1)  # [batch_size,1]
            # print("decoder_input:",decoder_input.size())
            # predict_result.append(decoder_input)
        #把结果转化为ndarray，每一行是一条预测结果
        predict_result = np.array(predict_result).transpose()
        return decoder_outputs, predict_result

    def evaluate_with_beam_search(self, encoder_hidden, encoder_outputs):
        """
        使用beam search完成评估，只能输入一个句子，得到一个输出
        :param encoder_hidden:
        :param encoder_outputs:
        :return:
        """
        # 第一个时间步的输入的hidden_state
        decoder_hidden = encoder_hidden  # [1,batch_size,128*2]
        # 第一个时间步的输入的input
        batch_size = encoder_hidden.size(1)
        assert batch_size == 1, "beam search的过程中，batch_size只能为1"
        decoder_input = torch.LongTensor([[config.target_ws.SOS]] * batch_size).to(config.device)  # [batch_size,1]

        prev_beam = Beam()
        prev_beam.add(1, False, [decoder_input], decoder_input, decoder_hidden)

        while True:
            cur_beam = Beam()
            for prob, complete, seq_list, decoder_input, decoder_hidden in prev_beam:
                if complete:  # 有可能前一次已经到达eos了，但是概率不是最大的
                    cur_beam.add(prob, complete, seq_list, decoder_input, decoder_hidden)
                else:
                    decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)

                    value, index = torch.topk(decoder_output_t, config.beam_width)
                    # print("value index size:",value[0].size(),index[0].size())
                    for m, n in zip(value[0], index[0]):
                        # print("m,n size:",m.size(),n.size(),m,n)
                        cur_prob = prob * m.item()
                        decoder_input = torch.LongTensor([[n.item()]]).to(config.device)
                        cur_seq_list = seq_list + [decoder_input]
                        if n == config.target_ws.EOS:
                            cur_complete = True
                        else:
                            cur_complete = False
                        cur_beam.add(cur_prob, cur_complete, cur_seq_list, decoder_input, decoder_hidden)

            best_prob, best_complete, best_seq, _, _ = max(cur_beam)
            if best_complete or len(best_seq) - 1 == config.chatbot_target_max_len:

                best_seq = [i.item() for i in best_seq]
                if best_seq[0] == config.target_ws.SOS:
                    best_seq = best_seq[1:]
                if best_seq[-1] == config.target_ws.EOS:
                    best_seq = best_seq[:-1]
                return best_seq


            else:
                prev_beam = cur_beam


class Beam:
    """保存每一个时间步的数据"""

    def __init__(self):
        self.heapq = list()
        self.beam_width = config.beam_width

    def add(self, prob, complete, seq_list, decoder_input, decoder_hidden):
        heapq.heappush(self.heapq, [prob, complete, seq_list, decoder_input, decoder_hidden])
        # 保证最终只有一个beam width个结果
        if len(self.heapq) > self.beam_width:
            heapq.heappop(self.heapq)

    def __iter__(self):
        for item in self.heapq:
            yield item


