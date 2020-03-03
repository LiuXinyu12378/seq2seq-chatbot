"""
进行模型的评估
"""

import torch
import torch.nn.functional as F
from chatbot.dataset import get_dataloader
from tqdm import tqdm
import config
import numpy as np
import pickle
from chatbot.seq2seq import Seq2Seq

def eval():
    model = Seq2Seq().to(config.device)
    model.eval()
    model.load_state_dict(torch.load("./models/model.pkl"))

    loss_list = []
    data_loader = get_dataloader(train=False)
    bar = tqdm(data_loader,total=len(data_loader),desc="当前进行评估")
    with torch.no_grad():
        for idx,(input,target,input_len,target_len) in enumerate(bar):
            input = input.to(config.device)
            target = target.to(config.device)
            input_len = input_len.to(config.device)

            decoder_outputs,predict_result = model.evaluate(input,input_len) #[batch_Size,max_len,vocab_size]
            loss = F.nll_loss(decoder_outputs.view(-1,len(config.target_ws)),target.view(-1),ignore_index=config.input_ws.PAD)
            loss_list.append(loss.item())
            bar.set_description("idx:{} loss:{:.6f}".format(idx,np.mean(loss_list)))
    print("当前的平均损失为：",np.mean(loss_list))


def interface():
    from chatbot.cut_sentence import cut
    import config
    #加载模型
    model = Seq2Seq().to(config.device)
    model.eval()
    model.load_state_dict(torch.load("./models/model.pkl"))

    #准备待预测的数据
    while True:
        origin_input =input("me>>:")
        # if "你是谁" in origin_input or "你叫什么" in origin_input:
        #     result = "我是小智。"
        # elif "你好" in origin_input or "hello" in origin_input:
        #     result = "Hello"
        # else:
        _input = cut(origin_input, by_word=True)
        input_len = torch.LongTensor([len(_input)]).to(config.device)
        _input = torch.LongTensor([config.input_ws.transform(_input,max_len=config.chatbot_input_max_len)]).to(config.device)

        outputs,predict = model.evaluate(_input,input_len)
        result = config.target_ws.inverse_transform(predict[0])
        print("chatbot>>:",result)


def interface_with_beamsearch():
    from chatbot.cut_sentence import cut
    import config
    # 加载模型
    model = Seq2Seq().to(config.device)
    model.eval()
    model.load_state_dict(torch.load("./models/model.pkl"))

    # 准备待预测的数据
    while True:
        origin_input = input("me>>:")
        _input = cut(origin_input, by_word=True)
        input_len = torch.LongTensor([len(_input)]).to(config.device)
        _input = torch.LongTensor([config.input_ws.transform(_input, max_len=config.chatbot_input_max_len)]).to(
            config.device)

        best_seq = model.evaluate_with_beam_search(_input, input_len)
        result = config.target_ws.inverse_transform(best_seq)
        print("chatbot>>:", result)




if __name__ == '__main__':
    # interface()
    interface_with_beamsearch()
