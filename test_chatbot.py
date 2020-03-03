"""
测试chatbot中所有的api
"""

from chatbot.word_sequece import WordSequence
from chatbot.dataset import get_dataloader
import torch
import pickle
from tqdm import tqdm
# from chatbot.train import train
# from chatbot.eval import eval,interface

def test_ws():
    ws = WordSequence()
    s = [["我","是","谁"],["你","好",'么']]
    for i in s:
        ws.fit(i)
    ws.build_vocab(min_count=1)
    print(ws.dict)
    ret = ws.transform(["你","好"],max_len=4,add_eos=False)
    print(ret)
    ret = ws.inverse_transform(ret)
    print(ret)

def prepar_ws():
    ws = WordSequence()
    for line in tqdm(open("./corpus/chatbot/input.txt",encoding="utf-8").readlines()):
        line = line.strip().split()
        ws.fit(line)
    ws.build_vocab()
    print(len(ws))  #3509
    pickle.dump(ws,open("./chatbot/models/ws_input.pkl","wb"))

    ws = WordSequence()
    for line in tqdm(open("./corpus/chatbot/target.txt",encoding="utf-8").readlines()):
        line = line.strip().split()
        ws.fit(line)
    ws.build_vocab()
    print(len(ws))  #4365
    pickle.dump(ws, open("./chatbot/models/ws_target.pkl","wb"))

def test_dataset():
    from chatbot.dataset import chatbot_data_split,ChatDataset
    # from chatbot.dataset import ChatDataset
    #train 296239 test:74440
    d_train = ChatDataset(train=False)
    input, target, input_len, target_len = d_train[100]
    # d_test = ChatDataset(train=False)
    # print(len(d_train))
    # print(len(d_test))
    print(target)
    # chatbot_data_split()

def test_dataloader():
    loader = get_dataloader()
    for idx,(input, target, input_len, target_len) in enumerate(loader):
        print(idx)
        print(input)
        print(target)
        print(input_len)
        print(target_len)
        break

# def train_chatbot(): #进行模型的训练
#     from chatbot.seq2seq import Seq2Seq
#     from torch.optim import Adam
#     from matplotlib import pyplot as plt
#     import config
#
#     model = Seq2Seq().to(config.device)
#     optimizer = Adam(model.parameters(),lr=0.1)
#
#     # 自定义初始化参数
#     for name, param in model.named_parameters():
#        if 'bias' in name:
#            torch.nn.init.constant_(param, 0.0)
#        elif 'weight' in name:
#            torch.nn.init.xavier_normal_(param)
#
#     loss_list = []
#
#     for i in range(5):
#         train(i,model,optimizer,loss_list)
#     plt.figure(figsize=(50, 8))
#     plt.plot(range(len(loss_list)), loss_list)
#     plt.show()


if __name__ == '__main__':
    # prepar_ws()
    # test_dataset()
    test_dataloader()
    # train_chatbot()
    # eval()
    # interface()










