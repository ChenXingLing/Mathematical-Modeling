import numpy
import torch
from torch import nn
from matplotlib import pyplot as plot
import sys

#————[模型搭建]————#
class FNN(nn.Module):
    def __init__(self,wide:list,act): #wide储存每层节点数
        super(FNN,self).__init__()
        self.act=act
        # self.drop=nn.Dropout(0.1)
        self.soft=nn.LogSoftmax(dim=0)
        self.line_n=len(wide)-1
        self.line=nn.ModuleList()
        for i in range(self.line_n):
            self.line.append(nn.Linear(wide[i],wide[i+1]))
        
            
    def forward(self,x):
        for i,line_f in enumerate(self.line):
            x=line_f(x) #线性函数
            if i<self.line_n-1:
                # x=self.drop(x) #随机丢弃参数
                x=self.act(x) #激活函数
            else:
                x=self.soft(x) #softmax
        return x

#————[性能评估]————#
def pred(model,criterion,x_,y_,name="test60",title="",draw=False,picname="",picnum=0):

    y_pred=model(x_) #对指定数据调用模型进行预测
    loss=criterion(y_pred,y_)

    acc,tot=0,0 #acc:预测正确的个数 tot:总个数
    for i,y_p in enumerate(y_pred):
        if y_p.argmax()==y_[i]:
            acc+=1
        tot+=1

    if draw==True:
        # 绘图比较
        fig,aex=plot.subplots()
        aex.set_title("[{}\n{}_acc={:.6f}]".format(title,name,acc/tot))
        aex.set_xlabel('x')
        aex.set_ylabel('y')
        
        xr,yr,xg,yg,xb,yb,xk,yk=[],[],[],[],[],[],[],[]
        for i,(x,y) in enumerate(x_): #点分类
            if y_pred[i].argmax()==y_[i]:
                if y_[i]==0:
                    xr.append(x)
                    yr.append(y)
                if y_[i]==1:
                    xg.append(x)
                    yg.append(y)
                if y_[i]==2:
                    xb.append(x)
                    yb.append(y)
            else:
                xk.append(x)
                yk.append(y)
        # 绘散点
        aex.scatter(xr,yr,c='r',label="0")
        aex.scatter(xg,yg,c='g',label="1")
        aex.scatter(xb,yb,c='b',label="2")
        aex.scatter(xk,yk,c='k',label="fail") #预测失败的点为黑色
        aex.legend() #添加图例
        plot.savefig('{}_{}_{}.png'.format(picname,name,picnum))
        # plot.show()
        #if draw_show==True:
            #plot.show()


    return loss.detach().numpy(),acc/tot #计算准确率

if __name__=='__main__':

    #————[参数]————#
    
    _dataset=2 #使用数据集1/2
    _N=450 #数据集总量
    
    _wide=128 #网络宽度(隐藏层每层节点数量)
    _depth=3 #网络深度(隐藏层数量)
    _act=nn.LeakyReLU() #激活函数类型(Sigmoid,Tanh,ELU,ReLU,LeakyReLU)
    
    _picname="(dataset=2)" #存图名称
    _pictitle="dataset={} width={} depth={}".format(_dataset,_wide,_depth) #绘图标题

    print("[使用数据集 dataset={}]\n[网络宽度 wide={}]\n[网络深度 depth={}]\n[激活函数类型 activation={}]\n".format(_dataset,_wide,_depth,_act))

    #————[读入数据集]————#
    numpy.random.seed(233)
    
    # 训练集大小450 测试集大小150
    train=numpy.loadtxt("dataset/insects-{}-training.txt".format(_dataset))

    numpy.random.shuffle(train) #随机打乱顺序
    x1,y1=train[:,0:2],train[:,2]
    y1=numpy.round(y1).astype(numpy.int64)

    test=numpy.loadtxt("dataset/insects-{}-testing.txt".format(_dataset))
    
    common_=numpy.concatenate((test[0:20],test[70:90],test[140:160])) #前60个
    new_=numpy.concatenate((test[20:70],test[90:140],test[160:210])) #后150个
    # print("common_.shape:",numpy.array(common_).shape,common_)
    # print("new_.shape:",numpy.array(new_).shape,new_)

    numpy.random.shuffle(common_) #随机打乱顺序
    x2,y2=common_[:,0:2],common_[:,2]
    y2=numpy.round(y2).astype(numpy.int64)
    numpy.random.shuffle(new_) #随机打乱顺序
    x3,y3=new_[:,0:2],new_[:,2]
    y3=numpy.round(y3).astype(numpy.int64)

    x_train,x_test60,x_test150=torch.Tensor(x1),torch.Tensor(x2),torch.Tensor(x3) #数据转化成张量
    y_train,y_test60,y_test150=torch.LongTensor(y1),torch.LongTensor(y2),torch.LongTensor(y3) #y使用长整型

    _times=3 #重复实验次数
    loss_60,acc_60,loss_150,acc_150=[],[],[],[] #网络性能（loss:损失值 acc:正确率）
    for T in range(_times):
        print("第{}/{}次重复实验：".format(T+1,_times))

        #————[模型搭建]————#
        model=FNN(wide=[2,*[_wide]*_depth,3],act=_act) #使用FNN模型
        criterion=nn.NLLLoss() #对数损失函数
        optimizer=torch.optim.Adam(model.parameters(),lr=0.0025) #Adam优化器

        #————[模型训练]————#
        _epochs=300 #总训练次数
        dataset=torch.utils.data.TensorDataset(x_train,y_train) #使用训练集
        dataloader=torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True) #将训练集分批次 (shuffle: 随机打乱数据)

        for i in range(_epochs+1):
            if i>0:
                for j, (x_, y_) in enumerate(dataloader):
                    y_pred=model(x_) #使用当前模型，预测训练数据x
                    loss=criterion(y_pred,y_) #利用对数损失函数计算预测值与真实值之间的误差
                    optimizer.zero_grad() #将梯度初始化清空
                    loss.backward() #通过自动微分计算损失函数关于模型参数的梯度
                    optimizer.step() #优化模型参数，减小损失值

            if (i%(_epochs/20)==0) or (i==1 or i==_epochs):
                # loss60,acc60=pred(model,criterion,x_test60,y_test60,"test60",_pictitle,i==_epochs,_picname,T)
                loss150,acc150=pred(model,criterion,x_test150,y_test150,"test150",_pictitle,i==_epochs,_picname,T)
                print("[epoch={}/{}, test150_acc={:.6f}]".format(i,_epochs,acc150))
        
        #————[调参分析/数据测试]————#
        loss_150.append(loss150),acc_150.append(acc150)
        print("[test150_acc={:.6f}]\n".format(acc150))
        plot.show()

    #计算网络性能平均值
    print("acc_150:",['{:.6f}'.format(p) for p in acc_150],"acc_150_average={:.6f}".format(numpy.mean(acc_150)))