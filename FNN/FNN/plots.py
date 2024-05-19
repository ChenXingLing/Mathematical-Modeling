import numpy
from matplotlib import pyplot as plot
import sys

def work(picname,draw_show,title,xlb,ylb,x,y,xticks=None):
    fig,aex=plot.subplots()
    aex.set_title(title)
    aex.set_xlabel(xlb)
    aex.set_ylabel(ylb)
    if xticks!=None:
        plot.xticks(x,xticks)
    aex.scatter(x,y)
    #for a,b in zip(x,y):
    #    plot.text(a,b,b,ha='center')
    aex.plot(x,y,linestyle=':')
    plot.savefig('{}.png'.format(picname))
    if draw_show==True:
        plot.show()

if __name__=='__main__':

    work(picname="Test150_act",draw_show=True,
        title="[dataset=1 wide=128 depth=3 act=x]",xlb='activation',ylb='acc',
        x=[1,2,3,4,5],
        y=[0.953333,0.982222,0.975556,0.982222,0.988889],
        xticks=['Sigmoid','Tanh','ELU','ReLU','LeakyReLU'])

    work(picname="Test150_depth",draw_show=True,
        title="[dataset=1 wide=128 depth=x act=LeakyReLU]",xlb='depth',ylb='acc',
        x=[1,2,3,4,5,6,7,8,9],
        y=[0.940000,0.991111,0.984444,0.977778,0.975556,0.975556,0.975556,0.971111,0.951111])

    work(picname="Test150_wide",draw_show=True,
        title="[dataset=1 wide=x depth=3 act=LeakyReLU]",xlb='wide',ylb='acc',
        x=[1,2,3,4,5,6,7],
        y=[0.900000,0.946667,0.973333,0.971111,0.973333,0.984444,0.986667],
        xticks=['4','8','16','32','64','128','256'])

    