from math import sqrt
from matplotlib import pyplot
import numpy as np
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
a=[]
r=[]
m=[]
n=[]
e=[]
b=[]
s=[]
count=1
count1=1
dataset_name="LORENZ2"#"TRAFFIC"
dataset_name1="lorenz"#"traffic"
for i in range(30):
    origin = read_csv('./NSNP-AU/'+dataset_name+'/origin/' + str(i+1) + '_'+dataset_name1+'.csv', delimiter=',',header=None).values
    predictions=read_csv('./NSNP-AU/'+dataset_name+'/prediction/' + str(i+1) + '_'+dataset_name1+'.csv', delimiter=',',header=None).values
    predictions1=read_csv('./LSTM/'+dataset_name+'/prediction/' + str(i+1) + '_'+dataset_name1+'.csv', delimiter=',',header=None).values
    predictions2=read_csv('./GRU/'+dataset_name+'/prediction/' + str(i+1) + '_'+dataset_name1+'.csv', delimiter=',',header=None).values
    predictions3=read_csv('./PeePholeLSTM/'+dataset_name+'/prediction/' + str(i+1) + '_'+dataset_name1+'.csv', delimiter=',',header=None).values
    origin = origin[:,0:1]
    predictions = predictions[:,0:1]
    predictions1 = predictions1[:,0:1]
    predictions2 = predictions2[:, 0:1]
    predictions3 = predictions3[:, 0:1]#1:2
    rmse = sqrt(mean_squared_error(origin, predictions))
    rmse2 = sqrt(mean_squared_error(origin, predictions2))
    rmse1 = sqrt(mean_squared_error(origin, predictions1))*sqrt(len(origin))/sqrt(len(origin)-1)
    rmse3 = sqrt(mean_squared_error(origin, predictions3))
    mse = mean_squared_error(origin, predictions)
    mse1 = mean_squared_error(origin, predictions1)
    mse2 = mean_squared_error(origin, predictions2)
    mse3 = mean_squared_error(origin, predictions3)
    mae= mean_absolute_error(origin, predictions)
    mae1= mean_absolute_error(origin, predictions1)
    mae2= mean_absolute_error(origin, predictions2)
    mae3= mean_absolute_error(origin, predictions3)
    meanV = np.mean(origin)  # 对整个origin求取均值返回一个数
    dominator = np.linalg.norm(predictions - meanV, 2)
    dominator1 = np.linalg.norm(predictions1 - meanV, 2)
    dominator2 = np.linalg.norm(predictions2 - meanV, 2)
    dominator3 = np.linalg.norm(predictions3 - meanV, 2)
    nmse = mse / np.power(dominator, 2)
    nmse1 = mse1 / np.power(dominator1, 2)
    nmse2 = mse2 / np.power(dominator2, 2)
    nmse3 = mse3 / np.power(dominator3, 2)
    error = abs(origin - predictions)
    error1 = abs(origin - predictions1)
    error2 = abs(origin - predictions2)
    error3 = abs(origin - predictions3)
    Len=len(origin)
    avr=np.sqrt(Len*(Len)*np.power(dominator, 2))
    avr1 = np.sqrt(Len * (Len) * np.power(dominator1, 2))
    avr2 = np.sqrt(Len * (Len) * np.power(dominator2, 2))
    avr3 = np.sqrt(Len * (Len) * np.power(dominator3, 2))
    nrmse=Len*mse/avr
    nrmse1 = Len * mse1 / avr1
    nrmse2 = Len * mse2 / avr2
    nrmse3 = Len * mse3 / avr3
    smape=2.0 * np.mean(np.abs(predictions - origin) / (np.abs(predictions) + np.abs(origin))) #* 100
    smape1 = 2.0 * np.mean(np.abs(predictions1 - origin) / (np.abs(predictions1) + np.abs(origin)))  # * 100
    smape2 = 2.0 * np.mean(np.abs(predictions2 - origin) / (np.abs(predictions2) + np.abs(origin)))  # * 100
    smape3 = 2.0 * np.mean(np.abs(predictions3 - origin) / (np.abs(predictions3) + np.abs(origin)))  # * 100
    a.append(mae3)
    r.append(rmse3)
    m.append(mse3)
    n.append(nmse3)
    e.append(error3)
    b.append(nrmse3)
    s.append(smape3)
    fig4 = pyplot.figure()
    ax41 = fig4.add_subplot(111)
    pyplot.xticks(fontsize=12)
    pyplot.yticks(fontsize=12)
    ax41.set_xlabel("Time", fontsize=12)
    ax41.set_ylabel("Magnitude", fontsize=12)
    pyplot.plot(origin[:500,0:1], 'k--', label=' original data')
    pyplot.plot(predictions[:500,0:1], 'r*-', label=' NSNP-AU ')
    pyplot.plot(predictions3[:500,0:1], 'b-*', label=' PeepholeLSTM ')
    pyplot.plot(predictions1[:500,0:1], 'y+-', label=' LSTM ')
    pyplot.plot(predictions2[:500,0:1], 'g--', label=' GRU ')
    pyplot.legend()
    pyplot.title("LORENZ" )
    tt_name = "F:\单步预测\单步预测\NSNP-AU\\" + dataset_name + '\\' + dataset_name + '{}.png'
    pyplot.savefig(tt_name.format(count))
    count = count + 1
    # pyplot.show()
    # # 作图展示2
    fig1 = pyplot.figure()
    ax42 = fig1.add_subplot(111)
    pyplot.xticks(fontsize=12)
    pyplot.yticks(fontsize=12)
    ax42.set_xlabel("Time", fontsize=12)
    ax42.set_ylabel("Magnitude", fontsize=12)
    pyplot.plot(error[:500,0:1], 'r-+', label='NSNP-AU')
    pyplot.plot(error1[:500,0:1], 'y-*', label='LSTM')
    pyplot.plot(error2[:500,0:1], 'k--', label='GRU')
    pyplot.plot(error3[:500,0:1], 'b--', label='PeepholeLSTM')
    # pyplot.plot(predictions, 'g+-', label='the predicted data')
    pyplot.legend()
    pyplot.title("LORENZ")
    tt_name = "F:\单步预测\单步预测\NSNP-AU\ERROR\\" + dataset_name + '\\' + dataset_name + '{}.png'
    pyplot.savefig(tt_name.format(count1))
    count1 = count1 + 1
    # pyplot.show()
A=np.var(a)
R=np.var(r)
M=np.var(m)
N=np.var(n)
E=np.var(b)
S=np.var(s)
# print(dataset_name+' Test S2R: %.15f,S2M:%.15f,S2N:%.15f' % (R,M,N))
b4=len(a)
b1 = len(r)
b2 = len(m)
b3 = len(n)
b5=len(b)
b6=len(s)
sum1 = 0
sum2 = 0
sum3 = 0
sum4=0
sum5=0
sum6=0
for i in r:
    sum1 = sum1 +i
AVGR=(sum1/b1)
for i in m:
    sum2 = sum2 +i
AVGM=(sum2/b2)
for i in n:
    sum3 = sum3 +i
AVGN=(sum3/b3)
for i in a:
    sum4 = sum4 +i
AVGA=(sum4/b4)
for i in b:
    sum5 = sum5 +i
AVGE=(sum5/b5)
for i in s:
    sum6 = sum6 +i
AVGS=(sum6/b6)

# print(r)
# print(m)
# print(n)
# print(e)
L=r.index(min(r))+1
print(r)
print(a)
print(b)
print(s)
print (dataset_name+' Test RMSE: %.15f ,RMSE: %.15f ,RMSE: %.15f,RMSE: %.15f,RMSE: %.15f ' %(r[r.index(min(r))],m[r.index(min(r))],n[r.index(min(r))],b[r.index(min(r))],s[r.index(min(r))]))
print(L)
# print(dataset_name+' Test AVGR: %.10f,AVGM:%.10f,AVGN:%.10f' % (AVGR,AVGM,AVGN))
print(dataset_name+' Test RMSE: %.10f ± %.15f,MAE: %.10f ± %.15f,MSE: %.10f ± %.15f,NMSE: %.10f ± %.15f,NRMSE: %.10f ± %.15f,SMAPE: %.10f ± %.15f' %(AVGR,R,AVGA,A,AVGM,M,AVGN,N,AVGE,E,AVGS,S))
