from datetime import time
import datetime
import math
import re
from turtle import speed
import numpy as np
from numpy.ma import sin
import pandas as pd
from pandas.core.frame import DataFrame
import pylab as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import optimize
import os

os.chdir('F:\\Mt\\windcube data\\hb\\data\\April-utc\\10ele\\')
ppifolder='F:\\Mt\\windcube data\\hb\\data\\April-utc\\10ele\\raw data\\' #文件夹路径
ppifilelist=os.listdir(ppifolder)
uncon=0
ava=pd.DataFrame()
con=0
data_all=pd.DataFrame()
drop_num=pd.DataFrame()

for i in ppifilelist:
    filename=i
    path=os.path.join(ppifolder,filename)
    DF=pd.read_csv(path,delimiter=';')  
    # DF=pd.read_csv('F:/Mt/windcube data/hb/data/April-utc/10ele/raw data/WLS100s-95_RadialWindData_2020-04-04_02-30-05_PPI_151_25m_WLS100S-95.csv',delimiter=';')
    df=pd.DataFrame(DF)
    desktop_path = "F:\\Mt\\windcube data\\hb\\data\\April-utc\\10ele\\out 100m dir\\"  # 新创建的文件的存放路径 
    full_path = desktop_path + i     

    #将时间转化为北京时
    filename1=filename[26:45]
    time0=str(filename1[0:4])+'/'+str(filename1[5:7])+'/'+str(filename1[8:10])+' '+str(filename1[11:13])+':'+str(filename1[14:16])+':00'
    time2 = datetime.datetime.strptime(time0,'%Y/%m/%d %H:%M:%S') + datetime.timedelta(hours=8)
    time2=str(time2)
    time_beijing=time2[0:4]+'-'+time2[5:7]+'-'+time2[8:10]+'_'+time2[11:13]+'.'+time2[14:16]
    time_compare=time2[:16]
    newname1=time_beijing+'.csv'
    newname2='拟合检验(R2大于0.65).csv'
    newpath1=os.path.join(desktop_path,newname1)
    newpath2=os.path.join(desktop_path,newname2)
    # df1=df
    # df1=df.loc[lambda x:x["Status"]==1]     #数据有效性过滤 置信度为1
    df1=df.loc[lambda x:x["CNR [dB]"] >=-27]     #CNR过滤 阙值为-27
    df1=df.loc[lambda x:x["CNR [dB]"] >=-27]

    allnum=0
    avanum=0
    avatime=time2[0:4]+'/'+time2[5:7]+'/'+time2[8:10]+'_'+time2[11:13]+':'+time2[14:16]
    date=avatime[8:10]
    month=avatime[5:7]
    data=pd.DataFrame()

    #雨天筛选，若有雨天则跳过
    count=0
    rainfall=pd.read_csv('F:/Mt/windcube data/hb/data/April-utc/10ele/rainfall.csv')
    rain_data=rainfall['time']
    for k in rain_data:
        if avatime == k:
            count +=1
    if count == 1:
        continue

    data_fit=pd.DataFrame()
    for k in range(50,625,25):   #js径向间隔为50m，hb和hb为25m  sin10=0.173648  625m
        j_re=k*np.sin(math.pi/18)
        j_re=round(j_re, 2)
        data1=df1.loc[lambda x:x["Range [m]"]==k] #选取径向距离
        data_com=df.loc[lambda x:x["Range [m]"]==k]

        # # #遮挡物方位角筛选 仅h b
        if k == 50:
            data1=data1[data1['LOS ID']!=2]
            data1=data1[data1['LOS ID']!=5]
            data1=data1[data1['LOS ID']!=6]
            data1=data1[data1['LOS ID']!=7]
            data1=data1[data1['LOS ID']!=8]
        if k == 75:
            data1=data1[data1['LOS ID']!=8]
        if k == 100 or k==125 or k==150:
            data1=data1[data1['LOS ID']!=6]
        
        data1=data1[["Timestamp","Azimuth [�]","Elevation [�]","Radial Wind Speed [m/s]","Range [m]",'CNR [dB]']]
        data1=data1.reset_index(drop=True)

        #CNR平均值计算
        cnr=data1['CNR [dB]']
        if len(cnr) !=0:
            cnr_avg=round(sum(cnr) / len(cnr),2)
        else:
            cnr_avg=0

        # 利用cnr筛选
        c=data1
        # for n in range(len(cnr)):
        #     a=cnr[n]
        #     qz=(a-cnr_avg)/cnr_avg

        #     if abs(qz) > 0.1:
        #         c=c.drop(n, axis = 0)
        #         d=np.array(data1.iloc[n])
        #         d_time=d[0]
        #         d_az=d[1]
        #         d_el=d[2]
        #         d_sp=d[3]
        #         d_ran=d[4]
        #         d_cnr=d[5]
        #         drop_num=drop_num.append(pd.DataFrame({'Time':[avatime],'Azimuth':[d_az],'Radial Wind Speed [m/s]':[d_sp],'Range [m]':[d_ran],'CNR [dB]':[d_cnr]}),ignore_index=True)
        # # print(drop_num)
        # c=c[abs(c['Radial Wind Speed [m/s]']) >= 1]   
        # c=c.reset_index(drop=True)     
        # print(c)
        # data2=c
        cnr_np=np.array(c['CNR [dB]'])
        cnr_std=np.std(cnr_np)
        # print(cnr_std)

        c2=c
        for n in range(len(c)):
            cnr_0=cnr_np[n]
            cnr_max=abs(cnr_0-cnr_avg)/cnr_std
            # print(cnr_max)
            if abs(cnr_max) > 1.2:   #尝试多种值，1--2之间，发现差距不大
                c2=c.drop(n, axis = 0)
                d=np.array(c.iloc[n])
                d_time=d[0]
                d_az=d[1]
                d_el=d[2]
                d_sp=d[3]
                d_ran=d[4]
                d_cnr=d[5]
                drop_num=drop_num.append(pd.DataFrame({'Time':[avatime],'Azimuth':[d_az],'Radial Wind Speed [m/s]':[d_sp],'Range [m]':[d_ran],'CNR [dB]':[d_cnr]}),ignore_index=True)
        c2=c2[abs(c2['Radial Wind Speed [m/s]']) >= 1]   
        c2=c2.reset_index(drop=True)     
        # print(c2)
        data2=c2
        # break
        try:
            if len(c2) > 10 :
                # data_12 = np.array(col_12)
                x0 = np.array(data_com["Azimuth [�]"])
                y0 = np.array(data_com["Radial Wind Speed [m/s]"])
                
                x = np.array(data2["Azimuth [�]"])
                y = np.array(data2["Radial Wind Speed [m/s]"])

                #进行非线性正弦拟合
                y_ave=sum(y)/len(y)   #计算平均值
                def fmax(x, a, b,c): 
                    return c+a*np.cos(np.pi*x/180-b)

                popt, pcov = curve_fit(fmax, x, y)       #最小二乘法拟合
                                                    
                a = popt[0]                              #获取popt里面是拟合系数,a为振幅，b为相移，c为偏移
                b = popt[1]
                c = popt[2]
                y2vals = fmax(x,a,b,c)
                # print(len(y2vals),y2vals)
                #拟合风速和实际风速误差
                y4=[]
                for h in range(len(y2vals)):
                    h_y=y[h]
                    h_y2=y2vals[h]
                    l=len(y2vals)
                    # print(h_y-h_y2)
                    speed_wc=np.sqrt(np.square(h_y-h_y2)/(l-3))
                    # print(speed_wc)
                    if speed_wc < 2:
                        y4.append(h_y2)
                y2vals=y4
                #利用标准化残差筛选 e1=Vf-V0
                Vf=np.array(y2vals)
                # print ('Vf:',Vf)
                V0=y
                e1=np.array(Vf)-np.array(V0)
                # print('e1:',e1)
                # break
                #计算残差标准偏差 e2
                def std(e1):
                    n = len(e1)
                    avg = sum(e1) / n
                    return (sum(map(lambda e: (e - avg) * (e - avg), e1)) / n) ** 0.5
                e2=(std(e1))
                # print('e2:',e2)
                #计算标准化残差 zre=e1/e2
                e1 = np.array(e1)
                zre = e1/e2 
                # x_r=x.reshape(-1, 1)
                # y_r=y.reshape(-1, 1)
                # e1_r=e1.reshape(-1, 1)
                # data3=np.hstack((x_r,y_r,zre_r,e1_r))
                zre_r=zre.reshape(-1, 1)
                data3=pd.DataFrame(zre_r,columns=['zre'])
                obj=data2
                obj['zre']=data3
                z0=obj
                for m in range(len(zre_r)):
                    zre_m=zre_r[m]        
                    if abs(zre_m) > 2:
                        z0=z0.drop(m, axis = 0)
                        z=np.array(obj.iloc[m])
                        z_time=z[0]
                        z_az=z[1]
                        z_el=z[2]
                        z_sp=z[3]
                        z_ran=z[4]
                        z_cnr=z[5]
                        drop_num=drop_num.append(pd.DataFrame({'Time':[avatime],'Azimuth':[z_az],'Radial Wind Speed [m/s]':[z_sp],'Range [m]':[z_ran],'CNR [dB]':[z_cnr]}),ignore_index=True)
                    # break
                z0=z0.reset_index(drop=True) 
                x = np.array(z0['Azimuth [�]'])
                y = np.array(z0['Radial Wind Speed [m/s]'])
                cnr_all=np.array(z0["CNR [dB]"])
                cnr_avg=round(sum(cnr_all) / len(cnr_all),2)

                def func(x, a2, b2,c2): 
                    return c2+a2*np.cos(np.pi*x/180-b2)
                popt, pcov = curve_fit(func, x,y)                                                         
                a2 = popt[0]                           
                b2 = popt[1]
                c2 = popt[2]
                y3vals = func(x,a2,b2,c2)

                #Find wind direction
                def func(x): 
                    return c2+a2*np.cos(np.pi*x/180-b2)
                minimum = optimize.fminbound(func, 0,360)   #风向
                
                #风向订正
                # if b2 > 0:
                #     minimum=minimum+180

                j_re=k*np.sin(math.pi/18)                # hb和hb数据ppi仰角为10度，js数据ppi仰角为15度，
                hd=abs(a/np.cos(math.pi/18))            #水平风速
                allnum += len(y0)
                avanum += len(y)
                rat=len(y0)/len(y)

                # 判断拟合优度  R^2 = ssr/sst
                ssr=sum((y3vals-y_ave)**2)                 #拟合值-平均值
                sst=sum((y-y_ave)**2)                     #观测值-平均值
                result=ssr/sst
    #     except:
    #         pass
    #     break
    # break
                # # #获取拟合后的曲线
                # x_fit=np.arange(0,360,10)
                # y_fit=c+a*np.cos(np.pi*x_fit/180-b)
                # # print(y_fit)
                # # #拟合绘图
                # j_re=round(j_re,2)
                # fig = plt.figure(figsize=(8,6))
                # ax1 = fig.add_subplot(111)
                # ax1.plot(x, y, 's',label='Original Values')
                # # ax1.plot(x, y2vals, 'o',label='Fit Values')
                # ax1.set_xlabel('Azimuth')
                # ax1.set_ylabel('Radial Wind Speed [m/s]')
                # title=str(time_beijing)+' '+str(j_re)+'m '+'R^2=' + str(result)
                # ax1.set_title(title)

                # ax1.plot(x_fit, y_fit, 'black',label='Sine Wave Fitting')
                # ax1.set_xlabel('Azimuth')
                # ax1.set_ylabel('Radial Wind Speed [m/s]')
                # ax1.legend(loc=2,frameon=False) 
                
            #     second_fit_name=str(time_beijing)+' '+str(j_re)+'m'+' fit'    #叠加CNR
            #     ax2 = ax1.twinx()
            #     ax2.plot(x, cnr_all, 'green',linestyle='--',label='CNR')
            #     ax2.set_ylim([-30, -5])
            #     ax2.set_ylabel('CNR [dB]')
            #     ax2.legend(loc=2,bbox_to_anchor=(0,0.86),frameon=False)

            #     file_path=(f'F:/Mt/windcube data/hb/data/April-utc/10ele/fit 100m/{month}/{date}')     #创建拟合图存储路径
            #     if not os.path.exists(file_path):
            #         os.makedirs(file_path)
            #     plt.savefig(f"F:/Mt/windcube data/hb/data/April-utc/10ele/fit 100m/{month}/{date}/{second_fit_name}.jpg")
            #     # plt.show()
            #     plt.close()

                data_fit=data_fit.append(pd.DataFrame({'time':[avatime],'h':j_re,'CNR':[cnr_avg],'R^2':[result]}),ignore_index=True)
                
                if result > 0.65 and result < 1:   
                    data=data.append(pd.DataFrame({'time':[avatime],'h':[j_re],'Horizontal Wind Speed':[hd],'Horizontal Wind Direction':[minimum],'R^2':[result],'cnr':[cnr_avg]}),ignore_index=True)
                    data.to_csv(newpath1,sep=';',index=False)
                else:
                    data=data=data.append(pd.DataFrame({'time':[avatime],'h':[j_re],'R^2':[result],'cnr':[cnr_avg]}),ignore_index=True)
                    data.to_csv(newpath1,sep=';',index=False)

                # data=data.append(pd.DataFrame({'h':[j_re],'Horizontal Wind Speed':[hd],'Horizontal Wind Direction':[minimum],'R^2':[result]}),ignore_index=True)
                # data.to_csv(newpath1,sep=';',index=False)
                # data_all=data_all.append(pd.DataFrame({'time':[avatime],'range':[j],'h':[j_re],'Lidar Speed':[hd],'Lidar Direction':[minimum],'CNR':[cnr_avg],'R^2':[result]}),ignore_index=True)
            else:
                data=data=data.append(pd.DataFrame({'time':[avatime],'h':[j_re],'cr':[cnr_avg]}),ignore_index=True)
                data.to_csv(newpath1,sep=';',index=False)
    
        except:
            uncon +=1
            data=data.append(pd.DataFrame({'time':[avatime],'h':[j_re]}),ignore_index=True)
            data.to_csv(newpath1,sep=';',index=False)
            # data_all=data_all.append(pd.DataFrame({'time':[avatime],'range':[j],'h':[j_re]}),ignore_index=True)
        # print(data)
        # break
        # break
    con += 1 
    data_fit.to_csv(newpath2,sep=';',mode='a',header=None,index=False) 
    ava=ava.append(pd.DataFrame({'time':[avatime],'all numbers':[allnum],'available numbers':[avanum]}),ignore_index=True)
    ava.to_csv('F:/Mt/windcube data/hb/data/April-utc/10ele/out 100m dir/10ele Availability.csv',sep=';',index=False)
    drop_num.to_csv('F:/Mt/windcube data/hb/data/April-utc/10ele/out 100m dir/drop data.csv',sep=';',index=False)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('now time is:',nowtime)
    print('run: ',con)
    # # break
    # break
# data_all.to_csv(newpath1,sep=';',index=False)
# ava.to_csv('F:/Mt/windcube data/hb/data/April-utc/10ele/10m/10ele Availability.csv',sep=';',index=False)

print('all finish!')