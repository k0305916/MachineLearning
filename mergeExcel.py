import sys
import os
import threading
import pandas as pd

def getFileName(path,myfilter):
    ''' 获取指定目录下的所有指定后缀的文件名 '''
    filelist = []
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == myfilter:
            filelist.append(i)
    return filelist

def FileProcess(filelist):
    # try:
    df1 = pd.DataFrame()
    director = ''
    colNo = 0
    for file in filelist:
        data = pd.read_excel(file,header=None)
        # print(data['Sheet1'])
        df1.insert(colNo,'col'+str(colNo),data[0])
        colNo+=1
    #若是采用xlsx格式，则需要还要一个库openPyXL   不想装了，就直接用xls格式了
    writer = pd.ExcelWriter('data/output.xls')
    df1.to_excel(writer,'Sheet1')
    writer.save()
    # except Exception:
    #     print('processing error,Check please.')

print('file path is:',sys.argv[1])
filelist = getFileName(sys.argv[1],'.xlsx')
filelist = list(map(lambda x: sys.argv[1]+'/'+x,filelist))
print('processing now, do not close windows please.')

#create a thread to process files
#args只有采用这种方式，传给target的才是一个参数，否则就会被拆解，然后传给target
FileProcessThread = threading.Thread(target=FileProcess, args=(filelist,))
FileProcessThread.start()
FileProcessThread.join()

print('processing over.')
