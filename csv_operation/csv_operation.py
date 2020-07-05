import csv
import os

def write_csv():
    header = ['number','name','gender','math','score']
    rows = [
            [1,'xiaoming','male',168,23],
            [2,'xiaohong','female',162,22],
            [3,'xiaozhang','female',163,21],
            [4,'xiaoli','male',158,21]
        ]
    #写入操作
    with open('test.csv','w')as f:
        f_csv = csv.writer(f)
        print(f_csv)
        print(type(f_csv))
        f_csv.writerow(header)
        f_csv.writerows(rows)

def read_csv():
    #读取操作
    list_headers=[]
    list_rows=[]
    with open('test.csv','r')as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            print(row)
            list_rows.append(row)
        list_rows.sort(key=lambda x:x[3],reverse=True)
    with open('new_test.csv','w')as f:
        f_csv = csv.writer(f)
        f_csv.writerows(list_rows)

def process(dir):
    for i in os.listdir(dir):
        print(str(dir)+'/'+ str(i))
        if os.path.isdir(str(dir)+'/'+ str(i)):
            for j in os.listdir(str(dir) + '/' + str(i)):
                if int(j.strip('jpgpng.')) % 3 !=0 :
                    os.remove(os.path.join((str(dir) + '/' + str(i)),j))

def process_csv(dir):
    with open(dir, 'rt') as inp, open('first_edit.csv', 'w',newline='') as out:
        writer = csv.writer(out)
        reader = csv.reader(inp)
        for row in reader:
            i ,j =row[0],row[1]
            if int(i.rpartition('/')[2].rstrip('.jpg')) % 6 == 0:
                writer.writerow([i,j])


dir = '/data/nyuv2/data/nyu2_train.csv'


if __name__ == '__main__':
    process_csv(dir)
