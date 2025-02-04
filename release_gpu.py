import os
import time
import argparse
 
def get_exec_out(sxcute_str):
    out_list = os.popen(sxcute_str).readlines()
    return out_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keywords", type=str,default=None)
    args = parser.parse_args()

    keywords = "test"

    if args.keywords==None:
        print("no keywords")
    else:
        keywords = args.keywords

    excute_str = 'nvidia-smi'
    out_list = get_exec_out(excute_str)
    #print(out_list)
    for oo in out_list:
        #print(oo)
        if oo.find(keywords) != -1:
            # split()函数默认可以按空格分割，并且把结果中的空字符串删除掉，留下有用信息
            proc_list = oo.split()
            pid = proc_list[4].strip()
            kill_str = 'kill -9 ' + pid
            print(kill_str)
            time.sleep(0.3)
            os.system(kill_str)
