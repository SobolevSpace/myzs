from pathlib import Path

if __name__=='__main__':
    val_path = Path('./validation/').absolute()
    val_list = [x for x in val_path.rglob('*.wav')]
    print(val_list)
    print('val end\n')
    data_path = Path('./dataset/english/test/').absolute()
    data_list = [x for x in data_path.rglob('*.wav')]
    print(data_list)
    print('data end\n')
    res = list()
    for wav_path in val_list:
        name = wav_path.name
        id = name[5:-4]
        for init_wav in data_list:
            if id in init_wav.name:
                t = ['./test/'+init_wav.name, name[0:4], name]
                res.append(t)
                break
    f = open('./dataset/english/synthesis.txt','w')
    f.write(str(res).replace('\'', '\"'))
    f.close()
    #print(res)
    #print(len(res))