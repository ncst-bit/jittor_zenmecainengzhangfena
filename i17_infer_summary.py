def infer(animal_path,caltech_path,food_path,thudog_path,car_path):

    f1=open(animal_path).readlines()
    f2=open(caltech_path).readlines()
    f3=open(food_path).readlines()
    f4=open(thudog_path).readlines()
    f5=open(car_path).readlines()

    classes = open('./classes_b.txt').read().splitlines()
    class2idx_summary={}
    for c in classes:
        c=c.replace('Animal_','').replace('Thu-dog_','').replace('Caltech-101_','').replace('Food-101_','').replace('Stanford-Cars_','')
        class2idx_summary[c.split(' ')[0]]=c.split(' ')[1]

    idx2class_animal={}
    count = 0
    for c in classes+['Animal_Shark','Animal_Shark','Animal_Eagle','Animal_Whale','Animal_Dolphin','Animal_Dolphin']:
        c1 = c.split(' ')[0]
        if c1.startswith('Animal'):
            c1 = c[7:].split(' ')[0]
            idx2class_animal[str(count)] = c1
            count += 1

    idx2class_food={}
    for c in classes:
        c1 = c.split(' ')[0]
        if c1.startswith('Food-101'):
            c1 = c[9:].split(' ')[0]
            idx2class_food[str(int(c.split(' ')[1]) - 143)] = c1

    idx2class_caltech={}
    count = 0
    for c in classes:
        c1 = c.split(' ')[0]
        if c1.startswith('Caltech-101'):
            c1 = c[12:].split(' ')[0]
            idx2class_caltech[str(count)] = c1
            count += 1

    idx2class_car={}
    count = 0
    for c in classes:
        c1 = c.split(' ')[0]
        if c1.startswith('Stanford-Cars'):
            c1 = c[14:].split(' ')[0]
            idx2class_car[str(count)] = c1
            count += 1

    result=''
    for i in f1:
        i=i.replace('\n','')
        temp=[class2idx_summary[idx2class_animal[idx]] for idx in i.split(' ')[1:]]
        result=result+i.split(' ')[0]+' '+' '.join(temp)+'\n'
    for i in f2:
        i = i.replace('\n', '')
        temp=[class2idx_summary[idx2class_caltech[idx]] for idx in i.split(' ')[1:]]
        result=result+i.split(' ')[0]+' '+' '.join(temp)+'\n'
    for i in f3:
        i = i.replace('\n', '')
        temp=[class2idx_summary[idx2class_food[idx]] for idx in i.split(' ')[1:]]
        result=result+i.split(' ')[0]+' '+' '.join(temp)+'\n'
    for i in f4:
        i = i.replace('\n', '')
        result=result+i.split(' ')[0]+' '+' '.join([str(int(idx)+244) for idx in i.split(' ')[1:]])+'\n'
    for i in f5:
        i=i.replace('\n','')
        temp=[class2idx_summary[idx2class_car[idx]] for idx in i.split(' ')[1:]]
        result=result+i.split(' ')[0]+' '+' '.join(temp)+'\n'
    f=open('result.txt','w')
    f.write(result)
    f.close()