from random import shuffle
import pandas as pd
import numpy as np
import h5py

src_dir = './dataset'
dest_file = './dataset/data.h5'

names = ['abhishek', 'ahmad', 'akarsh', 'avatar', 'chaitanya', 'champ', 'harshith', 'ishan',
       'kalvik', 'manish', 'nishad', 'pavan', 'phani', 'prabhu', 'raghu', 'rahul', 'sanjay', 'shuang',
       'subramaniam', 'sushal', 'temesgen', 'vinay']

num_people = len(names)
num_rows = 90000
val_split = 0.15
test_split = 0.15
num_obs = 15
num_cols = 6
mean_noise = 0
std_noise = 0.1

val_num = int((val_split*num_people*num_obs)/num_people)
test_num = int((test_split*num_people*num_obs)/num_people)
print("val_num: ", val_num, "test_num: ", test_num, "train_num: ", num_obs-test_num-val_num)

print("merging csv")

val_x = []
val_y = []
test_x = []
test_y = []
train_x = []
train_y = []

for index, name in enumerate(names):
    print('merging data for ', name)

    filenames = []
    for i in range(1,num_obs+1):
        filenames.append(src_dir + '/{0}/{0}{1}.csv'.format(name, i))
    
    dfs = []
    for filename in filenames:
        df = np.array(pd.read_csv(filename))
        df = df[0:num_rows, :]
        if (df.shape[0] != 90000):
            print (filename)
        dfs.append(df)
        
    shuffle(dfs) 
       
    val_x.extend(dfs[:val_num])
    test_x.extend(dfs[val_num:(val_num+test_num)])
    train_x.extend(dfs[(val_num+test_num):])
    
    val_y.extend(index for _ in range(val_num))
    test_y.extend(index for _ in range(test_num))
    train_y.extend(index for _ in range(num_obs-test_num-val_num))

print("transforming training data")   

train_x_temp = []
train_y_temp = []

for i in range(0,len(train_x)):
    

    print("transforming training data ", len(train_x), "/", i)   
    mag = train_x[i][:, :3]
    phase = train_x[i][:, 3:]
    
    flip_lr_mag = np.fliplr(mag)
    flip_lr_phase = np.fliplr(phase)
    
    flip_join = np.hstack((flip_lr_mag, flip_lr_phase))

    flip_updown = np.flipud(train_x[i])
    
    flip_combo = np.flipud(flip_join)
    
    gaussian_noise = np.random.normal(mean_noise, std_noise, [num_rows, num_cols])
    
    data_corrupt = train_x[i] + gaussian_noise
    
    train_x_temp.append(train_x[i])
    train_x_temp.append(flip_join)
    train_x_temp.append(flip_updown)
    train_x_temp.append(flip_combo)
    train_x_temp.append(data_corrupt)
    
    train_y_temp.extend(train_y[i] for _ in range(5))
    
train_x = train_x_temp
train_y = train_y_temp
    
print("Training label set  size: ", len(train_y), "\nTraining data set size: ", len(train_x))
print ("Converting data to numpy arrays")
val_x = np.array(val_x)
val_y = np.array(val_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
train_x = np.array(train_x)
train_y = np.array(train_y)

print ("Val data and labels: ", val_x.shape, val_y.shape, "\ntest data and labels: ", test_x.shape, test_y.shape, "\ntrain data and labels: ", train_x.shape, train_y.shape)
print ("Generating h5 file")

hf = h5py.File(dest_file, 'w')
hf.create_dataset('train_x', data=train_x)
hf.create_dataset('train_y', data=train_y)
hf.create_dataset('test_x', data=test_x)
hf.create_dataset('test_y', data=test_y)
hf.create_dataset('val_x', data=val_x)
hf.create_dataset('val_y', data=val_y)
hf.close()
    
