from random import shuffle
import pandas as pd
import numpy as np
import h5py

#dataset_gdrive = https://drive.google.com/open?id=0B6FZBAhUFX8OaUJZRDkzaFVJQlk

src_dir = './dataset'
dest_file = './dataset/data.h5'

names = ['abhishek', 'ahmad', 'akarsh', 'avatar', 'chaitanya', 'champ', 'harshith', 'ishan',
       'kalvik', 'manish', 'nishad', 'pavan', 'phani', 'prabhu', 'raghu', 'rahul', 'sanjay', 'shuang',
       'subramaniam', 'sushal', 'temesgen', 'vinay']

num_people = len(names)
num_rows = 90000
val_split = 0.20
num_obs = 15
val_num = int((val_split*num_people*num_obs)/num_people)
print("val_num: ", val_num)
num_cols = 6
n_steps = num_rows * num_obs
num_aug_ops = 5
total_data_steps = num_rows * num_obs * num_aug_ops
mean_noise = 0
std_noise = 0.1

print("merging csv")

val_x = []
val_y = []
train_x = []
train_y = []

for index, name in enumerate(names):
    print('preparing data for ', name)

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
    train_x.extend(dfs[val_num:])
    val_y.extend(index for _ in range(val_num))
    train_y.extend(index for _ in range(num_obs - val_num))

for i in range(0,len(train_x)):
    mag = train_x[i][:, :3]
    phase = train_x[i][:, 3:]
    
    flip_lr_mag = np.fliplr(mag)
    flip_lr_phase = np.fliplr(phase)
    
    flip_join = np.hstack((flip_lr_mag, flip_lr_phase))

    flip_updown = np.flipud(train_x[i])
    
    flip_combo = np.flipud(flip_join)
    
    gaussian_noise = np.random.normal(mean_noise, std_noise, [num_rows, num_cols])
    
    data_corrupt = train_x[i] + gaussian_noise
    
    train_x.append(flip_join)
    train_x.append(flip_updown)
    train_x.append(flip_combo)
    train_x.append(data_corrupt)
    
    train_y.extend(train_y[i] for _ in range(4))
    
print(len(train_y), len(train_x))

val_x = np.array(val_x)
val_y = np.array(val_y)
train_x = np.array(train_x)
train_y = np.array(train_y)

print (val_x.shape, val_y.shape, train_x.shape, train_y.shape)

hf = h5py.File(dest_file, 'w')
hf.create_dataset('train_x', data=train_x)
hf.create_dataset('train_y', data=train_y)
hf.create_dataset('val_x', data=val_x)
hf.create_dataset('val_y', data=val_y)
hf.close()

