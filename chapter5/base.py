import os, shutil

train_dir='/home/guohongliang/download/train'

train_cat_dir='/home/guohongliang/project/keras/dog_and_cat/train_dir/cats/'
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
	src = os.path.join(train_dir, fname)
	dst = os.path.join(train_cat_dir, fname)
	shutil.copyfile(src, dst)

print('total train cat images', len(os.listdir(train_cat_dir)))
val_cat_dir='/home/guohongliang/project/keras/dog_and_cat/val_dir/cats/'
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
	src = os.path.join(train_dir, fname)
	dst = os.path.join(val_cat_dir, fname)
	shutil.copyfile(src, dst)
print('total val cat images', len(os.listdir(val_cat_dir)))

test_cat_dir='/home/guohongliang/project/keras/dog_and_cat/test_dir/cats/'
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
	src = os.path.join(train_dir, fname)
	dst = os.path.join(test_cat_dir, fname)
	shutil.copyfile(src, dst)
print('total test cat images', len(os.listdir(test_cat_dir)))

train_dog_dir='/home/guohongliang/project/keras/dog_and_cat/train_dir/dogs/'
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
	src = os.path.join(train_dir, fname)
	dst = os.path.join(train_dog_dir, fname)
	shutil.copyfile(src, dst)
print('total train dog images', len(os.listdir(train_dog_dir)))

val_dog_dir='/home/guohongliang/project/keras/dog_and_cat/val_dir/dogs/'
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
	src = os.path.join(train_dir, fname)
	dst = os.path.join(val_dog_dir, fname)
	shutil.copyfile(src, dst)
print('total val dog images', len(os.listdir(val_dog_dir)))

test_dog_dir='/home/guohongliang/project/keras/dog_and_cat/test_dir/dogs/'
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
	src = os.path.join(train_dir, fname)
	dst = os.path.join(test_dog_dir, fname)
	shutil.copyfile(src, dst)
print('total test dog images', len(os.listdir(test_dog_dir)))
