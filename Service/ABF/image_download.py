# library import
import urllib.request
import glob
import argparse

txt_file = opt.txt
print('Reading txt file.......')

#open txt file
with open(txt_file, 'r') as file:
    txt = file.read().split('\n')

print('total txt length :',len(txt))

# download image
for idx in range(len(txt)):
    try:
        urllib.request.urlretrieve(txt[idx], './downloaded_image/'+str(idx)+'.jpg')
    except :
        pass

image_list = glob.glob('./images/'+ '*.jpg')
print('total saved images count :', len(image_list))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', type=str, default='image_link.txt')
    opt = parser.parse_args()