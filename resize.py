from PIL import Image
import glob

k = 0
for imageName in glob.glob('H:/SSIM/microlab/image/0125mo/test_AB/*.*'):
    print('ooo')
    img = Image.open(imageName).convert('RGB')
    (w, h) = img.size
    print('w=%d, h=%d', w, h)
    #img.show()
    new_img = img.resize((512, 256))
    #new_img.show()
    new_img.save('H:/SSIM/microlab/image/0125mr/AB/train/' + str(k)+ '.png')
    k+=1
#--------------------------------
#END
