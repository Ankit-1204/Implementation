import cv2
from K_Means import cluster

image= cv2.imread("imag.jpg")

re_image=image.reshape(-1,3)

k=256
clus=cluster(k)
final_cent , final_centroid=clus._predict(re_image)

new_im=re_image
for i in range(541696):
    new_im[i]=final_centroid[final_cent[i]]

fin_image=new_im.reshape(736,736,3)
cv2.imwrite('output_image.jpg', fin_image)