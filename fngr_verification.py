import cv2
import os
import random, glob


'''
This function accepts an image to be compred to templates,
comutes keypoints using sift detector and descriptor and 
return best matching score and template image name.
'''
def verify(test_img):

    best_score= 0
    file_name= None
    flag= False
    for file in [file for file in os.listdir('train_real')]:
        
        ref_img= cv2.imread(f'./train_real/{file}')

        sift= cv2.SIFT_create()

        kp_test, desc_test= sift.detectAndCompute(test_img, None)
        kp_ref, desc_ref= sift.detectAndCompute(ref_img, None)

        matches= cv2.FlannBasedMatcher(dict(algorithm=1, trees=10),
                                        dict()).knnMatch(desc_test, desc_ref, k=2)

        match_points= []

        for p,q in matches:
            if p.distance<(0.1*q.distance):
                match_points.append(p)
            
            keys= 0
            if len(kp_test)<=len(kp_ref):
                keys= len(kp_test)
            else:
                keys= len(kp_ref)
            
        score= len(match_points)/keys*100

        if score>best_score:
            best_score= score
            file_name= file

    return best_score, file_name
        
    '''if (score)>0.90:
            print(f'{score*100}% matching score with tempalte {file}')
            flag= True
    if flag!=True:
        print('Matchong according to 90% threshold faild!')'''



    
if __name__=='__main__':

    cwd = os.getcwd()
    imgs= glob.glob(cwd+'/test_alterd/*')
    random_img= random.choice(imgs)
    test_me= cv2.imread(random_img)    # enter your fingerprint image here
    
    best_score, file_name= verify(test_me)
    print(f'{best_score}% matching score with tempalte {file_name}')