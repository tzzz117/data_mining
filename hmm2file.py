#!/usr/bin/env python
import hmm
import sys

    
def main(mbegin, mend):   
    
    name = str(mbegin)+'_'+str(mend)    

    hmmre = open(name+'hmm_accu.txt', 'w') 
    nbre = open(name+'nb_accu.txt', 'w') 
    
    for i in range(mbegin, mend):
        
        n_char, nb_accu, hmm_accu = hmm.main(i)
        
        nbre.write(str(n_char)+' : '+str(nb_accu)+'\n')
        hmmre.write(str(n_char)+' : '+str(hmm_accu)+'\n')
    
    hmmre.close()
    nbre.close()
    
    
if __name__ == '__main__':
    if (len(sys.argv) > 1):
        mbegin = int(sys.argv[1])
        mend = int(sys.argv[2])
    else:
        mbegin = 0
        mend = 0
 
    main(mbegin, mend) 
# def getnofc(mt):        
#     
#     n = 0
#     for line in file:
#         
#         f = line.rstrip().split(' +++$+++ ')
#         
#         mid = int(f[2][1:])
#         if mid < mt :
#             continue
#         if mid > mt:
#             break
#         
#         n += 1
#     
#     print 'NofC =:\t'+str(n)
