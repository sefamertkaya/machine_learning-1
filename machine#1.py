import numpy as np

def sigmoid(x):                      # SİGMOİD FONKSİYONU TANIMLANDI
    return 1/(1+np.exp(-x))

def sigmoid_der(x):                   # SİGMODİDİN TÜREVİ TANIMLANDI
    return x*(1-x)


class NN:
    def __init__(self,inputs):  # yapıcı fonksiyon olması gerekli
        self.inputs=inputs           
        self.l=len(self.inputs)     #İNPUT sayısı l diye bir değişkene atandı. 
        self.li=len(self.inputs[0]) # İlk input li diye bir değişkene atandı.
       
        self.wi=np.random.random((self.li,self.l))  # input nöronların ağırlıkları random atıldı sanırım
        self.wh=np.random.random((self.l,1))       # buda sinapsların ağırlıkları olabilir
        
    def think(self,inp):
            s1=sigmoid(np.dot(inp,self.wi)) 
            s2=sigmoid(np.dot(s1,self.wh))
            return s2
        
    def train(self,inputs,outputs,it):
            for i in range(it):
                l0=inputs
                l1=sigmoid(np.dot(l0,self.wi))
                l2=sigmoid(np.dot(l1,self.wh))
                
                l2_err=outputs - l2  # BU HATA MİKTARI AĞIRLIKLARIN EĞİTİMİNDE KULLANILICAK
                l2_delta=np.multiply(l2_err,sigmoid_der(l2)) # MULTİPLY NE İŞE YARAR?
                
                l1_err=np.dot(l2_delta,self.wh.T) 
                l1_delta=np.multiply(l1_err,sigmoid_der(l1)) # NEDEN TÜREVLERİNİ PARAMETRE OLARAK YOLLADIK
                
                self.wh += np.dot(l1.T ,l2_delta)
                self.wi += np.dot(l0.T ,l1_delta)
                
                
inputs=np.array([[0.1,0.3],[0.2,0.6],[0.4,0.3]])
outputs=np.array([[0.4],[0.8],[0.7]])
n=NN(inputs)
print("Before Training :")
print (n.think(inputs))
n.train(inputs,outputs,200000)
print("After Training :")
print (n.think([0.6,0.1]))

                
