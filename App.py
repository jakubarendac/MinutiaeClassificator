from MinutiaeNetWrapper import MinutiaeNetWrapper

def main():
    minutiaeNet = MinutiaeNetWrapper()
    minutiaeNet.readImage("./ClassifyNet/testData/img_files/crd_0208s_01.png")
    minutiaeNet.predictImage()  
  
if __name__== "__main__":
    main()