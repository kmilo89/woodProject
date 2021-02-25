import cv2

def show_image(path):
    img = cv2.imread(path)
    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def main():
    #show_image('Dataset/1.jpg')
    img = cv2.imread(path)

if __name__== "__main__":
    main()
