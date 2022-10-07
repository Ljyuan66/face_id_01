import cv2
import sys

from PIL import Image


# 定义函数Capture_picture；目的：获取人脸数据集；参数：窗口名，相机编号，收集图片的数量，保存图片的路径
def Capture_picture(window_name, camera_id, picture_num, path_name):
    cv2.namedWindow(window_name)

    cap = cv2.VideoCapture(camera_id)  # camera_id = 0为内置摄像头，= 1为外接摄像头，也可以为本地视频的路径

    # opencv内置的级联分类器，找到anaconda的安装位置往下找就可以找到
    classfier = cv2.CascadeClassifier("D/home/liang/.conda/envs/stuenv/lib/python3.6/site-packages/opencv_contrib_python-3.4.1.15.dist-info")

    color = (0, 255, 0)  # 边框颜色

    num = 0

    while cap.isOpened():
        ok, frame = cap.read()  # cap.read()返回两个值，是否正确读取帧和当前帧的图片
        # print(ok)
        # print(frame)

        if not ok:
            break

        # 将图片转成灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # img_name = 'D:/desktop/face/gray/%d.jpg'%(num)
        # cv2.imwrite(img_name,gray)

        faceRects = classfier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(32, 32))
        # print(faceRects)

        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                img_name = 'D:/desktop/face/huangzhaohong/%d.jpg' % (num)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]  # 将人脸框相应的扩大
                cv2.imwrite(img_name, image)

                num += 1
                if num > (picture_num):
                    break

                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                # 图片，标注文字，左上角坐标，字体形式，字体大小，颜色，字体粗细
                cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 0), 4)

        if num > picture_num: break

        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)  # 这里参数为0等于视频暂停
        if c & 0xFF == ord(' '): break  # 手动结束视频按空格

    # 关闭摄像头，释放窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("!!!")
    else:
        Capture_picture("face", 0, 1000, "D:/desktop/face/huangzhaohong")

