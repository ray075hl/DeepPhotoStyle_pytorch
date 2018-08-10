import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy
import cv2

# img format: lab            for example  al bl
# guide img format: lab
# height x width x channel,


def wls_filter_test(img, guide, alpha=1.2, Lambda=0.032):
    epsilon = 0.0001
    height = guide.shape[0]
    width = guide.shape[1]
    n = height * width
    grayImgF_ = 1.0*cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)/255.0
    grayImgF = grayImgF_
    gradWeightX = np.zeros_like(grayImgF, dtype='float')
    gradWeightY = np.zeros_like(grayImgF, dtype='float')

    for y in range(height-1):
        for x in range(width-1):
            if x + 1 < width:
                gx = grayImgF[y, x + 1] - grayImgF[y, x]
                gradWeightX[y, x] = Lambda / (np.power(np.abs(gx), alpha) + epsilon)
            if y + 1 < height:
                gy = grayImgF[y + 1, x] - grayImgF[y, x]
                gradWeightY[y, x] = Lambda / (np.power(np.abs(gy), alpha) + epsilon)

    cv2.imwrite('grad_x.png', (gradWeightX*255.0).astype('uint8'))

    A = sparse.lil_matrix((n, n))

    bs0 = np.zeros(n, dtype='float')
    bs1 = np.zeros(n, dtype='float')
    bs2 = np.zeros(n, dtype='float')

    xs0 = np.zeros(n, dtype='float')
    xs1 = np.zeros(n, dtype='float')
    xs2 = np.zeros(n, dtype='float')

    for y in range(height):
        for x in range(width):
            a = np.zeros(5, dtype='float')

            ii = y * width + x
            if y - 1 >= 0:
                gyw = gradWeightY[y-1, x]
                a[2] += 1.0 * gyw
                a[0] -= 1.0 * gyw
                A[ii, ii- width] = a[0]

            if x - 1 >= 0:
                gxw = gradWeightX[y, x-1]
                a[2] += 1.0 * gxw
                a[1] -= 1.0 * gxw
                A[ii, ii - 1] = a[1]

            if x + 1 < width:
                gxw = gradWeightX[y, x]
                a[2] += 1.0 * gxw
                a[3] -= 1.0 * gxw
                A[ii, ii + 1] = a[3]

            if y + 1 < height:
                gyw = gradWeightY[y, x]
                a[2] += 1.0 * gyw
                a[4] -= 1.0 * gyw
                A[ii, ii + width] = a[4]

            a[2] += 1.0
            A[ii, ii] = a[2]

            r, g, b = img[y, x, :]

            xs0[ii] = 0.0
            xs1[ii] = 0.0
            xs2[ii] = 0.0

            bs0[ii] = float(r)/255
            bs1[ii] = float(g)/255
            bs2[ii] = float(b)/255

    xs0 = np.clip(spsolve(A, bs0), 0, 255)
    xs1 = np.clip(spsolve(A, bs1), 0, 255)
    xs2 = np.clip(spsolve(A, bs2), 0, 255)

    c0 = np.reshape(xs0, (height, width))
    c1 = np.reshape(xs1, (height, width))
    c2 = np.reshape(xs2, (height, width))

    print(np.max(cv2.merge((c0, c1, c2))))
    result = (cv2.merge((c0, c1, c2))*255).astype('uint8')
    return result
    #cv2.imwrite('result.png', result)
    #return cv2.merge((c0, c1, c2))/255.0


def each_channel(img_channel, guide_channel, alpha=1.2, Lambda=1.0):
    epsilon = 0.0001
    grayImgF = guide_channel
    height = grayImgF.shape[0]
    width = grayImgF.shape[1]
    n = height * width
    gradWeightX = np.zeros_like(grayImgF, dtype='float')
    gradWeightY = np.zeros_like(grayImgF, dtype='float')

    for y in range(height - 1):
        for x in range(width - 1):
            if x + 1 < width:
                gx = grayImgF[y, x + 1] - grayImgF[y, x]
                gradWeightX[y, x] = Lambda / (np.power(np.abs(gx), alpha) + epsilon)
            if y + 1 < height:
                gy = grayImgF[y + 1, x] - grayImgF[y, x]
                gradWeightY[y, x] = Lambda / (np.power(np.abs(gy), alpha) + epsilon)

    cv2.imwrite('grad_x.png', (gradWeightX * 255.0).astype('uint8'))

    A = sparse.lil_matrix((n, n))

    bs0 = np.zeros(n, dtype='float')
    bs1 = np.zeros(n, dtype='float')
    bs2 = np.zeros(n, dtype='float')

    xs0 = np.zeros(n, dtype='float')
    xs1 = np.zeros(n, dtype='float')
    xs2 = np.zeros(n, dtype='float')

    for y in range(height):
        for x in range(width):
            a = np.zeros(5, dtype='float')

            ii = y * width + x
            if y - 1 >= 0:
                gyw = gradWeightY[y - 1, x]
                a[2] += 1.0 * gyw
                a[0] -= 1.0 * gyw
                A[ii, ii - width] = a[0]

            if x - 1 >= 0:
                gxw = gradWeightX[y, x - 1]
                a[2] += 1.0 * gxw
                a[1] -= 1.0 * gxw
                A[ii, ii - 1] = a[1]

            if x + 1 < width:
                gxw = gradWeightX[y, x]
                a[2] += 1.0 * gxw
                a[3] -= 1.0 * gxw
                A[ii, ii + 1] = a[3]

            if y + 1 < height:
                gyw = gradWeightY[y, x]
                a[2] += 1.0 * gyw
                a[4] -= 1.0 * gyw
                A[ii, ii + width] = a[4]

            a[2] += 1.0
            A[ii, ii] = a[2]

            r = img_channel[y, x]

            xs0[ii] = 0.0

            bs0[ii] = float(r)

    xs0 = spsolve(A, bs0)
    c0 = np.reshape(xs0, (height, width))

    return c0

# guide lab  0~1
def wls_filter(img, guide, alpha=1.2, Lambda=0.032):


    grayImgF_ = guide#1.0*cv2.cvtColor(guide, cv2.COLOR_RGB2LAB)/255.0
    l, a, b = cv2.split(grayImgF_)


    c0 = each_channel(img[:, :, 0], l)
    c1 = each_channel(img[:, :, 1], l)
    c2 = each_channel(img[:, :, 2], l)

    print(np.max(cv2.merge((c0, c1, c2))))
    #result = (cv2.merge((c0, c1, c2))).astype('uint8')
    #cv2.imwrite('result.png', result)
    return cv2.merge((c0, c1, c2))

if __name__ == '__main__':
    img1 = 1.0 * cv2.imread('final_result7001400.png')/255.0
    img2 = 1.0 * cv2.imread('v2_2272.jpeg')/255.0

    result = wls_filter(img1, img2, 1.2, 1) + img2 - wls_filter(img2, img2, 1.2, 1)
    result = np.clip(result, 0, 1)
    diff = img1 - result
    cv2.imwrite('result2.png', (result*255.0).astype('uint8'))
    cv2.imwrite('diff.png', (diff).astype('uint8'))


