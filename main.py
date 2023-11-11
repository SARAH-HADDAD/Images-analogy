import numpy as np
import cv2
from sklearn.feature_extraction.image import extract_patches_2d as extract
from sklearn.neighbors import NearestNeighbors
import pyflann as pyflann

"""
To get pyflann to work I had to copy the x64 directory from Lib>site-packages>pyflann>lib>win32 
into envs>python>Lib>site-packages>pyflann>win32. If you have difficulties just use sklearn, and remove references
to pyflann.  
"""


def read_images(apath, appath, bpath):
    imgA = cv2.imread(apath, cv2.IMREAD_UNCHANGED)/255.0
    imgAp = cv2.imread(appath, cv2.IMREAD_UNCHANGED)/255.0
    imgB = cv2.imread(bpath, cv2.IMREAD_UNCHANGED)/255.0

    return imgA, imgAp, imgB


def add_pairs(imgA, imgAp, path1, path2):
    imgA2 = cv2.imread(path1, cv2.IMREAD_UNCHANGED) / 255.0
    imgAp2 = cv2.imread(path2, cv2.IMREAD_UNCHANGED) / 255.0
    imgA2 = cv2.resize(imgA2,(imgA.shape[1],imgAp.shape[0]))
    imgAp2 = cv2.resize(imgAp2, (imgAp.shape[1],imgAp.shape[0]))
    matA = cv2.hconcat([imgA,imgA2])
    matAp = cv2.hconcat([imgAp,imgAp2])

    return matA, matAp


def remap_y(imgA,imgB):
    meanA = np.mean(imgA)
    sdA = np.std(imgA)
    meanB = np.mean(imgB)
    sdB = np.std(imgB)

    imgA_remapped = sdB/sdA*(imgA-meanA)+meanB

    return imgA_remapped


def rgb2yiq(image, remap=False, remap_target=None, feature='y'):
    yiq_xform = np.array([[0.299, 0.587, 0.114],
                          [0.596, -0.275, -0.321],
                         [0.212, -0.523, 0.311]])
    yiq = np.dot(image, yiq_xform.T.copy())

    if remap:
        remap_y(image, remap_target)
    if feature == 'y':
        return yiq[:,:,0]
    elif feature == 'yiq':
        return yiq


def yiq2rgb(image):
    rgb_xform = np.array([[1., 0.956, 0.619],
                          [1., -0.272, -0.647],
                          [1., -1.106, 1.703]])
    rgb = np.dot(image, rgb_xform.T.copy())

    return rgb


def get_pyramid(image, levels):
    img = image.copy()
    pyr = [img]
    for i in range(levels):
        img = cv2.pyrDown(img)
        pyr.append(img)

    return pyr


def get_features(img, causal=False):

    #create 5x5 neighborhood for L, pad so that feature list is correct dimensions
    patches = cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_DEFAULT)
    patches = extract(patches, (5, 5))
    if causal:
        features = np.zeros((img.shape[0],img.shape[1],12))
    else:
        features = np.zeros((img.shape[0],img.shape[1],25))

    height, width = img.shape  # dimensions of the current level of the gaussian pyramid
    for i in range(height):
        for j in range(width):
                features[i, j, :] = patches[i * width + j].flatten()[0:features.shape[2]]

    return features


def make_analogy(lvl, Nlvl, A_L, Ap_L, B_L, Bp_L, s_L, kappa=0, method='pyflann'):
    A_f = get_features(A_L[lvl])
    Ap_f = get_features(Ap_L[lvl][:,:,0], causal=True)
    A_f = np.concatenate((A_f, Ap_f), 2)

    # initialize additional feature sets and B mats
    if lvl < Nlvl:
        Ad_f = cv2.resize(A_L[lvl+1], (A_L[lvl].shape[1],A_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
        Ad_f = get_features(Ad_f)
        Apd_f = cv2.resize(Ap_L[lvl + 1][:,:,0], (Ap_L[lvl].shape[1], Ap_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
        Apd_f = get_features(Apd_f)
        A_f = np.concatenate((A_f, Ad_f, Apd_f), 2)
        B1 = cv2.resize(B_L[lvl + 1], dsize=(B_L[lvl].shape[1], B_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
        Bp1 = cv2.resize(Bp_L[lvl + 1], dsize=(Bp_L[lvl].shape[1], Bp_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
        B1_border = cv2.copyMakeBorder(B1, 2, 2, 2, 2, cv2.BORDER_DEFAULT)
        Bp1_border = cv2.copyMakeBorder(Bp1, 2, 2, 2, 2, cv2.BORDER_DEFAULT)

    # initialize mat by taking previous pyramid level and resize it to the same shape as the current level
    # for lvl=Nlvl you can initialize it with current Ap, B or with some randomization function. You can
    # get some really interesting results from changing the source for the first B'
    if lvl < Nlvl:
        Bp_L[lvl] = cv2.resize(Bp_L[lvl+1], dsize=(Bp_L[lvl].shape[1], Bp_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        Bp_L[lvl] = cv2.resize(B_L[lvl], dsize=(Bp_L[lvl].shape[1], Bp_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)

    # resolve border issue by padding 2 pixels for 5x5 neighborhood
    B_border = cv2.copyMakeBorder(B_L[lvl],2,2,2,2,cv2.BORDER_DEFAULT)
    Bp_border = cv2.copyMakeBorder(Bp_L[lvl],2,2,2,2, cv2.BORDER_DEFAULT)

    # put feature list into index format M*N,numFeatures (25+12)
    A_f_2d = np.reshape(A_f, (A_f.shape[0]*A_f.shape[1], A_f.shape[2]))

    """  Begin Neighbor Search Methods  """
    if method == 'pyflann_kmeans':
        flann = pyflann.FLANN()

        print("Building FLANN kmeans index for size:", A_f.size, "for A size", Ap_L[lvl].size)
        flann_p = flann.build_index(A_f_2d, algorithm="kmeans", branching=32, iterations=-1, checks=16)
        print("FLANN kmeans index done...")

    elif method == 'pyflann_kdtree':
        flann = pyflann.FLANN()

        print("Building FLANN kdtree index for size:", A_f.size, "for A size", Ap_L[lvl].size)
        flann_p = flann.build_index(A_f_2d, algorithm="kdtree")
        print("FLANN kdtree index done...")

    elif method == 'sk_nn':
        print("Building Scikit Nearest Neighbors index for size:", A_f.size, "for A size", Ap_L[lvl].size)
        sknn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(A_f_2d)
        print("NN index done...")
    """  End Neighbor Search Methods  """

    coh_chosen = 0

    for x in range(2, B_border.shape[0]-2):
        if x%25 == 0:
            print("Rastering row", x, "of", B_border.shape[0]-4)

        for y in range(2, B_border.shape[1]-2):
            # this is where you really are in B
            bx, by = x-2, y-2

            B_patch = B_border[x-2:x+3,y-2:y+3,0].flatten()
            Bp_causal = Bp_border[x-2:x+1,y-2:y+3,0].flatten()[0:12]
            B_f = np.concatenate((B_patch, Bp_causal))

            if lvl < Nlvl:  # get same set features as A_F
                B1_patch = B1_border[x-2:x+3,y-2:y+3,0].flatten()
                Bp1_patch = Bp1_border[x-2:x+3,y-2:y+3,0].flatten()
                B_f = np.concatenate((B_f, B1_patch, Bp1_patch),0)

            if method == 'sk_nn':
                distance, neighbor = sknn.kneighbors(B_f[None, :])
                neighbor = int(neighbor[0])
            else:
                neighbor, distance = flann.nn_index(B_f, 1, checks=flann_p['checks'])
            distance = distance**2
            # get p. turn number in neighbor to coordinate in A_f
            m,n = np.unravel_index(neighbor, (A_f.shape[0], A_f.shape[1]))
            if kappa > 0:
                coh_neighbor, coh_distance = get_coherent(A_f, B_f, bx, by, s_L[lvl])
                # coh_fact is squared to get it closer to the performance as described in Hertzmann paper
                coh_fact = (1.0 + 2.0 ** (lvl - Nlvl) * kappa)**2
                if coh_distance <= distance*coh_fact:
                    m,n = coh_neighbor
                    coh_chosen += 1

            Bp_L[lvl][bx,by,0] = Ap_L[lvl][m,n,0]
            # save s
            s_L[lvl][bx, by, 0] = m
            s_L[lvl][bx, by, 1] = n

    print("coherent pixel chosen", coh_chosen, "times.")

    return Bp_L[lvl]


def make_analogy_color(lvl, Nlvl, A_L, Ap_L, B_L, Bp_L, s_L, kappa=0, method='pyflann'):
    A_f = get_features(rgb2yiq(A_L[lvl], remap=True, remap_target=B_L[lvl]))
    Ap_f = get_features(rgb2yiq(Ap_L[lvl], remap=True, remap_target=B_L[lvl]), causal=True)
    A_f = np.concatenate((A_f, Ap_f), 2)

    # initialize additional feature sets and B mats
    if lvl < Nlvl:
        Ad_f = cv2.resize(A_L[lvl+1], (A_L[lvl].shape[1],A_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
        Ad_f = get_features(rgb2yiq(Ad_f, remap=True, remap_target=B_L[lvl+1]))
        Apd_f = cv2.resize(Ap_L[lvl + 1], (Ap_L[lvl].shape[1], Ap_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
        Apd_f = get_features(rgb2yiq(Apd_f, remap=True, remap_target=B_L[lvl+1]))
        A_f = np.concatenate((A_f, Ad_f, Apd_f), 2)
        B1 = cv2.resize(B_L[lvl + 1], dsize=(B_L[lvl].shape[1], B_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
        Bp1 = cv2.resize(Bp_L[lvl + 1], dsize=(Bp_L[lvl].shape[1], Bp_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
        B1_border = cv2.copyMakeBorder(B1, 2, 2, 2, 2, cv2.BORDER_DEFAULT)
        Bp1_border = cv2.copyMakeBorder(Bp1, 2, 2, 2, 2, cv2.BORDER_DEFAULT)

    # initialize mat by taking previous pyramid level and resize it to the same shape as the current level
    # for lvl=Nlvl you can initialize it with current Ap or with some randomization function
    if lvl < Nlvl:
        Bp_L[lvl] = cv2.resize(Bp_L[lvl+1], dsize=(Bp_L[lvl].shape[1], Bp_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        Bp_L[lvl] = cv2.resize(B_L[lvl], dsize=(Bp_L[lvl].shape[1], Bp_L[lvl].shape[0]), interpolation=cv2.INTER_CUBIC)

    # resolve border issue by padding 2 pixels for 5x5 neighborhood
    B_border = cv2.copyMakeBorder(B_L[lvl],2,2,2,2,cv2.BORDER_DEFAULT)
    Bp_border = cv2.copyMakeBorder(Bp_L[lvl],2,2,2,2, cv2.BORDER_DEFAULT)

    # put feature list into index format M*N,numFeatures (25+12)
    A_f_2d = np.reshape(A_f, (A_f.shape[0]*A_f.shape[1], A_f.shape[2]))

    """  Begin Neighbor Search Methods  """
    if method == 'pyflann_kmeans':
        flann = pyflann.FLANN()

        print("Building FLANN kmeans index for size:", A_f.size, "for A size", Ap_L[lvl].size)
        flann_p = flann.build_index(A_f_2d, algorithm="kmeans", branching=32, iterations=-1, checks=16)
        print("FLANN kmeans index done...")

    elif method == 'pyflann_kdtree':
        flann = pyflann.FLANN()

        print("Building FLANN kdtree index for size:", A_f.size, "for A size", Ap_L[lvl].size)
        flann_p = flann.build_index(A_f_2d, algorithm="kdtree")
        print("FLANN kdtree index done...")

    elif method == 'sk_nn':
        print("Building Scikit Nearest Neighbors index for size:", A_f.size, "for A size", Ap_L[lvl].size)
        sknn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(A_f_2d)
        print("NN index done...")
    """  End Neighbor Search Methods  """

    coh_chosen = 0

    for x in range(2, B_border.shape[0]-2):
        #Bp_int = np.uint8(Bp_L[lvl][:,:,0].copy()*255)
        #cv2.imshow("bp", Bp_int)
        #cv2.waitKey(1)
        if x%25 == 0:
            print("Rastering row", x, "of", B_border.shape[0]-4)

        for y in range(2, B_border.shape[1]-2):
            # this is where you really are in B
            bx = x-2
            by = y-2

            B_patch = rgb2yiq(B_border[x-2:x+3,y-2:y+3]).flatten()
            Bp_causal = rgb2yiq(Bp_border[x-2:x+1,y-2:y+3]).flatten()[0:12]
            B_f = np.concatenate((B_patch, Bp_causal))

            if lvl < Nlvl:  # get same set features as A_F
                B1_patch = rgb2yiq(B1_border[x-2:x+3,y-2:y+3]).flatten()
                Bp1_patch = rgb2yiq(Bp1_border[x-2:x+3,y-2:y+3]).flatten()
                B_f = np.concatenate((B_f, B1_patch, Bp1_patch),0)

            if method == 'sk_nn':
                distance, neighbor = sknn.kneighbors(B_f[None, :])
                neighbor = int(neighbor[0])
            else:
                neighbor, distance = flann.nn_index(B_f, 1, checks=flann_p['checks'])
            distance = distance**2
            # get p
            # turn number in neighbor to coordinate in A_f
            m,n = np.unravel_index(neighbor, (A_f.shape[0], A_f.shape[1]))
            if kappa > 0:  # kappa > 0
                coh_neighbor, coh_distance = get_coherent(A_f, B_f, bx, by, s_L[lvl])
                # coh_fact is squared to get it closer to the performance as described in Hertzmann paper
                coh_fact = (1.0 + 2.0 ** (lvl - Nlvl) * kappa)**2
                if coh_distance <= distance*coh_fact:
                    m,n = coh_neighbor
                    coh_chosen += 1


            Bp_L[lvl][bx,by,:] = Ap_L[lvl][m,n,:]  # move value into Bprime

            # save s
            s_L[lvl][bx, by, 0] = m
            s_L[lvl][bx, by, 1] = n


    print("Coherent pixel chosen", coh_chosen, "/", Bp_L[lvl].size, "times.")

    return Bp_L[lvl]


def get_coherent(A_f,B_f,x,y,s):  # tuned for 5x5 patches only
    min_distance = np.inf
    cohxy = [-1, -1]
    for i in range(-2, 3, 1):
        for j in range(-2, 3, 1):
            if i == 0 and j == 0:  # only do causal portion
                break
            if x+i >= s.shape[0] or y+j >= s.shape[1]:
                continue
            sx,sy = int(s[x+i,y+j,0]),int(s[x+i,y+j,1])
            if sx == -1 or sy == -1:
                continue
            rx, ry = sx-i, sy-j
            if rx < 0 or rx >= A_f.shape[0] or ry < 0 or ry >= A_f.shape[1]:
                continue
            rstar = np.sum((A_f[rx,ry,:]-B_f)**2)
            if rstar < min_distance:
                min_distance = rstar
                cohxy = rx, ry

    return cohxy, min_distance
"""
Frame work for looping through make_analogy
Parameters:
A.jpg, Ap.jpg, B.jpg : Inputs into the framework. Will attempt to create novel image Bp.
remap_A : specifies whether an attempt should be made to normalize the luminance of A and Ap to B. Can help with
            artifacting with images of noticeably different luminance averages.
pyr_levels : number of Gaussian pyramids to iterate along. Increases expression of low frequency features.
kappa : coherence weight, refer to Hertzmann paper for some suggested values
search_method : each works well. Three are implemented: pyflann_kmeans, pyflann_kdtree, and sk_nn. kmeans iterations 
                is set to reach convergence. If you want to speed up kmeans you may want to change it to 7-10. kdtree
                is probably the fastest, sk_nn is the slowest.
type : two types are implemented. The default is luminance since the paper predominantly did its work only in the
        luminance domain. The color setting lets you apply the color from textures and colorize grayscale features.
                 
"""


"""  Start User Defined Inputs  """
remap_A = 1
pyr_levels = 5
kappa = 0
# method can be pyflann_kmeans, pyflann_kdtree or sk_nn, default:pyflann_kmeans
search_method = 'pyflann_kdtree'
# type can be luminance or color, most effects work with luminance. Texture synthesis works well with color
# default:luminance
type = 'luminance'
additional_pairs = False
# type the path of your images here. You can change the output path at the bottom
imgA, imgAp, imgB = read_images("src3/A.jpg", "src3/Ap.jpg", "src3/B.jpg")
if additional_pairs:  # add/remove a line for each pair you want to cat to A and Ap for training on a single loop
    imgA, imgAp = add_pairs(imgA, imgAp, "src3/A2.jpg", "src3/Ap2.jpg")
    imgA, imgAp = add_pairs(imgA, imgAp, "src3/A3.jpg", "src3/Ap3.jpg")
"""  End User Defined Inputs  """

"""
Compute Gaussian pyramids for A, A', and B
"""
# initialize pyramids (denoted by _L)
if type == 'luminance':
    A_L = get_pyramid(rgb2yiq(imgA), pyr_levels)
    B_L = get_pyramid(rgb2yiq(imgB, feature='yiq'), pyr_levels)
    Ap_L = get_pyramid(rgb2yiq(imgAp, feature='yiq'), pyr_levels)
elif type == 'color':
    A_L = get_pyramid(imgA, pyr_levels)
    B_L = get_pyramid(imgB, pyr_levels)
    Ap_L = get_pyramid(imgAp, pyr_levels)

Bp_L = []
s = []
for i in range(len(B_L)):
    Bp_L.append(np.zeros(B_L[i].shape))
    s.append(np.zeros((B_L[i].shape[0],B_L[i].shape[1],2))-1)

# process pyramid from coursest to finest
for lvl in range(pyr_levels, -1, -1):
    print("Starting Level: ", lvl, "of ", pyr_levels)

    if type == 'luminance':
        Bp_L[lvl] = make_analogy(lvl, pyr_levels, A_L, Ap_L, B_L, Bp_L, s, kappa, search_method)
        Bp_int = np.uint8(Bp_L[lvl][:, :, 0].copy() * 255)
        cv2.imshow("bp", Bp_int)
        cv2.waitKey(1)
        imgBp = yiq2rgb(Bp_L[lvl])

    elif type == 'color':
        Bp_L[lvl] = make_analogy_color(lvl, pyr_levels, A_L, Ap_L, B_L, Bp_L, s, kappa, search_method)
        Bp_int = np.uint8(Bp_L[lvl].copy() * 255)
        cv2.imshow("bp", Bp_int)
        cv2.waitKey(1)
        imgBp = Bp_L[lvl]

    imgBp = imgBp * 255.0
    imgBp[imgBp > 255] = 255
    imgBp[imgBp < 0] = 0

    write_name = 'out/Bp_PyrLvl-'+str(lvl)+'.jpg'
    cv2.imwrite(write_name, imgBp)
cv2.imwrite("output/output_image.jpg", imgBp)