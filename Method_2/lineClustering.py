import numpy as np
import cv2
import matplotlib.pyplot as plt
from Method_2.utils import resize
from scipy.signal import find_peaks


def piecewise_linear(x, x0, y0, k1, k2):
    """Define a piecewise, lienar function with two line segments."""
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def getHorizontalProjectionProfileSmoothed(image):
    horizontal_projection = np.sum(image==255, axis = 1) 
    box = np.ones(40)/40
    smooth = np.convolve(horizontal_projection,box,mode='valid')
    return smooth

# comp_img = cv2.imread('./results/fac_03008_arsberattelse_1930_1/components_filtered2.png')
# comp_img = cv2.cvtColor(comp_img,cv2.COLOR_BGR2GRAY)
# hor_prof = getHorizontalProjectionProfileSmoothed(comp_img)
# peaks, _ = find_peaks(hor_prof,height=15,distance=100)
# plt.plot(hor_prof)
# plt.plot(peaks, hor_prof[peaks], 'x')
# plt.show()
# print(peaks)
# i = len(peaks)
# print(i)

def determine_components_n(Ms, lower_bounds, all_means):
    """Determine the optimal number of GMM components based on loss."""

    """
    Explanation:
    
    Lower bounds looks somewhat like a piecewise function with 2 lines
    The changepoint from one line to another tends to be the correct
    number of GMM components!

    This makes sense since any additional components would help the
    error (lower bound) a lot less, leading to a much less steep line

    Then, the goal is to find this changepoint, which we can do by
    fitting a piecewise, 2-line function with scipy.optimize.curve_fit
    For whatever reason, method='trf' works best!
    """

    x = np.array([float(i) for i in range(len(lower_bounds))])
    y = np.array(lower_bounds)

    from scipy import optimize
    p, e = optimize.curve_fit(piecewise_linear, x, y, method='trf')

    plt.xlabel('components')
    plt.ylabel('lower_bounds')
    plt.plot(x, y, 'o')
    x = np.linspace(x.min(), x.max(), 1000)
    plt.plot(x, piecewise_linear(x, *p))
    plt.show()
    # config['save_inter_func'](config, None, "gmm_components", plot=True)

    # p[0] is the changepoint parameter
    return int(np.round(p[0])) + 1
    

def gmm_clustering(cY, components,file,path):
    """Uses GMM models to cluster text lines based on their y values."""
    from sklearn.mixture import GaussianMixture
    # Ms = list(range(1, 50))

    # lower_bounds = []
    # all_means = []
    # for m in Ms:
    #     gmm = GaussianMixture(n_components=m, random_state=0).fit(np.expand_dims(cY, 1))
    #     lower_bounds.append(gmm.lower_bound_)        
    #     means = gmm.means_.squeeze()

    #     # Sort if multiple means, or turn into an array is just one
    #     try:
    #         means.sort()
    #     except:
    #         means = np.array([means])

    #     all_means.append(means)

    # Different methods for selecting the number of components
    # n = determine_components_n(Ms, lower_bounds, all_means)

    # Horizontal projection profile to determine lines
    comp_img = cv2.imread('./results/{}/components_filtered2.png'.format(file))
    comp_img = cv2.cvtColor(comp_img,cv2.COLOR_BGR2GRAY)
    hor_prof = getHorizontalProjectionProfileSmoothed(comp_img)
    peaks, _ = find_peaks(hor_prof,height=16,distance=100)
    plt.rcParams["figure.autolayout"] = True
    plt.plot(hor_prof)
    plt.plot(peaks, hor_prof[peaks], 'x')
    plt.savefig('results/{}/line_histogram.png'.format(file))
    plt.close()
    n = len(peaks)
    # n = no_lines
    if n == 0:
        print('Number of lines set: 0')
        return []
    n = len(peaks)

    # stream = open(u'{}{}.jpg'.format(path,file), "rb")
    # bytes = bytearray(stream.read())
    # numpyarr = np.asarray(bytes,dtype=np.uint8)
    # image = cv2.imdecode(numpyarr, cv2.IMREAD_UNCHANGED)
    # resized_img = resize(image, 15)
    # def nothing(x):
    #     pass
    # cv2.namedWindow('lines')
    # cv2.createTrackbar('no_lines','lines',8,50,nothing)
    # while(1):
    #     cv2.imshow('lines_part',resized_img)
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break 
    # n = cv2.getTrackbarPos('no_lines','lines')

    print('Number of lines set: ', n)

    # Perform analysis with determined number of components n
    gmm = GaussianMixture(n_components=n, random_state=0).fit(np.expand_dims(cY, 1))
    cluster_means = (gmm.means_.squeeze()).astype(np.int32)
    if cluster_means.size > 1:
        cluster_means.sort()
    return cluster_means


def line_clustering(components,file,path):
    """Clusters components into horizontal lines."""

    # Organize and sort component data by y values
    c_area = components.area
    cX = components.x
    cY = components.y
    boundingRect = components.bounding_rect
    sorted_indices = cY.argsort(axis=0)
    c_area = c_area[sorted_indices]
    cX = cX[sorted_indices]
    cY = cY[sorted_indices]
    boundingRect = boundingRect[sorted_indices]
    mean_height = boundingRect[:, 3].mean()

    # Perform GMM analysis to determine lines based on y values
    cluster_means = gmm_clustering(cY, components,file,path)
    # Now that we've found the cluster y values, assign components to each cluster based on y
    component_clusters = np.zeros(len(components))
    component_clusters_min_dist = np.zeros(len(components))

    cluster_i = 0
    line_components = [[]]
    component_clusters = []
    for i in range(len(cY)):
        if cluster_i < cluster_means.size - 1:
            if abs(cY[i] - cluster_means[cluster_i]) > abs(cluster_means[cluster_i + 1] - cY[i]):
                cluster_i += 1
                line_components.append([])
        
        line_components[-1].append(i)
        component_clusters.append(cluster_i)
    
    component_clusters = np.array(component_clusters)

    # Convert the 'sorted y' indices back to the original component indices
    for i, l in enumerate(line_components):
        sorter = np.argsort(sorted_indices)
        line_components[i] = sorter[np.searchsorted(sorted_indices,
                                                    np.array(l), sorter=sorter)]

    # Filter out lines with very little area
    # lines = [i for i, l in enumerate(line_components) if
    #                    components.area[l].sum() >= components.min_area*2]
    # line_components = [l for i, l in enumerate(line_components) if i in lines]
    # cluster_means = [m for i, m in enumerate(cluster_means) if i in lines]

    # Create display image
    keep_components = np.zeros((components.output.shape))
    for c in range(cluster_means.size):
        for i in range(len(components)):
            if component_clusters[i] == c:
                keep_components[components.output == components.allowed[i] + 1] = 255

    if cluster_means.size > 1:
        for i, cc in enumerate(cluster_means):
            cv2.line(keep_components, (0, cluster_means[i]), (
                keep_components.shape[1], cluster_means[i]), 255, 3)
    else:
        cv2.line(keep_components, (0,int(cluster_means)), (keep_components.shape[1],int(cluster_means)), 255, 3)
    # config['save_inter_func'](config, keep_components, "lines")

    cv2.imwrite('results/{}/lines.png'.format(file), keep_components)
    return line_components

