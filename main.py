from face_detect import detect_face
import matplotlib.pyplot as plt
import glob, cv2

'''
for i in glob.glob("John/*.JPG"):
    A = detect_face(i)
    plt.imshow(A.reshape((100,100)))
    plt.show()
'''
X = []
Y = []
count = 0
target_names = ["Jack", "John"]
for j in target_names:
    for i in glob.glob("%s/*.JPG" % j):
        img = cv2.imread(i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append(gray.reshape((10000,)))
        Y.append(count)
    count += 1


from sklearn.cross_validation import train_test_split
from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.25)
(n_classes, w,h) = (2, 100,100)

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 50

print "Extracting the top %d eigenfaces from %d faces" % (n_components, len(X_train))
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print "done in %0.3fs" % (time() - t0)


eigenfaces = pca.components_.reshape((n_components, h, w))
print "Projecting the input data on the eigenfaces orthonormal basis"
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print "done in %0.3fs" % (time() - t0)
print "\n====================\n"

###############################################################################
# Train a SVM classification model

print "Fitting the classifier to the training set"
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
clf = clf.fit(X_train_pca, Y_train)
print "done in %0.3fs" % (time() - t0)
print "Best estimator found by grid search:"
print clf.best_estimator_
print "\n====================\n"


###############################################################################
# Quantitative evaluation of the model quality on the test set

print "Predicting people's names on the test set"
t0 = time()
Y_pred = clf.predict(X_test_pca)
print "done in %0.3fs" % (time() - t0) 

print classification_report(Y_test, Y_pred, target_names=target_names)
print confusion_matrix(Y_test, Y_pred, labels=range(n_classes))



###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=6):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(Y_pred, Y_test, target_names, i)
                     for i in range(Y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
