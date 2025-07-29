→

from sklearn.pipeline import Pipeline

from sklearn.model selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear model import LogisticRegression

from sklearn.ensemble import RandomforestClassifier, Gradient BoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler, OneHot Encoder

X_train, X_test, y_train, y_test train_test_split(x, y, test size=8.2, random_state=42)

models = {

"LogisticRegression": LogisticRegression(),

"RandomForest: RandomForestClassifier(),