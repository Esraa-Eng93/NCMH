import pandas as pd
import matplotlib.pyplot as plt #for drawing 
import seaborn as sns
from sklearn.model_selection import train_test_split # AI Start from here
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans #unsupervised learning
from sklearn.preprocessing import StandardScaler

# 1- loding File
df = pd.read_csv("synthetic_students_500_AI_.csv", dtype={"AI_RecommendedAction": "object"})
#because now it type is float64 this from pandas
# print(df.head())

# 2- clean data
#a- fill non exit nale in int col with -1
#print(df.isnull().sum())
num_cols = ['Age', 'AttendanceRate', 'AcademicPerformance', 'CounselingSessions', 'HotlineCalls']
df[num_cols] = df[num_cols].fillna(-1)
#fill non exit nale in text col with unkown
text_cols = ['Name', 'BehaviorIncidents']
df[text_cols] = df[text_cols].fillna("unknown")
#print(df.isnull().sum())

#b-  to insure that i use correct datatype
#print(df.dtypes) #to insure that i use correct datatype

#c- check the repeatness
# print(df[df.duplicated(subset=['StudentID'])])
df.drop_duplicates(subset=['StudentID'], inplace=True)

#d- make sure that the we have appropate data that AI can hadle it
df['GenderNum'] = df['Gender'].map({'M':0, 'F':1})
df['BehaviorIncidentsNum'] = df['BehaviorIncidents'].factorize()[0] # it give num for each catagoary

#E- check anomaly value 
df['AttendanceRate'] = df['AttendanceRate'].clip(0, 100)
df['AcademicPerformance'] = df['AcademicPerformance'].clip(0, 100)
df['CounselingSessions'] = df['CounselingSessions'].clip(0)

#F save all cleaness
#df.to_csv("synthetic_students_500_cleaned.csv", index=False)

#3- analyse the relation between the catagories

'''
print(df.describe())
print(df['Gender'].value_counts())
print(df['BehaviorIncidents'].value_counts())

df['AttendanceRate'].hist(bins=20)
plt.title('Distribution of Attendance Rate')
plt.show()

bins = range(0, 101, 5) 
attendance_groups = pd.cut(df['AttendanceRate'], bins=bins, right=False)
counts = attendance_groups.value_counts().sort_index() 
print(counts)

df['AcademicPerformance'].hist(bins=20)
plt.title('Distribution of Academic Performance')
plt.show()

sns.scatterplot(x='AttendanceRate', y='AcademicPerformance', data=df)
plt.show()

sns.scatterplot(x='AcademicPerformance', y='BehaviorIncidents', data=df)
plt.show()
'''
#4-AI MODEl
'''
def assign_action(row):
    if row['AttendanceRate'] < 60 or row['AcademicPerformance'] < 50 or row['BehaviorIncidents'] == "عنف":
        return "Immediate Intervention"
    elif row['CounselingSessions'] > 3:
        return "Counseling Referral"
    elif row['AttendanceRate'] >= 90 and row['AcademicPerformance'] >= 75 and row['BehaviorIncidents'] == "سلوك طبيعي":
        return "Regular Monitoring"
    else:
        return "Teacher Monitoring + Awareness Materials"

df['AI_RecommendedAction'] = df.apply(assign_action, axis=1)

features = ['Age', 'GenderNum', 'AttendanceRate', 'AcademicPerformance',
            'CounselingSessions', 'BehaviorIncidentsNum', 'HotlineCalls']
X = df[features]
y = df['AI_RecommendedAction']
#training modle
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
#prinnting secduale
#df.to_csv("synthetic_students_500_predicted.csv", index=False)
'''

from sklearn.cluster import KMeans
import pandas as pd

# بيانات الطلاب بعد تنظيفها
features = ['Age', 'AttendanceRate', 'AcademicPerformance', 'CounselingSessions', 'HotlineCalls', 'GenderNum', 'BehaviorIncidentsNum']
X = df[features]


# قياس الخصائص لتكون على نفس المقياس
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# تقسيم الطلاب إلى 3 مجموعات
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(df[['Name', 'Cluster']].head())
# 4- تحديد التوصية لكل طالب بناءً على متوسط المجموعة
for cluster in df['Cluster'].unique():
    subset = df[df['Cluster'] == cluster]
    if subset['AttendanceRate'].mean() < 60 or subset['AcademicPerformance'].mean() < 50:
        action = "Immediate Intervention"
    elif subset['CounselingSessions'].mean() > 3:
        action = "Counseling Referral"
    else:
        action = "Teacher Monitoring + Awareness Materials"
    
    # ضع التوصية مباشرة لكل طالب في هذه المجموعة
    df.loc[df['Cluster'] == cluster, 'AI_RecommendedAction'] = action

# 5- تحقق من النتائج
print(df[['Name', 'Cluster', 'AI_RecommendedAction']].head())

# 6- حفظ الملف النهائي
df.to_csv("synthetic_students_500_ai_predictions.csv", index=False)