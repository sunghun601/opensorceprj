import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import requests as requests
from bs4 import BeautifulSoup

phone = '글로벌-스마트폰-점유율-분기별-데이터'
url = "https://korea.counterpointresearch.com/" + phone

html = requests.get(url).content
soup = BeautifulSoup(html,'html.parser')

item = []
for i in range(4,10):
    line = soup.select('#post-1016 > div.entry-content > div:nth-child(27) > table:nth-child(4) > tbody > tr:nth-child(%d)' %i)
    item.append(line) # 크롤링 할 line 들을 item에 추가

value = []
csvdata = []
for tag in item:
    for it in tag:
        it = it.find_all('span')
        value.append(it)

for i in value:
    data = []
    for j in i:
        data.append(j.get_text().strip('%')) # 통계를 위해 % 제거
    csvdata.append(data)

df = pd.DataFrame(csvdata)
ndf = df.T

ndf[6] = ['Year','2018Y','2019Y','2019Y','2019Y','2019Y',
               '2020Y','2020Y','2020Y','2020Y','2021Y','2021Y','2021Y']

ndf[7] = ['Quarter','2018 (Q4)','2019 (Q1)','2019 (Q2)','2019 (Q3)','2019 (Q4)',
                    '2020 (Q1)','2020 (Q2)','2020 (Q3)','2020 (Q4)','2021 (Q1)','2021 (Q2)','2021 (Q3)']


ndf.to_excel('smartphoneShare.xlsx')

ndf2 = ndf.rename(columns=ndf.iloc[0])
ndf3 = ndf2.drop(ndf2.index[0])

ndf3['Samsung'] = ndf3['Samsung'].astype(int)
ndf3['Apple'] = ndf3['Apple'].astype(int)
ndf3['Xiaomi'] = ndf3['Xiaomi'].astype(int)
ndf3['vivo'] = ndf3['vivo'].astype(int)
ndf3['OPPO'] = ndf3['OPPO'].astype(int)
ndf3['Others'] = ndf3['Others'].astype(int)


grouped = ndf3.groupby(['Year'])

for key, group in grouped:
    print('* key :', key)
    print('* number : ', len(group))
    print(group)
    print('\n')

average = grouped.mean() # 메소드1. 년도별 평균
print(average)
print('\n')


grouped_two = ndf3.groupby(['Year','Quarter'])

for key, group in grouped_two:
    print('* key :', key)
    print('* number : ', len(group))
    print(group)
    print('\n')

average_two = grouped_two.mean() # 메소드2. 년도와 년도 분기별 평균
print(average_two)
print('\n')


def min_max(x):
    return x.max() - x.min()

agg_minmax = grouped.agg(min_max) # 메소드3. 년도별 점유율 증감률 2018년도는 전년도 데이터가 없으니 0
print(agg_minmax)
print('\n')



Samsung_filter = grouped.filter(lambda x: x.Samsung.mean() >= 20)
print(Samsung_filter) # 메소드4. 삼성 평균 점유율이 20% 이상인 년도 모든 데이터 출력
print('\n')



ndf4 = ndf3
ndf3.drop(labels='Year', axis=1,inplace=True)




plt.style.use('ggplot')

wd = 0.15
nrow = ndf4.shape[0] # 행의 갯수
idx = np.arange(nrow)

plt.figure(figsize = (10, 5))
plt.title('Global Smartphone Shipments Market Share (%)')
plt.xlabel('2018Q4 ~ 2021Q3')
plt.ylabel('Shape (%)')
plt.bar(idx - 3 * wd, ndf4['Samsung'], width = wd, label = 'Samsung')
plt.bar(idx - 2 * wd, ndf4['Apple'], width = wd, label = 'Apple')
plt.bar(idx - wd, ndf4['Xiaomi'], width = wd, label = 'Xiaomi')
plt.bar(idx, ndf4['vivo'], width = wd, label = 'vivo')
plt.bar(idx + wd, ndf4['OPPO'], width = wd, label = 'OPPO')
plt.bar(idx + 2 * wd, ndf4['Others'], width = wd, label = 'Others')
plt.xticks(idx, ndf4['Quarter'], rotation = 30)
plt.legend(ncol = 6)

plt.show()


ratio = average.iloc[1].values
labels = ['Samsung','Apple','Xiaomi','vivo','OPPO','Others']

colors = ['#ff9999', '#ffc000', '#8fd9b6', '#d395d0','red','blue']
wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}


plt.pie(ratio, labels=labels, autopct='%.1f%%', startangle=260,
        counterclock=False, colors=colors, wedgeprops=wedgeprops)

plt.title('Global Smartphone Shipments Market Share (%) - 2019',size=10)
plt.legend(labels = labels, loc = 'upper right')

plt.show()




x=ndf3[['Xiaomi']]
y=ndf3[['Others']]


from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, # 독립 변수
                                                    y, # 종속 변수
                                                    test_size=0.3, # 검증 30%
                                                    random_state=10) # 랜덤 추출 값

from  sklearn.linear_model import LinearRegression

# 단순회귀분석 모형 객체 생성
lr = LinearRegression()
# train data를 가지고 모형 학습
lr.fit(x_train, y_train)

print('기울기 a : ', lr.coef_) # 기울기
print('y절편 b : ', lr.intercept_)

# 모형에 전체 x데이터를 입력하여 예측한 y_hat을 실제 y값과 비교
y_hat = lr.predict(x)

plt.figure(figsize=(10,5))
ax1 = sns.histplot(y,kde=True,label='y',color='red')
ax2 = sns.histplot(y_hat,kde=True, label="y_hat",color='blue' ,ax=ax1)
plt.legend()

# 곡선의 형태가 더 적합

plt.show()










