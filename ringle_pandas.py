import pandas as pd
import numpy as np

obj1 = pd.Series([-5, 0, 10, 3], index=['a', 'b', 'd', 'c'])
print(obj1)
print(obj1['a'])

인구 = {'서울': 9776, '부산':3429, '대전':1531, '세종':276, '충남':2148, '대구':2465}
시리3 = pd.Series(인구)
도시들 = ['대전', '세종', '대구', '충남', '인천']
시리4 = pd.Series(인구, index=도시들)

#null값(NaN) 확인
pd.isnull(시리4)
pd.notnull(시리4)





데이터 = {'지역': ['대전', '대전', '대전', '서울', '서울', '서울'],
      '년도': [2015, 2016, 2017, 2015, 2016, 2017],
      '인구': [1542, 1535, 1531, 9941, 9852, 9776]}
프레임 = pd.DataFrame(데이터)
print(프레임)

print(프레임.head(2))  #가장 윗 값들만 보기 default 5개

pd.DataFrame(데이터, columns=['지역', '인구', '년도']) #원하는 순서로 column 정하기

#자산을 추가하면 NaN으로 추가해줌
프레임2 = pd.DataFrame(데이터, columns=['지역', '인구', '년도', '자산'],
             index=['하나', '둘', '셋', '넷', '다섯', '여섯'])

# 프레임2['지역']   프레임2.지역  2가지 방법으로 데이터 접근가능
#근데 프레임2.지역 으로 새로운 값을 줄 수는 없음

del 프레임['자산']  #가능
#del 프레임.자산  #불가능



###### Re indexing
시 = pd.Series([3, -8, 2, 0], index=['d', 'b', 'a', 'c'])
시.reindex(['a', 'b', 'c', 'd', 'e'])
print (시)
# 행 제거
시.drop('c')   #2개 이상 제거하려면 시.drop(['a', 'c'])

#열 제거
시.drop(['A', 'B'], axis='columns')

#넘파이 범용 함수 사용방법
rng = np.random.RandomState(1234)

데 = pd.DataFrame(rng.randn(3 * 4).reshape(3, 4), columns=list('abcd'), index=list('ㄱㄴㄷ'))
데
np.abs(데)
