import geonamescache
import numpy as np
import pandas as pd
from unidecode import unidecode

# 加载数据
df = pd.read_csv('database/extracted_data.csv')

# 基本初始化
gnc = geonamescache.GeonamesCache()
countries_info = gnc.get_countries()
cities_info = gnc.get_cities()
list_longitude = []
list_latitude = []

# 去除空数据行
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)


# 从国家名中获取country code
def get_countrycode_from_name(country_name):
    for code, country_entry_value in countries_info.items():
        if unidecode(country_entry_value['name']).strip() == country_name:
            return code
    return None


df['countrycode'] = df['country'].apply(get_countrycode_from_name)

# 匹配城市的经纬度
city_column = df['city']
countrycode_column = df['countrycode']
for index in df.index:
    print(f'\r{index}', end='')
    city_matches = {}
    for city_ID, city_entry_value in cities_info.items():
        if city_entry_value['countrycode'] == countrycode_column[index]:
            # 先考虑正式名的匹配，匹配不上再考虑别名的匹配
            if unidecode(city_entry_value['name']).strip() == city_column[index]:
                city_matches[city_ID] = [city_entry_value['longitude'], city_entry_value['latitude'],
                                         len(city_entry_value['alternatenames']), city_entry_value['population']]
            else:
                for alternate in city_entry_value['alternatenames']:
                    if unidecode(alternate).strip() == city_column[index]:
                        city_matches[city_ID] = [city_entry_value['longitude'], city_entry_value['latitude'],
                                                 len(city_entry_value['alternatenames']),
                                                 city_entry_value['population']]
    # 对于同一个城市名和国家代码，可能存在多个匹配项，其中取别名数量最多的，别名数量相同的话取人口数量最多的。别名和人口数量能反映城市的文化影响力。
    max_alternate = 0
    for ID, info in city_matches.items():
        max_alternate = max(max_alternate, info[2])
    max_population = 0
    right_longitude = np.nan
    right_latitude = np.nan
    for ID, info in city_matches.items():
        if info[2] == max_alternate:
            if info[3] > max_population:
                max_population = info[3]
                right_longitude = info[0]
                right_latitude = info[1]
    list_longitude.append(right_longitude)
    list_latitude.append(right_latitude)

# 扩增
df['longitude'] = list_longitude
df['latitude'] = list_latitude

# 后期处理
df = df[df.columns[[0, 1, 2, 4, 5, 3]]]  # 调整列的顺序
print(df[df.isnull().T.any()])  # 显示出有空数据的行
df.dropna(inplace=True)  # 删除有空数据的行
df.to_csv('database/match_data.csv', index=False)
print(df.info())
print(df.describe())
