import re
import time

import geonamescache
import numpy as np
import pandas as pd
from unidecode import unidecode

gc = geonamescache.GeonamesCache()
countries_info = gc.get_countries()
us_states_info = gc.get_us_states()
us_counties_info = gc.get_us_counties()
cities_info = gc.get_cities()


# 功能：获取列表中最长字符串
def longest_str(original_list):
    if original_list:
        result = original_list[0]
        for i in original_list:
            if len(i) > len(result):
                result = i
        return result
    else:
        return None


# 功能：从新闻标题中找国家名称
# 注意：传入的是已经unidecode了的标题和已经unidecode了的正式城市名，返回值也unidecode了；该函数在调用match_city_country后调用
def match_country_from_headline(headline, original_country_name):
    matched_country = []
    for country_entry_value in countries_info.values():
        country_name = unidecode(country_entry_value['name']).strip()
        compiled_normalized_name = re.compile(r'\b' + re.escape(country_name) + r'\b')
        country_name_matches = compiled_normalized_name.findall(headline)
        if country_name_matches:
            matched_country.append(country_name_matches[0])
    if matched_country:
        return longest_str(matched_country)
    else:
        return original_country_name


counter_match_city = 0


# 功能：根据标题匹配城市和初步匹配国家
# 注意：传入的是已经unidecode了的标题，返回值也unidecode了；该函数要先于match_country_from_headline函数调用
def match_city_country(headline):
    global counter_match_city
    counter_match_city += 1
    print('begin to match headline {}: {}'.format(counter_match_city, headline))

    matched_city = {}
    for city_ID, city_entry_value in cities_info.items():
        city_names = [unidecode(city_entry_value['name']).strip()]
        for alternate in city_entry_value['alternatenames']:
            striped_normalized_alternate = unidecode(alternate).strip()
            if striped_normalized_alternate:
                city_names.append(striped_normalized_alternate)
        for name in city_names:
            # 对unidecode后的字符串中的符号进行转义，并且要匹配边界，匹配到不完整的单词或词组
            compiled_normalized_name = re.compile(r'\b' + re.escape(name) + r'\b')
            city_name_matches = compiled_normalized_name.findall(headline)
            if city_name_matches:
                matched_city[city_ID] = [city_name_matches[0], city_names[0], len(city_entry_value['alternatenames']),
                                         city_entry_value['population']]
    if matched_city:
        # 先找最长匹配项，相同长度的最长匹配项选正式的城市名而非别名，还相同的话选别名多的城市，还相同的话选人口多的城市

        # 找出最长的匹配项长度，并去除其它长度的匹配项
        max_length = 0
        for skimmed_city_info in matched_city.values():
            max_length = max(max_length, len(skimmed_city_info[0]))
        has_formal_name = False
        delete_ID = []
        for city_ID, skimmed_city_info in matched_city.items():
            if len(skimmed_city_info[0]) < max_length:
                delete_ID.append(city_ID)
            else:
                if skimmed_city_info[0] == skimmed_city_info[1]:
                    has_formal_name = True
        for id in delete_ID:
            matched_city.pop(id)
        delete_ID.clear()

        # 如果有正式的名字，删除其它非正式的
        if has_formal_name:
            for city_ID, skimmed_city_info in matched_city.items():
                if skimmed_city_info[0] != skimmed_city_info[1]:
                    delete_ID.append(city_ID)
            for id in delete_ID:
                matched_city.pop(id)
            delete_ID.clear()

        # 删除别名数量少的城市，别名数量可以体现一个城市的文化影响力
        max_num_alternate = 0
        for skimmed_city_info in matched_city.values():
            max_num_alternate = max(max_num_alternate, skimmed_city_info[2])
        for city_ID, skimmed_city_info in matched_city.items():
            if skimmed_city_info[2] < max_num_alternate:
                delete_ID.append(city_ID)
        for id in delete_ID:
            matched_city.pop(id)
        delete_ID.clear()

        # 返回剩下的匹配项中人口数量最多的，人口数量也可体现城市的影响力
        max_population = 0
        for skimmed_city_info in matched_city.values():
            max_population = max(max_population, skimmed_city_info[3])
        for city_ID, skimmed_city_info in matched_city.items():
            if skimmed_city_info[3] == max_population:
                country_name = countries_info[cities_info[city_ID]['countrycode']]['name']
                return skimmed_city_info[1], unidecode(country_name).strip()

    # 如果没有城市被匹配到，可能出现的是美国的州名
    matched_us_state = []
    for us_state_entry_value in us_states_info.values():
        compiled_state_name = re.compile(r'\b' + re.escape(us_state_entry_value['name']) + r'\b')
        state_name_matches = compiled_state_name.findall(headline)
        if state_name_matches:
            matched_us_state.append(state_name_matches[0])
    if matched_us_state:
        return longest_str(matched_us_state), 'United States'

    # 存在较多相同名字的US county，但是没有关系，因为对应的国家都是United States
    matched_us_county = []
    for county_info in us_counties_info:
        compiled_county_name = re.compile(r'\b' + re.escape(county_info['name']) + r'\b')
        county_name_matches = compiled_county_name.findall(headline)
        if county_name_matches:
            matched_us_county.append(county_name_matches[0])
    if matched_us_county:
        return longest_str(matched_us_county), 'United States'
    return None, None


def read_and_extract_headline_data(start, end=None):
    result = np.genfromtxt("database/headlines.txt", 'str', delimiter='\n', encoding='utf-8')
    result = pd.DataFrame(result, columns=['headline'], dtype='str')
    result.drop_duplicates(ignore_index=True, inplace=True)
    if end is None:
        result = result.iloc[start:].reset_index(drop=True)
    else:
        result = result.iloc[start:end].reset_index(drop=True)
    result['headline'] = result['headline'].apply(unidecode)
    num_headline = len(result)

    result['city'], result['country'] = None, None
    for i in range(num_headline):
        result.loc[i, 'city'], result.loc[i, 'country'] = match_city_country(result.loc[i, 'headline'])
        result.loc[i, 'country'] = match_country_from_headline(result.loc[i, 'headline'], result.loc[i, 'country'])
    return result


def main():
    result = read_and_extract_headline_data(0)
    result.to_csv('database/extracted_data.csv', index=False, encoding='utf-8')

    print(result.describe())
    print(result.info())
    print(result[result.isnull().T.any()])


if __name__ == "__main__":
    timestamp1 = time.time()
    main()
    timestamp2 = time.time()
    print("总共用时 %f 秒" % (timestamp2 - timestamp1))
