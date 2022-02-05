import time

import numpy as np
from unidecode import unidecode
import geonamescache
import pandas as pd
import requests


def cut_off():
    print('-----------------------------------------------------------------------------------------')


gc = geonamescache.GeonamesCache()
countries_info = gc.get_countries()
cities_info = gc.get_cities()
continents_info = gc.get_continents()
us_states_info = gc.get_us_states()
us_counties_info = gc.get_us_counties()

counter = 0


def run1():
    for city_ID1, city_entry_value1 in cities_info.items():
        for city_ID2, city_entry_value2 in cities_info.items():
            if city_entry_value1['name'] == city_entry_value2['name'] and city_ID1 != city_ID2 and (
                    city_entry_value1['population'] - city_entry_value2['population']) * (
                    len(city_entry_value1['alternatenames']) - len(city_entry_value2['alternatenames'])) > 0:
                print(city_ID1, countries_info[city_entry_value1['countrycode']]['name'], city_entry_value1['name'],
                      city_entry_value1['population'])
                print(city_ID2, countries_info[city_entry_value2['countrycode']]['name'], city_entry_value2['name'],
                      city_entry_value2['population'])


def print_same_name_ones(name):
    for continent_code, continent_entry_value in continents_info.items():
        if unidecode(continent_entry_value['asciiName']).strip() == name:
            print(continent_code, continent_entry_value)
    for city_ID, city_entry_value in cities_info.items():
        if unidecode(city_entry_value['name']).strip() == name:
            print(countries_info[city_entry_value['countrycode']]['name'], city_ID, city_entry_value)
        else:
            for alternate in city_entry_value['alternatenames']:
                if unidecode(alternate).strip() == name:
                    print(countries_info[city_entry_value['countrycode']]['name'], city_ID, city_entry_value)
    for state_code, us_state in us_states_info.items():
        if unidecode(us_state['name']).strip() == name:
            print('United States', state_code, us_state)
    for us_county in us_counties_info:
        if unidecode(us_county['name']).strip() == name:
            print('United States', us_states_info[us_county['state']], us_county)
    for country_ID, countries_entry_value in countries_info.items():
        if unidecode(countries_entry_value['name']).strip() == name:
            print(country_ID, countries_entry_value)


def download_headline_data():
    res = requests.get("https://lv-resources.s3-us-west-2.amazonaws.com/course-data/93/headlines.txt")
    with open("database/headlines.txt", "w", encoding="utf-8") as file:
        file.write(res.text)


def run2():
    data = {
        '辣条': [14, 20],
        '面包': [7, 3],
        '可乐': [8, 13],
        '烤肠': [10, 6]
    }
    df = pd.DataFrame(data, index=['2020-01-01', '2020-01-02'])
    print(df)
    df['2倍'], df['3倍'] = None, None
    print(df)


def run3():
    data = np.array([[2, 4, 5], [7, 9, 10]])
    print(data)
    result = pd.DataFrame(data, columns=['a', 'b', 'c'])
    print(result)


class ProcessReporter:
    def __init__(self):
        self.timestamp1 = time.time()
        self.timestamp2 = time.time()

    def report(self, data):
        self.timestamp2 = time.time()
        print(f'\r{self.timestamp2 - self.timestamp1} seconds passed. Report: {data}', end='')


def skim_city_info():
    result = pd.DataFrame({"name": [], "formal_name": [], "num_alternate_names": [], "population": [], "country": []})
    i = 0
    reporter = ProcessReporter()
    for city_ID, city_entry_value in cities_info.items():
        city_names = [unidecode(city_entry_value['name']).strip()]
        reporter.report([i, city_names[0]])
        for alternate in city_entry_value['alternatenames']:
            striped_normalized_alternate = unidecode(alternate).strip()
            if striped_normalized_alternate:
                city_names.append(striped_normalized_alternate)
        for name in city_names:
            result.loc[i, "name"] = name
            result.loc[i, "formal_name"] = city_names[0]
            result.loc[i, "num_alternate_names"] = len(city_entry_value['alternatenames'])
            result.loc[i, "population"] = city_entry_value['population']
            result.loc[i, 'country'] = unidecode(countries_info[city_entry_value['countrycode']]['name']).strip()
            i += 1
    result.drop_duplicates(inplace=True, ignore_index=True)
    result.to_csv('database/skimmed_city_info.csv', index=False, encoding='utf-8')


def check_ascii():
    result = np.genfromtxt("database/headlines.txt", 'str', delimiter='\n', encoding='utf-8')
    with open("headlines.txt", "w", encoding="utf-8") as file:
        for i in result:
            if not i.isascii():
                file.write(i + "\n")


if __name__ == "__main__":
    # skim_city_info()

    # data = pd.read_csv('database/skimmed_city_info.csv')
    # print(data.shape)
    # print(len(data))
    # print(data)

    data = pd.read_csv('database/extracted_data.csv')
    print(data.describe())
    print(data.info())
    unmatched = data[data.isnull().T.any()]
    print(unmatched)
    unmatched.to_csv("database/unmatched.csv", index=False, encoding='utf-8')

    # check_ascii()

