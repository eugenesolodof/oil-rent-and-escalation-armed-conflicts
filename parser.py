import requests
from bs4 import BeautifulSoup
import urllib.parse
import pandas as pd
from time import sleep

# list of countries that we wanna to obtain
countries = [
    'Algeria',
    'Angola',
    'Angola-RepublicoftheCongo',
    'Azerbaijan-Turkmenistan',
    'Azerbaijan'
    'Bahrain',
    'Brunei',
    'RepublicoftheCongo',
    'Equatorial Giunea',
    'Gabon',
    'Iran',
    'Iraq',
    'Iran-Iraq',
    'Kuwait-SaudiArabia',
    'Kuwait-SaudiArabia-Iran',
    'Kazakhstan',
    'Kuwait',
    'Libya',
    'Nigeria',
    'Oman',
    'Qatar',
    'SaudiArabia',
    'SaudiArabia-Bahrain',
    'SaudiArabia-Iran',
    'Turkmenistan',
    'TrinidadandTobago',
    'UnitedArabEmirates',
    'UnitedArabEmirates-Iran',
    'Venezuela',
    'Yemen',
    'Russia',
    'Russia-Kazakhstan',
    'Timor-Leste',
    'SouthSudan',
    'Syria',
    'Egypt',
    'Indonesia',
    'Malaysia',
    'Chad',
    'Ecuador'
    ] 

# get_next_page() для получения ссылки на следующую страницу сайта

def get_next_page(soup):
    url = 'https://www.gem.wiki' + soup.find_all('a', {'title':'Category:Oil and gas extraction'})[1]['href']
    return url

# get_link() для получения ссылки на страницу с описанием нефтяного месторождения
def get_link(d):
    url = 'https://www.gem.wiki' + d.find('a')['href']
    encoded_url = urllib.parse.quote(url, safe=':/%&')
    return encoded_url

"""
ПОЛУЧАЕМ ССЫЛКИ НА ВСЕ СТРАНИЦЫ С САЙТА
"""

links = ['https://www.gem.wiki/Category:Oil_and_gas_extraction']

for i, link in enumerate(links):

    response = requests.get(link)

    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content,'html.parser')
        title = soup.title.string
        print(f"Заголовок страницы: {title}, {i}")
    else:
        print(f"Не удалось получить страницу, статус-код: {response.status_code}")
    
    next_page = get_next_page(soup)

    if next_page.split('&')[-1].split('=')[0] == 'pageuntil':
        break
    else:
        links.append(next_page)

"""
ПОЛУЧАЕМ ДАННЫЕ С КАЖДОЙ СТРАНИЦЫ САЙТА
"""

tabs = []
excepts = []

for num,link in enumerate(links):

    response = requests.get(link)

    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content,'html.parser')
        title = soup.title.string
        print(f"Заголовок страницы: {title}, {num}")
    else:
        print(f"Не удалось получить страницу, статус-код: {response.status_code}")

    div = soup.find_all('div', {'class':'mw-category mw-category-columns'})[0].find_all('li')

    for d in div:

        country = d.find('a')['href'].split('(')[-1].split(',')[-1].replace('_','').replace(')','')

        if country in countries:

            link_url = get_link(d)

            response = requests.get(link_url)

            if response.status_code == 200:
                html_content = response.text
                soup = BeautifulSoup(html_content, 'html.parser')
                title = soup.title.string
                print(f"Заголовок страницы: {title}")

                try:
                
                    tab = soup.find('table', {'class':'wikitable'}).find('tbody')
                    headers = [th.text.strip() for th in tab.find_all('th')]
                    data = []
                    for row in tab.find_all('tr')[1:]:
                        cols = row.find_all('td')
                        cols = [ele.text.strip() for ele in cols]
                        data.append(cols)

                    df = pd.DataFrame(data,columns=headers)
                    df.insert(0,'Country',country)

                    tabs.append(df)
                    print(f"Таблица со страницы {link_url} получена")
                    sleep(0.01)

                except Exception as e:
                    print(f"Таблица со страницы {link_url} не получена. Причина: {e}")
                
            else:
                print(f"Не удалось получить страницу, статус-код: {response.status_code}")
                excepts.append(link_url)
            
        else:
            print(f"Страны нет в списке: {country}")

dataframe = pd.concat(tabs,ignore_index=True)

dataframe.to_csv(f'/Users/evgeniisolodov/Desktop/project/source/parsing.csv',index=False)
