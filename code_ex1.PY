"""
@Title: Ex1 needle in data haystack
@Description: A crawling pipeline running on kickstart.com website.
@authors : Rom Cohen , Roy Schossberger.
"""
import datetime
import json
import re
import time
from random import uniform
import requests
from bs4 import BeautifulSoup

# Constants :
REQUESTED_PAGES = 300
PROJ_IN_PAGE = 12
INDEX_PAGE = "https://www.kickstarter.com/discover/advanced?category_id=16&woe_id=0&sort=magic&seed=2729300&page={}"


def crawl(numbers_of_pages):
    """
    Main method, runs the all pipeline of crawling the website and creating our Json object.
    :param numbers_of_pages: the number of pages the user would to get data from.
    :return: python dictionary object contains an entrance for each project in kickstart.com
    """
    json_object = {'records': {'record': []}}
    for i in range(numbers_of_pages):
        page_data = crawl_category_page(i)
        time.sleep(uniform(0, 1))
        dic_lst = extract_projects_information(page_data)
        length = len(dic_lst)
        for j in range(length):
            print("finished {}%".format(int(((j + (PROJ_IN_PAGE * i)) / (PROJ_IN_PAGE * numbers_of_pages)) * 100)))
            dic_lst[j]['story'] = get_story_from_dic(dic_lst[j])
            time.sleep(uniform(0, 1))
            dic_lst[j]['days_to_go'] = get_days_to_go(extract_link_from_dic(dic_lst[j]))
            time.sleep(uniform(0, 1))
            print('finished with {}'.format(dic_lst[j]['name']))
            json_object['records']['record'].append(extract_data_from_dic(dic_lst[j]))
    return json_object


def write_to_file(json_object):
    """
    Write dictionary to Json file
    :param json_object:
    """
    with open('output/data.json', 'w') as writer:
        writer.write(json.dumps(json_object))


def extract_data_from_dic(dic):
    """
    Extracts the data ,needed to us, to our own dictionary.
    :param dic: dictionary of project.
    :return: dictionary.
    """
    output = dict()
    try:
        output['Creator'] = dic['creator']['name']
        output['Title'] = dic['name']
        output['Text'] = dic['story']
        output['DollarsPledged'] = dic['pledged']
        output['DollarsGoal'] = dic['goal']
        output['NumBackers'] = dic['backers_count']
        output['DaysToGo'] = dic['days_to_go']
        output['AllOrNothing'] = get_date_from_ts(dic['deadline'])
    except:
        print('none value has been given')
    finally:
        output = {}
    return output


# TODO : add try catch to all reads.
def get_days_to_go(link):
    """
    Extracts from an inner link the days left field
    :param link: link of the project.
    :return: string stands for days left for the project.
    """
    result = requests.get(link)
    soup = BeautifulSoup(result.text, "html.parser")
    if soup is not None:
        val = soup.find('span', class_='block type-16 type-28-md bold dark-grey-500')
        if val is not None:
            return val.contents[0]
    else:
        return '-1'


def get_date_from_ts(time_stamp):
    """
    Converts the deadline time to due date.
    :param time_stamp:
    :return: date of all or nothing field.
    """
    time_stamp = int(time_stamp)
    dt = datetime.datetime.fromtimestamp(time_stamp)
    return str(dt.date())


def get_story_from_dic(dic):
    """
    Gets the link from dic and calls a method for extracting the story.
    :param dic: dictionary of the chosen project.
    :return: story of the project
    """
    link = extract_link_from_dic(dic)
    return get_story(link)[0]


def extract_link_from_dic(page_dic):
    """
    Return the link from the page dictionary
    :param page_dic:
    :return: link - string
    """
    return page_dic['urls']['web']['project']


def crawl_category_page(page_number):
    """
    Crawl one page of the main pages on our website.
    :param page_number: the page number in the main pages.
    :return: soup object contains the data in root folder
    """
    result = requests.get(INDEX_PAGE.format(page_number))
    soup = BeautifulSoup(result.text, 'html.parser')
    projects = soup.find_all('div', class_='js-react-proj-card grid-col-12 grid-col-6-sm grid-col-4-lg')
    return projects


def extract_projects_information(projects_data):
    """
    Extract projects dictionaries list from BeautifulSoup object.
    :param projects_data: BeautifulSoup object.
    :return: list of dictionaries each one belongs to a single project in page.
    """
    project_list = list()
    for project in projects_data:
        project_list.append(json.loads(project['data-project']))
    return project_list


def get_story(link):
    """
    Extracts the story from the project link.
    :param link: link of project.
    :return: story and it title tuple
    """
    slug = re.search('/projects/(.*)', link).group(1)
    s = requests.Session()
    r = s.get("https://www.kickstarter.com")
    soup = BeautifulSoup(r.text, 'html.parser')
    xcsrf = soup.find("meta", {"name": "csrf-token"})["content"]

    query = """
    query Campaign($slug: String!) {
      project(slug: $slug) {
        risks
        story(assetWidth: 680)
      }
    }"""

    r = s.post("https://www.kickstarter.com/graph",
               headers={
                   "x-csrf-token": xcsrf
               },
               json={
                   "operationName": "Campaign",
                   "variables": {
                       "slug": slug
                   },
                   "query": query
               })

    result = r.json()
    story_ret = result["data"]["project"]["story"]
    soup = BeautifulSoup(story_ret, 'html.parser')
    story = ""
    for p in soup.find_all('p'):
        story = story + "\n" + p.get_text()
    # print(story)
    return story, slug


# running crawl on REQUESTED_PAGES.
if __name__ == '__main__':
    json_obj = crawl(REQUESTED_PAGES)
    write_to_file(json_obj)
