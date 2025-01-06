import streamlit as st
import pandas as pd
import requests
import datetime
import sqlite3
import math
import lightgbm as lgb
import plotly.express as px
import numpy as np
from joblib import dump, load
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
client = gspread.authorize(creds)

# locs = pd.read_csv('locations.csv')
# abb = pd.read_csv('abb.csv')
# # rental = pd.read_csv('rental.csv')
# props = list(set(locs['prop_name']))



st.set_page_config(page_title='Bnb Calculator',  layout='wide')

col1, col2 = st.columns([1, 8])
with col1:
    st.write("bbb")
    #st.image("bnb.png",width=100)

with col2:
    st.title("Airbnb Calculator for Dubai")


if 'beds' not in st.session_state:
    st.session_state.beds = 0
if 'vals' not in st.session_state:
    st.session_state.vals = 0
if 'baths' not in st.session_state:
    st.session_state.baths = 1
if 'capacity' not in st.session_state:
    st.session_state.capacity = 1
if 'results' not in st.session_state:
    st.session_state.results = None
if 'rental' not in st.session_state:
    st.session_state.rental = None
if 'dld' not in st.session_state:
    st.session_state.dld = None
if 'occupancy' not in st.session_state:
    st.session_state.occupancy = 20

# option = st.selectbox(
#     "Select Building",
#     props,
# )

option = st.text_input("Enter the Co-ordinates")


option = option.split(",")
lat = option[0]
lng = option[1]
st.write(lat,lng)

#model = load('calculator.joblib')


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
         math.sin(dLon / 2) * math.sin(dLon / 2) * math.cos(lat1) * math.cos(lat2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def filter_data(df,lat,lng,radius,beds):


    df['distance'] = df.apply(lambda row: haversine(lat, lng, row['lat'], row['lng']), axis=1)
    df = df[(df['distance']<radius)& (df['beds']==beds)]
    
    return df



def predict_adr_lgb(occ, lat, lng, beds, baths, capacity, model):
    # Prepare the feature vector in the correct order
    features = np.array([[occ, lat, lng, beds, baths, capacity]])
    
    # Predict using the provided LightGBM model
    try:
        prediction = model.predict(features)[0]  # model.predict returns an array of predictions
        return prediction
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def calculate_revenue(occupancy, adr):
    # Calculating the revenue
    return 365 * occupancy * adr / 100 

def revenue(data, lat, lng, beds, baths, capacity):
    # Calculate occupancy statistics
    occ_stats = {
        'mean_occ': data['occ'].mean(),
        'median_occ': data['occ'].median(),
        'q3_occ': data['occ'].quantile(0.75),
        'p90_occ': data['occ'].quantile(0.90)
    }

    # Collect data for new DataFrame
    stats = ['mean', 'q3', 'p90']
    perf = ['Average','Good','Great']
    results = []

    for stat in stats:
        occ_key = f"{stat}_occ"
        # Use the predict_adr function to get ADR for the given occupancy
        predicted_adr = predict_adr_lgb(occ_stats[occ_key], lat, lng, beds, baths, capacity,model)
        # Calculate revenue using the predicted ADR and the occupancy statistic
        revenue = calculate_revenue(occ_stats[occ_key], predicted_adr)
        results.append({
            
            'Occupancy (%)': occ_stats[occ_key],
            'ADR': predicted_adr,
            'Estimated Revenue ($)': revenue
        })

    
    results_df = pd.DataFrame(results)
    results_df['Performance'] = perf
    # Return the DataFrame
    return results_df


def get_dld_data(beds,page,code):
    cookies = {
    'sp': '07b1c80b-bacc-4493-bdc2-e8325d963202',
    'consentGroup': 'control',
    'cookie_for_measurement_id': 'G-WC7F61HJCT',
    '__rtbh.uid': '%7B%22eventType%22%3A%22uid%22%2C%22id%22%3A%22018edd43f809001bb4963841b21905075004e06d009dc%22%7D',
    '_scid': '28b062d2-dfc7-423b-9c1d-2bdaf49dd685',
    '_fbp': 'fb.1.1713109203387.636951166',
    '_tt_enable_cookie': '1',
    '_ttp': 'gtcKj-v5zOY8TpbCcZiH1UZXa3U',
    'g_state': '{"i_p":1713195673140,"i_l":2}',
    'flagship_user_id': 'et0mck9m55jff5euoe88f7',
    'anonymous_user_id': 'et0mck9m55jff5euoe88f7',
    '_ga': 'GA1.1.117968577.1713109202',
    '_gcl_au': '1.1.1553292717.1721358279',
    'flagship_user_id': 'et0mck9m55jff5euoe88f7',
    'channeloriginator': 'notcriteo',
    'channelcloser': 'notcriteo',
    'channelflow': 'notcriteo|displayads|1725176593598',
    '_gcl_gs': '2.1.k1$i1722584741',
    '_gcl_aw': 'GCL.1722584760.Cj0KCQjwh7K1BhCZARIsAKOrVqF77422U1xtJ9W_47CMVav73yqnsbn2_4Zf_nDX7oLKNwYWRlMNRM4aAhiIEALw_wcB',
    '_sctr': '1%7C1722801600000',
    '__gads': 'ID=7792582665712a65:T=1713109204:RT=1722831553:S=ALNI_MaEMwe9irEv2E5ZyG16buUKKpLV9g',
    '__gpi': 'UID=00000d5d35b4d561:T=1713109204:RT=1722831553:S=ALNI_MaR_xEmkAW7jys_6Kw8mkxk5KMbFA',
    '__eoi': 'ID=0b2aae7580cf731a:T=1713109204:RT=1722831553:S=AA-AfjZ_zj7ZwiyHFGF_CzjHzfVN',
    '_clck': 'c3bh5%7C2%7Cfo4%7C0%7C1565',
    'criteo_user_id': 'Q7BRvl9Kd3Q5Q004UEZXUTdOS2pJN2JBNnpkUHNLNzN1UDFXYmFmNXNIciUyQkZhelklM0Q',
    '_ga_4XL587PN9G': 'GS1.1.1723010842.17.0.1723010842.60.0.1990898842',
    '_scid_r': '28b062d2-dfc7-423b-9c1d-2bdaf49dd685',
    '_sp_id.b9c1': '2ae0040b-80aa-45ec-abdb-c90991696c64.1707971574.30.1723010843.1722837422.9b5ab516-7085-45b8-b8c2-24e0a1d91c92.0948d07f-0e52-4282-bed9-e5e5149a198a.a3feae09-fcac-4d79-930a-41c4121786d5.1723010834120.27',
    '_ScCbts': '%5B%5D',
    'cto_bundle': '3DUV1l9ockpIdHR1REtsVjIwZzhMUkNJTmgxcldxVkhGa1ZqekplWXBGN1RUVUprZXNReUpvZ1FBb1dKSXpzeSUyQmY5SExraFFkSlBpZlZZak9DTVVPQmtxd2xwVEtoNzU3M0pCTjlMQWU2eVZvNnFsNEZxNWJ1JTJCRFhlbXBSTXBVdVlOdTY0dkxUTzROMTRVY3VRZzMlMkY2elh2NWxWMUxQZEd0QzlNbTk3UWNtUDJpWEE5ZVN6MmlKbjZqYU1KN2UzQTFXcXZRbkY1RWYyZjlRUkVmZGY0RTB6R2trOEl3MTNYQVlOVDU1SjRGQzVBUUE2JTJCWUolMkYlMkJMcjUxNzl6dE9xblF6NUdyb2dlZGl2V3JZMXQ0QWg2TExEYW5OelVXakJ1bENFd0tNclF3ZlJId0d6YmFJeW9lM0Jxc3JiVWRSUmU1NTZCNkRxbHFOQkJZVG5WUGNyN25JVXFZV2NYWUt6YWxIdTRzWDZmR2NJbEhtdmZyV1drNWlTVjlIMiUyRjVPRGNkZzh3Qw',
    'aws-waf-token': '7b8c3575-4efe-4020-8f94-ae4b85b5e96b:BgoAuoJAXr08AAAA:t4UPpWDUsBy4LutuT2orxjx1n0IxUBAyuwylxwZ3nc2p5+NJXky/LFkN+pcnpQjmKDg5+3ivjlhU384RBYjGzUd3+avPhfagNDV7bVrOs2kjJglm790M3ZuSnn4OQJigaMU+zsA+l0oLFSMO0dJxEiNzpnpi3TJRIQDDQMMSTav8vLs/YmbZNZ7gHIfM+PG/Z6+Oflwcuc4cXmlVZlR/5Fe6hn0Z6OOxWdeKTM5y0giyyqhCkdRsESgkmetosw==',
    '_clsk': 'xiam9j%7C1723022271287%7C3%7C1%7Ck.clarity.ms%2Fcollect',
    'utag_main': 'v_id:018edd43f809001bb4963841b21905075004e06d009dc$_sn:29$_se:3$_ss:0$_st:1723024076181$dc_visit:24$ses_id:1723022274320%3Bexp-session$_pn:1%3Bexp-session$user_phone_number_formatted_for_fb:',
    '_ga_WC7F61HJCT': 'GS1.1.1723022274.32.0.1723022276.58.0.1070529026',
    '_dd_s': 'logs=1&id=8d4a2ab3-761d-4695-aa69-e5875a0afe79&created=1723022270115&expire=1723023177879',
}

    headers = {
        'accept': '*/*',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
        # 'cookie': 'sp=07b1c80b-bacc-4493-bdc2-e8325d963202; consentGroup=control; cookie_for_measurement_id=G-WC7F61HJCT; __rtbh.uid=%7B%22eventType%22%3A%22uid%22%2C%22id%22%3A%22018edd43f809001bb4963841b21905075004e06d009dc%22%7D; _scid=28b062d2-dfc7-423b-9c1d-2bdaf49dd685; _fbp=fb.1.1713109203387.636951166; _tt_enable_cookie=1; _ttp=gtcKj-v5zOY8TpbCcZiH1UZXa3U; g_state={"i_p":1713195673140,"i_l":2}; flagship_user_id=et0mck9m55jff5euoe88f7; anonymous_user_id=et0mck9m55jff5euoe88f7; _ga=GA1.1.117968577.1713109202; _gcl_au=1.1.1553292717.1721358279; flagship_user_id=et0mck9m55jff5euoe88f7; channeloriginator=notcriteo; channelcloser=notcriteo; channelflow=notcriteo|displayads|1725176593598; _gcl_gs=2.1.k1$i1722584741; _gcl_aw=GCL.1722584760.Cj0KCQjwh7K1BhCZARIsAKOrVqF77422U1xtJ9W_47CMVav73yqnsbn2_4Zf_nDX7oLKNwYWRlMNRM4aAhiIEALw_wcB; _sctr=1%7C1722801600000; __gads=ID=7792582665712a65:T=1713109204:RT=1722831553:S=ALNI_MaEMwe9irEv2E5ZyG16buUKKpLV9g; __gpi=UID=00000d5d35b4d561:T=1713109204:RT=1722831553:S=ALNI_MaR_xEmkAW7jys_6Kw8mkxk5KMbFA; __eoi=ID=0b2aae7580cf731a:T=1713109204:RT=1722831553:S=AA-AfjZ_zj7ZwiyHFGF_CzjHzfVN; criteo_user_id=Q7BRvl9Kd3Q5Q004UEZXUTdOS2pJN2JBNnpkUHNLNzN1UDFXYmFmNXNIciUyQkZhelklM0Q; _ScCbts=%5B%5D; website_ab_tests=TEST_HOMEPAGE_HOT_PROJECTS=original,TEST_PRIMARY_CTA=original,TEST_PLP_FRESHNESS=original,TEST_SERP_CARAT=variantB,TEST_WHATSAPP_CAPTCHA=variantA,TEST_PLP_HISTORICAL_TRANSACTIONS=original,TEST_PLP_AGENT_REVIEW=variantA,SEARCH_HP_NP_CATEGORY=variantB,TEST_PLP_TO_PDP=variantA,TEST_PLP_RECOMMENDATIONS=variantB,TEST_HOMEPAGE_CTA=variantA,TOGGLE_HEADER_INSIGHTSHUB_ENTRYPOINT=original,TOGGLE_HEADER_COMMUNITIES_ENTRYPOINT=off,NP-442-hot-projects=original,TEST_PLP_UPFRONT_COST=variantA,SEARCH_HOMEPAGE_NEW_PROJECTS_CATEGORY=original,TEST_SWITCH_DATAGURU_TO_EXPLORE=variantA,TEST_PLP_PRICE_POSITION=variantA,TEST_PLP_DATAGURU_ENTRYPOINTS=variantA,TEST_SERP_DYNAMIC_RANKING=variantA,TEST_PLP_NEW_CTA=variantA,NP-534-new-projects-nav-label=original,TEST_NEW_PROJECT_CARDS=variantA,TOGGLE_TOWERINSIGHTS_FLOORPLANS=original,WEBSITE_PLP_PROJECT_LINK=original,serpDownPaymentEgp=original,test136=variantA; _sp_ses.b9c1=*; _sp_id.b9c1=2ae0040b-80aa-45ec-abdb-c90991696c64.1707971574.31.1723179189.1723010843.b9f89d4d-ddf0-4b20-b9a3-028e11a30e05.9b5ab516-7085-45b8-b8c2-24e0a1d91c92...0; _clck=c3bh5%7C2%7Cfo6%7C0%7C1565; _ga_WC7F61HJCT=GS1.1.1723179189.34.0.1723179189.60.0.1562356047; utag_main=v_id:018edd43f809001bb4963841b21905075004e06d009dc$_sn:30$_se:1$_ss:1$_st:1723180989187$dc_visit:25$ses_id:1723179189187%3Bexp-session$_pn:1%3Bexp-session$dc_event:1%3Bexp-session$dc_region:eu-west-1%3Bexp-session; _scid_r=28b062d2-dfc7-423b-9c1d-2bdaf49dd685; _ga_4XL587PN9G=GS1.1.1723179190.19.0.1723179190.60.0.2070753819; cto_bundle=_-A2UV9ockpIdHR1REtsVjIwZzhMUkNJTmg5JTJCciUyRnJYZkNXY0xraUdKZDhiMXhiSnBHdVRzT1JDa1NWSzhJSjlOM0lZa1NheXpuU0o4dThOdzFDVzVZUCUyRklEJTJCJTJCR0FQSFdzbDJmVWVFZWlEQ3hpMGhkQmdMV1MzalVqZUdraTJjJTJCaFdwYU5QOXJ3VyUyQk5KeDJxQll3MjZQc3BUdFFmUGl4JTJCem1ISDIyMHhwSmt5ak9WQlpvTGRHcGRESSUyQlo2Rk40MlYxNTlPQmJGTk5nS3BpT2lHWHRqTVoxck9kVWRGJTJGc0JFWWFNc21aZ2ZPemQwelklMkJ0RDZhQmRQam84amtLdTRoTldYSzYlMkZPU2J3bCUyRmpldkclMkZ1bUl0UldsN0J5MlNEQ2E3SU5DYkxndml3UlZVUE1WZ29wNDdwSVV5ZElMMko3ZE1vZHdMTTFoSmxKUmd4TjJETVZXJTJCRkpTdXJINFN4STJCZ1NUS1h1bUNvNmVWTFdMSGt1TEpmSWcwcm5LNDZ2bDdUOHg; aws-waf-token=23fcb0ab-e99b-4ceb-a9d5-df3957638988:BgoAmskgz4LfAQAA:spRFh3cjEtKbanmzuWZjopYGE10dyF+EABARZiulQhmoeuRpRdMD3nLAGnG/U1qs08+0f5k+UqYzUMAGACGsgN8T5edQGxHx0tYTX7nxQJ/4e4yfN8imqlUkU7ViMgBuaN8FVszH3RxsGnG2Bqqmh19p2W+ptH9gTOaJFNNA9jm7NTftMzC/1ta3D4mKCMEKQmFZVsMX1QCePEEi54yUVQ3gtyPBT0bE+Gq6dhIn23gJdJeDSStK/uhK3qAbCA==; _clsk=13s7voo%7C1723179192447%7C2%7C0%7Cu.clarity.ms%2Fcollect; _dd_s=logs=1&id=22fa0345-63c3-41a4-b71e-e426a891ff04&created=1723179189066&expire=1723180094293',
        'priority': 'u=1, i',
        'referer': 'https://www.propertyfinder.ae/en/transactions/rent/dubai',
        'sec-ch-ua': '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
        'x-nextjs-data': '1',
    }

    params = {
        'period': '12m',
        'bdr[]': '{}'.format(beds),  # Assuming 'beds' is defined somewhere in your code
        'fu': '0',
        'rp': 'y',
        'ob': 'mr',
        'category': 'rent',
        'page': page,  # Assuming 'page' is defined somewhere in your code
        'slug': [
            'dubai',
            '{}'.format(code),  # Assuming 'code' is defined somewhere in your code
        ],
    }

    response = requests.get(
         f'https://www.propertyfinder.ae/dataguru/_next/data/L5BaGX8YBGngFG9VUlhJ6/en/transactions/rent/dubai/{prop_code}.json',
        params=params,
        #cookies=cookies,
        headers=headers,
    )
    #st.write(response.text)
    return response.json()['pageProps']['list']['transactionList']

pages = [1,2,3,4,5]

coordinate_template = '({lat},{lng})'
def get_rev_data(lat,lng,beds,baths):
    cookies = {
        '_ga': 'GA1.1.1395615341.1722789518',
        '_fbp': 'fb.1.1722789518668.516233519452759718',
        'FPID': 'FPID2.2.ZrpvVbTlVZOSimQuCsUWKb5qHIKHo4y8BgcGqDJI6Pg%3D.1722789518',
        'FPAU': '1.2.833997574.1722789519',
        'intercom-id-k4m5dmsd': 'e10aca0c-0714-489c-9802-50d8fc6d3c0e',
        'intercom-device-id-k4m5dmsd': '9b8ee412-8b42-46c6-8ce2-2d11a06a8466',
        '__stripe_mid': '270e4096-b0a9-4489-b4c6-028dd7115d979370c9',
        'intercom-session-k4m5dmsd': '',
        '__client_uat': '1730287417',
        '__client_uat_mgHTUfCB': '1730287417',
        '_gtmeec': 'e30%3D',
        'FPLC': 'lsUuj4ap%2FzxnlTHZntom0gSC9sVu5J9NcJbs4V2j%2B9g7ZOPSjIr3U%2BAuGaHkfGri5DSuU7cjGetVhnkR0QjI62%2B%2BEPBrwd0G44s7CQu6tb1ZBiM6SnyngoURb2%2B4AQ%3D%3D',
        '__stripe_sid': 'dbe4f43b-c3c2-4c30-bc6e-1e4e968c8a4daaff64',
        '_rdt_uuid': '1722789517756.5aa748ba-f9fa-454e-8c1a-5d8a8d5e9943',
        '_ga_DQCJRX8Z10': 'GS1.1.1730464124.18.1.1730464904.0.0.1321138456',
        'FPGSID': '1.1730464124.1730464910.G-DQCJRX8Z10.drq9jZM-eaSJ0PCWaJRpmQ',
        '__session': 'eyJhbGciOiJSUzI1NiIsImNhdCI6ImNsX0I3ZDRQRDExMUFBQSIsImtpZCI6Imluc18yUlJRaTNRQWcwY2JETVRiZEN6V2Y3eFZZQXUiLCJ0eXAiOiJKV1QifQ.eyJhenAiOiJodHRwczovL3d3dy5ibmJjYWxjLmNvbSIsImV4cCI6MTczMDQ2NDk4NSwiaWF0IjoxNzMwNDY0OTI1LCJpc3MiOiJodHRwczovL2NsZXJrLmJuYmNhbGMuY29tIiwibmJmIjoxNzMwNDY0OTE1LCJzaWQiOiJzZXNzXzJvOWd0QWlmVUlsT3U1SnkwaFBPTHdIdDR4MiIsInN1YiI6InVzZXJfMmtuSzcxSUtUUVFZSDVaek9uM2JqNFpLQXJYIn0.QJ48znAs5XePPXxZR5LhOuBuZFcQXOheVcIhHh5b4U-bTDsisrFTmr0S0B666rgkoZZvpHDSEZUMFJi8SY_xXLCFb3tI7ayb2MjqnAvtDvBlBXS81c70Ah-RKq0Hzagh9miXJzDWsWqKU7O0Wd5SnL0qs_HLuR-cSKMvkNT_-_5CdWCMQxDbO9vtqR2Z7ZxyDAeBPppbIP3Ke1qadXceuy5SlG1ocpIVjvhmkyEsoHTWFoVjMH9G8v8r4NWge2dCEPAHjisFu4d6xvFAzNjXGtoT4arYZZspqMQ021hEsmZttpcvCjv7JvAZ1C3Dhs7I5B48N10tpgIU2WMn37iM3A',
        'ph_phc_y6myC6jTw03r1STdgAWKNVHTxdqQH6T6lFZg8ORMQgL_posthog': '%7B%22distinct_id%22%3A%2266c0d6034d469c146aa8dfd0%22%2C%22%24sesid%22%3A%5B1730464961139%2C%220192e7b3-00a2-7cca-9464-47e6ffbb1ca0%22%2C1730464120994%5D%2C%22%24session_is_sampled%22%3Atrue%2C%22%24epp%22%3Atrue%7D',
    }

    headers = {
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
        'authorization': 'Bearer eyJhbGciOiJSUzI1NiIsImNhdCI6ImNsX0I3ZDRQRDExMUFBQSIsImtpZCI6Imluc18yUlJRaTNRQWcwY2JETVRiZEN6V2Y3eFZZQXUiLCJ0eXAiOiJKV1QifQ.eyJhenAiOiJodHRwczovL3d3dy5ibmJjYWxjLmNvbSIsImV4cCI6MTczMDQ2NDk4NSwiaWF0IjoxNzMwNDY0OTI1LCJpc3MiOiJodHRwczovL2NsZXJrLmJuYmNhbGMuY29tIiwibmJmIjoxNzMwNDY0OTE1LCJzaWQiOiJzZXNzXzJvOWd0QWlmVUlsT3U1SnkwaFBPTHdIdDR4MiIsInN1YiI6InVzZXJfMmtuSzcxSUtUUVFZSDVaek9uM2JqNFpLQXJYIn0.QJ48znAs5XePPXxZR5LhOuBuZFcQXOheVcIhHh5b4U-bTDsisrFTmr0S0B666rgkoZZvpHDSEZUMFJi8SY_xXLCFb3tI7ayb2MjqnAvtDvBlBXS81c70Ah-RKq0Hzagh9miXJzDWsWqKU7O0Wd5SnL0qs_HLuR-cSKMvkNT_-_5CdWCMQxDbO9vtqR2Z7ZxyDAeBPppbIP3Ke1qadXceuy5SlG1ocpIVjvhmkyEsoHTWFoVjMH9G8v8r4NWge2dCEPAHjisFu4d6xvFAzNjXGtoT4arYZZspqMQ021hEsmZttpcvCjv7JvAZ1C3Dhs7I5B48N10tpgIU2WMn37iM3A',
        'content-type': 'application/json',
        # 'cookie': '_ga=GA1.1.1395615341.1722789518; _fbp=fb.1.1722789518668.516233519452759718; FPID=FPID2.2.ZrpvVbTlVZOSimQuCsUWKb5qHIKHo4y8BgcGqDJI6Pg%3D.1722789518; FPAU=1.2.833997574.1722789519; intercom-id-k4m5dmsd=e10aca0c-0714-489c-9802-50d8fc6d3c0e; intercom-device-id-k4m5dmsd=9b8ee412-8b42-46c6-8ce2-2d11a06a8466; __stripe_mid=270e4096-b0a9-4489-b4c6-028dd7115d979370c9; intercom-session-k4m5dmsd=; __client_uat=1730287417; __client_uat_mgHTUfCB=1730287417; _gtmeec=e30%3D; FPLC=lsUuj4ap%2FzxnlTHZntom0gSC9sVu5J9NcJbs4V2j%2B9g7ZOPSjIr3U%2BAuGaHkfGri5DSuU7cjGetVhnkR0QjI62%2B%2BEPBrwd0G44s7CQu6tb1ZBiM6SnyngoURb2%2B4AQ%3D%3D; __stripe_sid=dbe4f43b-c3c2-4c30-bc6e-1e4e968c8a4daaff64; _rdt_uuid=1722789517756.5aa748ba-f9fa-454e-8c1a-5d8a8d5e9943; _ga_DQCJRX8Z10=GS1.1.1730464124.18.1.1730464904.0.0.1321138456; FPGSID=1.1730464124.1730464910.G-DQCJRX8Z10.drq9jZM-eaSJ0PCWaJRpmQ; __session=eyJhbGciOiJSUzI1NiIsImNhdCI6ImNsX0I3ZDRQRDExMUFBQSIsImtpZCI6Imluc18yUlJRaTNRQWcwY2JETVRiZEN6V2Y3eFZZQXUiLCJ0eXAiOiJKV1QifQ.eyJhenAiOiJodHRwczovL3d3dy5ibmJjYWxjLmNvbSIsImV4cCI6MTczMDQ2NDk4NSwiaWF0IjoxNzMwNDY0OTI1LCJpc3MiOiJodHRwczovL2NsZXJrLmJuYmNhbGMuY29tIiwibmJmIjoxNzMwNDY0OTE1LCJzaWQiOiJzZXNzXzJvOWd0QWlmVUlsT3U1SnkwaFBPTHdIdDR4MiIsInN1YiI6InVzZXJfMmtuSzcxSUtUUVFZSDVaek9uM2JqNFpLQXJYIn0.QJ48znAs5XePPXxZR5LhOuBuZFcQXOheVcIhHh5b4U-bTDsisrFTmr0S0B666rgkoZZvpHDSEZUMFJi8SY_xXLCFb3tI7ayb2MjqnAvtDvBlBXS81c70Ah-RKq0Hzagh9miXJzDWsWqKU7O0Wd5SnL0qs_HLuR-cSKMvkNT_-_5CdWCMQxDbO9vtqR2Z7ZxyDAeBPppbIP3Ke1qadXceuy5SlG1ocpIVjvhmkyEsoHTWFoVjMH9G8v8r4NWge2dCEPAHjisFu4d6xvFAzNjXGtoT4arYZZspqMQ021hEsmZttpcvCjv7JvAZ1C3Dhs7I5B48N10tpgIU2WMn37iM3A; ph_phc_y6myC6jTw03r1STdgAWKNVHTxdqQH6T6lFZg8ORMQgL_posthog=%7B%22distinct_id%22%3A%2266c0d6034d469c146aa8dfd0%22%2C%22%24sesid%22%3A%5B1730464961139%2C%220192e7b3-00a2-7cca-9464-47e6ffbb1ca0%22%2C1730464120994%5D%2C%22%24session_is_sampled%22%3Atrue%2C%22%24epp%22%3Atrue%7D',
        'origin': 'https://www.bnbcalc.com',
        'priority': 'u=1, i',
        'referer': 'https://www.bnbcalc.com/create-analysis',
        'sec-ch-ua': '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
        'sec-ch-ua-mobile': '?1',
        'sec-ch-ua-platform': '"Android"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36',
    }

    json_data = {
        'coordinate': coordinate_template.format(lat=lat, lng=lng),
        'bedrooms': beds,   # You can set these values as needed
        'bathrooms': baths, # You can set these values as needed
        'apiResponseType': 'estimator_with_comps_ltm',
    }

    response = requests.post('https://www.bnbcalc.com/api/airbtics', cookies=cookies, headers=headers, json=json_data)

    d = response.json()
    st.write(d)
    return d


with st.form(key='property_form'):
    # Create a selectbox for the number of beds
    beds = st.selectbox('Select number of beds:', options=[0, 1, 2, 3, 4,5 ,6])

    # Create a selectbox for the number of baths
    baths = st.selectbox('Select number of baths:', options=[1, 2, 3, 4, 5,6])

    currency = st.selectbox("select currency:", options=["AED","USD"])

    # Create a slider for selecting capacity
    capacity = st.slider('Select capacity:', min_value=1, max_value=12, value=1)


    # Create a submit button
    submit_button = st.form_submit_button("Submit")

if submit_button:

    st.session_state.selected_beds = beds
    st.session_state.selected_baths = baths
    st.session_state.selected_capacity = capacity

    d = get_rev_data(lat,lng,beds,baths)

    percentile_df = pd.DataFrame.from_dict(d['message']['last_12_months_summary']['quartiles'], orient='index')
    percentile_df.reset_index(inplace=True)
    percentile_df.columns = ['Month', 'adr', 'occ', 'revenue']

    monthly_df = pd.DataFrame.from_dict(d['message']['monthly_summary'],orient='index')
    
    st.write(percentile_df.T)
    st.write(pd.DataFrame(d['message']['comps']))
    
    if currency == "AED":
        ##covert to aed
        monthly_df[['average_daily_rate', 'average_revenue']] *= 3.67

    st.write(monthly_df.T)

    median_adr = percentile_df.loc[percentile_df['Month'] == '50th_percentile', 'adr'].values[0]
    median_occ = percentile_df.loc[percentile_df['Month'] == '50th_percentile', 'occ'].values[0]
    median_revenue = percentile_df.loc[percentile_df['Month'] == '50th_percentile', 'revenue'].values[0]

    # Calculate scale factors for 75th and 90th percentiles
    scale_75th_adr = percentile_df.loc[percentile_df['Month'] == '75th_percentile', 'adr'].values[0] / median_adr
    scale_75th_occ = percentile_df.loc[percentile_df['Month'] == '75th_percentile', 'occ'].values[0] / median_occ


    scale_75th_revenue = percentile_df.loc[percentile_df['Month'] == '75th_percentile', 'revenue'].values[0] / median_revenue

    new_monthly = pd.DataFrame()
    new_monthly['adr_75th'] = monthly_df['average_daily_rate'] * scale_75th_adr
    new_monthly['occ_75th'] = np.maximum(monthly_df['average_occupancy_rate'] * scale_75th_occ,97.4354)
    new_monthly['revenue_75th'] = monthly_df['average_revenue'] * scale_75th_revenue
    new_monthly = new_monthly.T
    # Assuming your DataFrame is named `df` and the columns are already loaded as in your image
# Convert column names to datetime
    new_monthly.columns = pd.to_datetime(new_monthly.columns, format='%Y-%m')

   
    # Rename columns to abbreviated month names
    new_monthly.columns = new_monthly.columns.strftime('%b')
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    new_monthly = new_monthly.reindex(columns=sorted(new_monthly.columns, key=lambda x: months_order.index(x.split(' ')[0])))

    # Optionally sort the DataFrame by the column names if they aren't in order
    

    st.write(new_monthly)

    
    sheet = client.create('New Monthly Data Sheet')  # Name your sheet
    worksheet = sheet.get_worksheet(0)  # Get the first sheet
    
    # Convert DataFrame to list of lists
    data_to_export = [new_monthly.columns.values.tolist()] + new_monthly.values.tolist()
    
    # Update the Google Sheet with data
    worksheet.update('A1', data_to_export)  
    #new_monthly.T.to_csv('kk.csv')

    #st.write(f"You selected {beds} beds, {baths} baths, and a capacity of {capacity}.")
    # df = filter_data(abb,lat, lng,0.05,beds)
    # st.session_state.vals = len(df) 
    # rental_df = filter_data(rental,lat,lng,0.05,beds)
    # res = revenue(df,lat, lng,beds,baths,capacity)
    # st.session_state.results = res
    # st.session_state.rental = rental_df
    # prop_code = locs[locs['prop_name'] == option]['prop_code'].iloc[0]
   # st.write(prop_code)

    dld = pd.DataFrame()
    # for page in pages:
    #     dld_df = pd.DataFrame(get_dld_data(beds,page,prop_code))
    #     dld = pd.concat([dld,dld_df],axis=0)
     
    st.session_state.dld = dld


if st.session_state.vals >0:
    st.title('Performance Overview')
   
    col1, col2, col3 = st.columns(3)
    performance_containers = {
        'Average': col1,
        'Good': col2,
        'Great': col3
    }

    st.markdown("""
    <style>
    div.stMarkdown {
        border: 1px solid #f0f0f0;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    # Iterate through each row in the DataFrame and display the information
    for index, row in st.session_state.results.iterrows():
        performance = row['Performance']
        occupancy = row['Occupancy (%)']
        adr = row['ADR']
        revenue = row['Estimated Revenue ($)']

        container = performance_containers[performance]
        with container:
            # Using one markdown block to create a single bordered "container"
            markdown_content = f"""
    <div class="border-md">
        <h4>{performance} Performance</h4>
        <p><strong>Expected Revenue:</strong> AED {revenue:,.2f}</p>
        <p><strong>Occupancy:</strong> {occupancy:.2f}%</p>
        <p><strong>ADR:</strong> AED {adr:,.2f}</p>
    </div>
    """
            container.markdown(markdown_content, unsafe_allow_html=True)
            
        # st.markdown(f"""
        # ### {performance} Performance:
        # {performance} occupancy in this building/region is **{occupancy:.2f}%**. 
        # With this performance, you can expect to make **AED {revenue:,.2f}** 
        # at an ADR of **AED {adr:.2f}**.
        # """)


    # st.title('Predicted Revenue based on Occupancy]')
    # value = st.slider(
    # 'Select Occupancy',  # Title of the slider
    # min_value=0,      # Starting value of the slider
    # max_value=90,     # Maximum value of the slider
    # value=20,          # Initial value of the slider
    # step=5             # Step size for the slider
    # )   


    # predicted_adr = predict_adr_lgb(value, lat, lng, beds, baths, capacity,model)
    # rev = calculate_revenue(value, predicted_adr)

    # st.markdown(f"""
    #     With **{value:.2f}%** occupancy, you can expect to make **AED {np.round(rev,0)}**
    #     """)
    cola, colb, colc = st.columns(3)

# First column for DLD data
    with cola:
       
        avg_price_dld = np.round(st.session_state.dld[dld['status']=='Renewed']['price'].mean(), 0)
        dld_html_content = f"""
        <div style="border: 2px solid #ccc; padding: 10px; border-radius: 5px;">
            <h4>DLD Data - Renewed Rentals</h4>
            <p><strong>Average Rental Rate:</strong> AED {avg_price_dld:,.2f}</p>
            <p>For {beds} Bed Apartment in {option}</p>
        </div>
        """
        st.markdown(dld_html_content, unsafe_allow_html=True)

    with colb:
       
        avg_price_dld = np.round(st.session_state.dld[dld['status']=='New']['price'].mean(), 0)
        dld_html_content = f"""
        <div style="border: 2px solid #ccc; padding: 10px; border-radius: 5px;">
            <h4>DLD Data - New Rentals</h4>
            <p><strong>Average Rental Rate:</strong> AED {avg_price_dld:,.2f}</p>
            <p>For {beds} Bed Apartment in {option}</p>
        </div>
        """
        st.markdown(dld_html_content, unsafe_allow_html=True)

    # Second column for PropertyFinder/Bayut data
    with colc:
        #st.header('PropertyFinder/Bayut Data')
        avg_price_rental = np.round(st.session_state.rental['price'].mean(), 0)
        rental_html_content = f"""
        <div style="border: 2px solid #ccc; padding: 10px; border-radius: 5px;">
            <h4>PropertyFinder/Bayut Data</h4>
            <p><strong>Average Rental Ask Rate:</strong> AED {avg_price_rental:,.2f}</p>
            <p>For {beds} Bed Apartment in {option}</p>
        </div>
        """
        st.markdown(rental_html_content, unsafe_allow_html=True)
    fig = px.histogram(st.session_state.rental, x="price",nbins=20)

    st.header("Distribution of Rental Ask rates")
    # Display the figure in Streamlit
    st.markdown("The below chart shows distribution of what landlords are asking. Each column shows the range of rent and number of properties in that range")
    st.plotly_chart(fig)

    st.dataframe(st.session_state.dld)
    df_ = st.session_state.dld
    col_1, col_2, col_3, col_4 = st.columns(4)

    with col_1:
        period = st.checkbox("MTD") 
    with col_2:
        period = st.checkbox("1 Month")  
    with col_3:
        period = st.checkbox("3 Months") 
    with col_4:
        period = st.checkbox("6 Months")  
    
    if period:
        df_ = df_[df_['']]

else:
    st.markdown("## Data not available for the Building/Area")

# footer = """
# <style>
# .footer {
#     position: fixed;
#     left: 0;
#     bottom: 0;
#     width: 100%;
#     background-color: transparent;
#     color: gray;
#     text-align: center;
#     padding: 10px;
#     font-size: 16px;
# }
# </style>
# <div class="footer">
#     <p>Made with ❤️ by <a href="https://yourwebsite.com" target="_blank">Purushottam Deshpande</a></p>
# </div>
# """

# st.markdown(footer, unsafe_allow_html=True)

#     """)
     
