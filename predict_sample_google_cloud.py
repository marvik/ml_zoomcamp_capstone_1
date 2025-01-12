import requests


url = 'https://online-shopper-prediction-558797510368.us-central1.run.app/predict'  

# Sample input data
customer = {
    'administrative': 4,
    'administrative_duration': 61.0,
    'informational': 0,
    'informational_duration': 0.0,
    'productrelated': 2867,
    'productrelated_duration': 607.0,
    'bouncerates': 0.1,
    'exitrates': 0.002,
    'pagevalues': 17.5,
    'specialday': 1.0,
    'month': 'feb',
    'operatingsystems': 1,
    'browser': 1,
    'region': 7,
    'traffictype': 4,
    'visitortype': 'returning_visitor',
    'weekend': True
}

# Send request to the prediction endpoint
try:
    response = requests.post(url, json=customer, timeout=10).json()
    print(response)

    if response['revenue']:
        print('Customer will complete a purchase')
    else:
        print('Customer will not complete a purchase')

except requests.exceptions.ConnectionError as e:
    print(f"Error: Could not connect to the server at {url}. Please make sure the server is running and the URL is correct.")
    print(f"Details: {e}")

except requests.exceptions.Timeout as e:
    print(f"Error: Connection to the server at {url} timed out.")
    print(f"Details: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")