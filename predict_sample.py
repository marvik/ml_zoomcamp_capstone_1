import requests


url = 'http://localhost:9696/predict'  

# Sample input data
customer = {
    'administrative': 0,
    'administrative_duration': 0.0,
    'informational': 0,
    'informational_duration': 0.0,
    'productrelated': 1,
    'productrelated_duration': 0.0,
    'bouncerates': 0.2,
    'exitrates': 0.2,
    'pagevalues': 0.0,
    'specialday': 0.0,
    'month': 'feb',
    'operatingsystems': 2,
    'browser': 2,
    'region': 1,
    'traffictype': 3,
    'visitortype': 'returning_visitor',
    'weekend': False
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