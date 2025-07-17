import urllib.request
import json

import streamlit as st




def process_with_promptflow(data, endpoint, api_key):

    body = str.encode(json.dumps(data))


    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")


    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(endpoint, body, headers)

    with st.spinner("Waiting for response..."):
        try:
            response = urllib.request.urlopen(req)

            result = response.read()
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))

          
            print(error.info())
            print(error.read().decode("utf8", 'ignore'))
        
    return result




def feedback(feedback, api_key, feedback_endpoint ):
    
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
    try:
        feedback_body = str.encode(json.dumps(feedback))
        feedback_req = urllib.request.Request(feedback_endpoint, feedback_body, headers)
        response = urllib.request.urlopen(feedback_req)
        result=response.read()

    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))