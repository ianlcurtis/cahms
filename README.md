# Streamlit Prompt flow App

Lightweight Streamlit app to connect to a Prompt flow API

## To deploy

Create a resource group for the app to land in.

Login to the Azure cli with:

```console
az login
```

```console
az group create --n rg-streamlittest --location uksouth
```



Deploy with:

```console
az deployment group create --resource-group rg-streamlittest --parameter infra/test.bicepparam 
```

To preview changes

```console
az deployment group what-if 
```


### Connect to Prompt flow
1. Create a promptflow.
2. Deploy managed endpoint.
3. Create a .env file and add the following:
    - TITLE="" [optional]
    - LOGO_URL="" [optional]
    - PROMPTFLOW_ENDPOINT=""
    - FEEDBACK_ENDPOINT=""
    - PROMPTFLOW_KEY=""