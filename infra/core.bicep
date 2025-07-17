param location string = 'uksouth'

param suffix string =  uniqueString('st-chat')

 @description('Branch of the repository for deployment.')
 param repositoryBranch string = 'main'

resource appServicePlan 'Microsoft.Web/serverfarms@2020-06-01' = {
  name: 'streamlit-app-service-plan-${suffix}'
  location: location
  properties: {
    reserved: true
  }
  sku: {
    name: 'B1'
  }
  kind: 'app'
}

resource appService 'Microsoft.Web/sites@2020-06-01' = {
  name: 'streamlit-chat-app-${suffix}'
  location: location
  properties: {
    serverFarmId: appServicePlan.id
    siteConfig: {
      appSettings: [
        {
          name: 'WEBSITES_PORT'
          value: '8501'
        }
        {
          name: 'TEXT_TO_REPLACE_TITLE'
          value: 'text to replace title'
        }
        {
          name: 'PROMPT_FLOW_ENDPOINT'
          value: 'endpoint'
        }
        {
          name: 'SCM_BASIC_AUTH_ENABLED' // Enable SCM basic authentication
          value: 'true'
        }
      ]
      linuxFxVersion: 'PYTHON|3.12'
      webSocketsEnabled: true
      appCommandLine: 'python -m streamlit run src/Home.py --server.port 8000 --server.address 0.0.0.0'
    }
  }
}

resource srcControls 'Microsoft.Web/sites/sourcecontrols@2021-01-01' = {
  parent: appService
  name: 'web'
  properties: {
    repoUrl: 'https://github.com/Hsenrab/streamlit_chat.git'
    branch: repositoryBranch
    isManualIntegration: true
  }
}

output appServiceName string = appService.name
