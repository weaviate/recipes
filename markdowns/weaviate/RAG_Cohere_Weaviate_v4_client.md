---
layout: recipe
colab: https://colab.research.google.com/github/weaviate/recipes/blob/main/integrations/cloud-hyperscalers/aws/RAG_Cohere_Weaviate_v4_client.ipynb
toc: True
title: "RAG with Cohere on Amazon Bedrock and Weaviate on AWS Market place"
featured: False
integration: False
agent: False
tags: ['RAG', 'Cohere', 'AWS', 'Amazon Bedrock']
---
    
# Retrieval-Augmented Generation with Cohere language models on Amazon Bedrock and Weaviate vector database on AWS Market place

The example use case generates targeted advertisements for vacation stay listings based on a target audience. The goal is to use the user query for the target audience (e.g., “family with small children”) to retrieve the most relevant vacation stay listing (e.g., a listing with playgrounds close by) and then to generate an advertisement for the retrieved listing tailored to the target audience.

Note that the following code uses the newer `v4` Weaviate Python client, which uses gRPC under the hood and is currently in beta (as of November 2023).

This notebook should work well with the Data Science 3.0 kernel in SageMaker Studio.

## Dataset Overview
The dataset is available from [Inside AirBnB](http://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2023-09-03/data/listings.csv.gz) and is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

Download the data and save it in a folder called `data`.


```python
!wget http://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2023-09-03/data/listings.csv.gz
```


```python
!gzip -d listings.csv.gz
!mkdir data
!mv listings.csv ./data
```


```python
import pandas as pd
import json

# Read CSV file
csv_file = './data/listings.csv'
df = pd.read_csv(csv_file, usecols=['host_name',
                                    'property_type',
                                    'description',
                                    'neighborhood_overview',
                                    ])

df.fillna('Unknown', inplace=True)

display(df.head())
```

## Prerequisites
To be able to follow along and use any AWS services in the following tutorial, please make sure you have an [AWS account](https://signin.aws.amazon.com/signin?redirect_uri=https%3A%2F%2Fportal.aws.amazon.com%2Fbilling%2Fsignup%2Fresume&client_id=signup).

## Step 1: Enable components of the AI-native technology stack
First, you will need to enable the relevant components discussed in the solution overview in your AWS account.
First, enable access to the Cohere Command and Embed foundation models available on Amazon Bedrock through the AWS Management Console. Navigate to the Model access page, click on Edit, and select the foundation models of your choice.

Next, set up a Weaviate cluster. First, subscribe to the [Weaviate Kubernetes Cluster on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-cicacyv63r43i). Then, launch the software using a [CloudFormation template according to your preferred availability zone](https://weaviate.io/developers/weaviate/installation/aws-marketplace#aws-marketplace). The CloudFormation template is pre-filled with default values. To follow along in this guide, edit the following fields:
* Stack name: Enter a stack name
* Authentication: It is recommended to enable authentication by setting helmauthenticationtype to apikey and defining a helmauthenticationapikey.
* Enabled modules: Make sure “tex2vec-aws” and “generative-aws” are present in the list of enabled modules within Weaviate.

This template takes about 30 minutes to complete.

## Step 2: Connect to Weaviate
On the SageMaker console, navigate to Notebook instances and create a new notebook instance.

Then, install the Weaviate client package with the required dependencies:


```python
! pip install --pre "weaviate-client==4.*"
! pip install grpcio
```

Now, you can connect to your Weaviate instance with the following code. You can find the relevant information as follows:
* Weaviate URL: Access Weaviate via the load balancer URL. Go to the Services section of AWS, under EC2 > Load Balancers find the load balancer, and look for the DNS name column.
* Weaviate API Key: This is the key you set earlier in the CloudFormation template (helmauthenticationapikey).
* AWS Access Key: You can retrieve the access keys for your user in the AWS Identity and Access Management (IAM) Console.


```python
import weaviate

client = weaviate.WeaviateClient(
    weaviate.ConnectionParams.from_url("<YOUR-WEAVIATE-URL>", grpc_port=50051),  # e.g. weaviate.ConnectionParams.from_url("http://localhost:8080", grpc_port=50051)
    auth_client_secret=weaviate.AuthApiKey("<YOUR-WEAVIATE-API-KEY>"),
    additional_headers={
        "X-AWS-Access-Key": "<AWS-ACCESS-KEY>",
        "X-AWS-Secret-Key": "<AWS-ACCESS-SECRET>"
    }
)

print(client.is_ready())

client.get_meta()
```

## Step 3: Configure the Amazon Bedrock module to enable Cohere models

Next, you will define a data collection (i.e., `class`) called `Listings` to store the listings’ data objects, which is analogous to creating a table in a relational database. In this step, you will configure the relevant modules to enable the usage of Cohere language models hosted on Amazon Bedrock natively from within the Weaviate vector database. The vectorizer (`"text2vec-aws"`) and generative module (` "generative-aws"`) are specified in the data collection definition. Both of these modules take three parameters:
* `"service"`: `"bedrock"` for Amazon Bedrock (Alternatively, `"sagemaker"` for Amazon Sagemaker Jumpstart)
* `"Region"`: The region where your model is deployed
* `"model"`: The foundation model’s name

In this step, you will also define the structure of the data collection by configuring its properties. Aside from the property’s name and data type, you can also configure if only the data object shall be stored or if it shall be stored together with its vector embeddings. In this example, `host_name` and `property_type` are not vectorized.

Run the following code to create the collection in your Weaviate instance.


```python
import weaviate.classes.config as wc

client.collections.delete("Listings")

listings = client.collections.create(
    name="Listings",
    description="AirBnb Listings",
    vectorizer_config=wc.Configure.Vectorizer.text2vec_aws(
        service="bedrock",
        region="us-east-1",
        model="cohere.embed-english-v3",
        vectorize_class_name=False,
    ),
    generative_config=wc.Configure.Generative.aws(
        service="bedrock",
        region="us-east-1",
        model="cohere.command-text-v14",
    ),
    properties=[
        wc.Property(
            name="description",
            data_type=wc.DataType.TEXT,
            description="The description of the apartment listing",
        ),
        wc.Property(
            name="host_name",
            data_type=wc.DataType.TEXT,
            description="The name of the host of the apartment listing",
            skip_vectorization=True,
            vectorize_property_name=False
        ),
        wc.Property(
            name="neighborhood_overview",
            data_type=wc.DataType.TEXT,
            description="The description of  the neighbourhood of the apartment listing",
        ),
        wc.Property(
            name="property_type",
            data_type=wc.DataType.TEXT,
            description="The type of property of the listing",
            skip_vectorization=True,
            vectorize_property_name=False
        )
    ]
)

listings = client.collections.get("Listings")
print(listings.config)
```

## Step 4: Ingest data into the Weaviate vector database

You can now add objects to Weaviate. You will be using a batch import process for maximum efficiency. Run the code below to import data. During the import, Weaviate will use the defined vectorizer to create a vector embedding for each object. The following code loads objects initializes a batch process, and adds objects to the target collection one by one.


```python
listings_to_add = [
    {
        "description": row["description"],
        "host_name": row["host_name"],
        "neighborhood_overview": row["neighborhood_overview"],
        "property_type": row["property_type"],
    } for _, row in df.iterrows()
]

response = listings.data.insert_many(listings_to_add)
print(response)
```

## Step 5: Retrieval-Augmented Generation to generate targeted advertisements

Finally, you can build a RAG pipeline by implementing a generative search query on your Weaviate instance. For this, you will first define a prompt template in the form of an f-string that can take in the user query (`{target_audience}`) directly and the additional context (`{{host_name}}`, `{{property_type}}`, `{{description}}`, `{{neighborhood_overview}}`) from the vector database at runtime.

Next, you will run a generative search query. This prompts the defined generative model with a prompt that is comprised of the user query as well as the retrieved data. The following query retrieves one listing object (`.with_limit(1)`) from the `Listings` collection that is most similar to the user query (`.with_near_text({"concepts": target_audience})`). Then the user query (`target_audience`) and the retrieved listings properties (`["description", "neighborhood", "host_name", "property_type"]`) are fed into the prompt template.


```python
def generate_targeted_ad(target_audience):
    generate_prompt = f"""You are a copywriter.
    Write short advertisement for the following vacation stay.
    Host: {{host_name}}
    Property type: {{property_type}}
    Description: {{description}}
    Neighborhood: {{neighborhood_overview}}
    Target audience: {target_audience}
    """

    result = listings.generate.near_text(
        query=target_audience,
        limit=1,
        single_prompt=generate_prompt
    )

    return result
```

Below, you can see that the results for the `target_audience = “Family with small children”`.


```python
result = generate_targeted_ad("Families with young children")
print(result.objects[0].generated)
```

Here is another example for an elderly couple.


```python
result = generate_targeted_ad("Elderly couple")
print(result.objects[0].generated)
```
