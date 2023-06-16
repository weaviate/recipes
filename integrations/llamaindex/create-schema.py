import weaviate 

client = weaviate.Client("http://localhost:8080")

schema = {
   "classes": [
       {
           "class": "BlogPost",
           "description": "Blog post from the Weaviate website.",
           "vectorizer": "text2vec-openai",
           "moduleConfig": {
               "generative-openai": { 
                    "model": "gpt-3.5-turbo"
                }
           },
           "properties": [
               {
                  "name": "Content",
                  "dataType": ["text"],
                  "description": "Content from the blog post",
               }
            ]
        }
    ]
}

client.schema.delete_all()

client.schema.create(schema)

print("Schema was created.")