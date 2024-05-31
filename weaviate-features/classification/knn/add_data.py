import weaviate


training_data = [
    {"sentiment": "positive", "text": "I absolutely love this product!"},
    {"sentiment": "positive", "text": "This is the best day ever!"},
    {"sentiment": "positive", "text": "I can't believe how wonderful this is!"},
    {"sentiment": "positive", "text": "The weather is perfect for a picnic."},
    {"sentiment": "neutral", "text": "The weather today is neither hot nor cold."},
    {"sentiment": "neutral", "text": "I'm having a regular day at work."},
    {"sentiment": "neutral", "text": "The traffic was just as usual this morning."},
    {"sentiment": "neutral", "text": "I'm experiencing a typical weekend."},
    {"sentiment": "negative", "text": "Conflict with friends is bringing me down."},
    {"sentiment": "negative", "text": "The food at that restaurant was terrible!"},
    {"sentiment": "negative", "text": "My new pet is causing me endless trouble."},
    {"sentiment": "negative", "text": "My birthday party was a disaster!"},
    {"sentiment": "negative", "text": "I received bad news from a loved one."},
]

data_to_classify = [
    {"sentiment": "positive", "text": "I'm so grateful for all the support I received."},
    {"sentiment": "positive", "text": "This book is a masterpiece of literature."},
    {"sentiment": "positive", "text": "My family is amazing and supportive."},
    {"sentiment": "positive", "text": "I aced the exam; I'm on cloud nine!"},
    {"sentiment": "positive", "text": "My favorite team won the championship!"},
    {"sentiment": "positive", "text": "The sunset at the beach was breathtaking."},
    {"sentiment": "positive", "text": "I just got a promotion at work! So happy!"},
    {"sentiment": "positive", "text": "Spending time with friends always brightens my day."},
    {"sentiment": "positive", "text": "The food at that restaurant was delicious!"},
    {"sentiment": "positive", "text": "I'm in love with the new puppy we adopted."},
    {"sentiment": "positive", "text": "My birthday party was a blast!"},
    {"sentiment": "positive", "text": "I received a surprise gift from a loved one."},
    {"sentiment": "positive", "text": "Reuniting with an old friend was heartwarming."},
    {"sentiment": "positive", "text": "This movie made me laugh so hard."},
    {"sentiment": "positive", "text": "I reached a personal milestone today!"},
    {"sentiment": "neutral", "text": "My breakfast was plain, but satisfying."},
    {"sentiment": "neutral", "text": "The news headlines are uneventful today."},
    {"sentiment": "neutral", "text": "I'm taking a casual stroll in the park."},
    {"sentiment": "neutral", "text": "I have some standard chores to complete."},
    {"sentiment": "neutral", "text": "The movie I watched was neither good nor bad."},
    {"sentiment": "neutral", "text": "I'm attending a regular meeting this afternoon."},
    {"sentiment": "negative", "text": "I'm feeling really down and frustrated."},
    {"sentiment": "negative", "text": "This day has been a total disaster."},
    {"sentiment": "negative", "text": "I can't believe how awful this situation is."},
    {"sentiment": "negative", "text": "The rainy weather is ruining my plans."},
    {"sentiment": "negative", "text": "I'm so disappointed with the service I received."},
    {"sentiment": "negative", "text": "This book is a waste of time."},
    {"sentiment": "negative", "text": "My family is causing me a lot of stress."},
    {"sentiment": "negative", "text": "I failed the exam; I'm in the dumps."},
    {"sentiment": "negative", "text": "My favorite team lost the championship."},
    {"sentiment": "negative", "text": "The view from the window is depressing."},
    {"sentiment": "negative", "text": "I just got laid off at work! So upset!"},
]

sentiments = [
    "positive",
    "neutral",
    "negative"
]

client = weaviate.Client("http://localhost:8080")

# clear
client.schema.delete_all()

sentiments_class = {
            "class": 'Sentiment',
            "description": 'sentiment',  
            "properties": [
                {
                "dataType": [ 'text'],
                "description": 'name of sentiment',
                "name": 'name',
                }            
            ]
        }
if not client.schema.exists("Sentiment"):
    client.schema.create_class(sentiments_class)
    # add the categories as data objects
    print("#### ADDING Sentiments")
    client.batch.configure(batch_size=100)  # Configure batch
    with client.batch as batch:
        for item in sentiments:
            item_id = batch.add_data_object({"name": item}, "Sentiment", weaviate.util.generate_uuid5(item))
            print(item, item_id)

comment_class = {
            "class": 'Comment',
            "description": 'comment',  
            "properties": [
                {
                "name": 'body',
                "description": 'comment text',
                "dataType": [ 'text'],
                },
                {
                "name": 'trained_sentiment',
                "description": 'sentiment provided and mapped',
                "dataType": [ 'text'],
                },
                {
                "name": 'sentiment',
                "description": 'comment sentiment',
                "dataType": ["Sentiment"],
                }
            ]
        }
if not client.schema.exists("Comment"):
    client.schema.create_class(comment_class)
    print("#### ADDING Training Comments")
    client.batch.configure(batch_size=5)  # Configure batch
    # map relations to add after
    relations = []
    with client.batch as batch:
        relations = []
        for item in training_data:
            print(item)
            item_id = batch.add_data_object(
                {"body": item["text"]}, "Comment"
            )
            # save relations to add later
            relations.append(
                [item_id, weaviate.util.generate_uuid5(item["sentiment"])]
            )
    for relation in relations:
        print("ADDING RELATION", relation)
        # lets add the relation
        reference = client.data_object.reference.add(
        from_class_name="Comment",
        from_uuid=relation[0],
        from_property_name="sentiment",
        to_class_name="Sentiment",
        to_uuid=relation[1],
        )

    print("#### ADDING Non Classificated Comments")
    client.batch.configure(batch_size=5)  # Configure batch
    # map relations to add after
    relations = []
    with client.batch as batch:
        for item in data_to_classify:
            print(item)
            item_id = batch.add_data_object(
                {"body": item["text"]}, "Comment"
            )
            # save relations to add later
            relations.append(
                [item_id, weaviate.util.generate_uuid5(item["sentiment"])]
            )

#lets check our data
print("#### OUR DATA SHOULD BE IMPORTED")
client.query.get("Comment", "body sentiment{... on Sentiment{name}}").do()

# # now the fun part
classification_status = (
    client.classification.schedule()
    .with_type("knn")
    .with_class_name("Comment")
    .with_based_on_properties(["body"])
    .with_classify_properties(["sentiment"])
    .with_settings({"k": 3})
    .do()
)
print("CLASSIFICATION STATUS: ", classification_status)
# # just like that, you have your items categorized based on your training data
# lets check our data now
results = client.query.get(
    "Comment", "body sentiment{... on Sentiment{name}}"
).with_additional(
    "classification{basedOn classifiedFields completed id scope}"
).do()

print("DATA CLASSIFIED", results)