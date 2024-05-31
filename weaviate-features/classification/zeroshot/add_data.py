import weaviate

tickets = [
    {"body": "I cannot connect to the internet", "id":1},
    {"body": "I wat to put some text in a paper", "id":2},
    {"body": "My computer is slow, I cannot turn it on", "id":3},
    {"body": "I want to create some spreadsheets, but I cannot open the program", "id":4},
]

categories = [
    "Network",
    "Printing",
    "Hardware",
    "Software",
]

client = weaviate.Client("http://localhost:8080")

# clear
#client.schema.delete_all()

category_class = {
            "class": 'Category',
            "description": 'support ticket',  
            "properties": [
                {
                "dataType": [ 'text'],
                "description": 'name of category',
                "name": 'name',
                }            
            ]
        }
if not client.schema.exists("Category"):
    client.schema.create_class(category_class)
    # add the categories as data objects
    print("#### ADDING CATEGORIES")
    client.batch.configure(batch_size=100)  # Configure batch
    with client.batch as batch:
        for category in categories:
            cat_id = batch.add_data_object({"name": category}, "Category", weaviate.util.generate_uuid5(category))
            print(category, cat_id)

ticket_class = {
            "class": 'Ticket',
            "description": 'support ticket',  
            "properties": [
                {
                "name": 'body',
                "description": 'ticket text',
                "dataType": [ 'text'],
                },
                {
                "name": 'ticket_id',
                "description": 'ticket id',
                "dataType": [ 'number'],
                },                
                {
                "name": 'category',
                "description": 'ticket topic',
                "dataType": ["Category"],
                }
            ]
        }
if not client.schema.exists("Ticket"):
    client.schema.create_class(ticket_class)
    print("#### ADDING TICKETS")
    client.batch.configure(batch_size=100)  # Configure batch
    with client.batch as batch:
        for ticket in tickets:
            ticket_id = batch.add_data_object(
                {"body": ticket["body"], "ticket_id": ticket["id"]}, "Ticket", 
                weaviate.util.generate_uuid5(ticket["id"])
            )
            print(ticket, ticket_id)


# now the fun part
client.classification.schedule()\
            .with_type("zeroshot")\
            .with_class_name("Ticket")\
            .with_classify_properties(["category"])\
            .with_based_on_properties(["body"])\
            .do()

# just like that, you have your items categorized
results = client.query.get("Ticket", "body category{ ... on Category{name}}").do()
for ticket in results["data"]["Get"]["Ticket"]:
    print("#" * 10)
    print("Ticket:", ticket["body"])
    print("Category: ", ticket["category"][0]["name"])